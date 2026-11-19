"""The 'Ask the assistant' window. Everything here fails softly: a problem never breaks the rest of StatsPro."""
from __future__ import annotations

import hashlib
import logging
import uuid

import streamlit as st

from . import assistant, licensing

log = logging.getLogger(__name__)

STARTERS = ("What is a t-test?", "What does a p-value mean?", "Explain my results in simple words")
PLAN_NAMES = {"free": "Free", "week": "Pro 7-day", "full": "Pro"}


def _identity(tier: str, pro_key: str | None) -> str:
    if tier != "free" and pro_key:
        return "key:" + hashlib.sha256(pro_key.strip().upper().encode()).hexdigest()[:16]
    if "assist_id" not in st.session_state:
        st.session_state["assist_id"] = uuid.uuid4().hex
    return "session:" + st.session_state["assist_id"]


def tier_for(has_pro: bool, pro_key: str | None) -> str:
    """free / week / full. Self-hosted copies without licensing configured get everything."""
    if not licensing.pro_enabled():
        return "full"
    if not has_pro:
        return "free"
    if pro_key and licensing.is_signed_key(pro_key):
        return licensing.key_plan(pro_key)
    return "full"


def _limit_card(tier: str, paused: bool) -> None:
    if paused:
        st.info("The assistant is taking a short break because many people are using it today. "
                "Please try again tomorrow. Everything else in StatsPro still works.")
        return
    nxt = ("Get **Pro 7-day** for 15 questions a day, or **Pro 30-day** for unlimited questions."
           if tier == "free" else "Get **Pro 30-day** for unlimited questions.")
    st.warning(f"You've used all of today's assistant questions. They reset at midnight (Dhaka time). {nxt}")
    if licensing.bkash_enabled() or licensing.get_setting("CHECKOUT_URL"):
        st.caption("Open **🔑 Pro license** in the sidebar to upgrade.")


def _answer(question: str, upload, tier: str, who: str, df, stats_results) -> None:
    key = licensing.get_setting("GROQ_API_KEY")
    history = st.session_state.get("assist_msgs", [])
    image_b64 = mime = None
    if upload is not None:
        try:
            image_b64, mime = assistant.prepare_image(upload.getvalue())
        except assistant.ImageError as exc:
            st.session_state["assist_notice"] = ("error", str(exc))
            return
    if not question.strip() and not image_b64:
        st.session_state["assist_notice"] = ("warning", "Type a question or attach a screenshot first.")
        return

    q = assistant.quota(_cap())
    status = q.consume(who, assistant.tier_limit(tier))
    if status != "ok":
        st.session_state["assist_notice"] = ("limit_paused" if status == "paused" else "limit", tier)
        return

    shown = question.strip() or "Please explain this screenshot."
    history.append({"role": "user", "content": shown + ("  \n📷 *screenshot attached*" if image_b64 else "")})
    with st.chat_message("user"):
        st.markdown(history[-1]["content"])
    messages = assistant.build_messages(
        shown, assistant.build_context(df, stats_results), [h for h in history[:-1]],
        assistant.find_terms(shown), image_b64, mime or "image/jpeg")
    text = ""
    try:
        with st.chat_message("assistant"):
            def pieces():
                nonlocal text
                for p in assistant.GroqAssistant(key).stream(messages, vision=bool(image_b64)):
                    text += p
                    yield p
            st.write_stream(pieces())
    except assistant.AssistantError as exc:
        q.refund(who)
        st.session_state["assist_notice"] = ("error", exc.message)
        history.pop()
        st.session_state["assist_msgs"] = history
        return
    except Exception:
        log.exception("assistant failure")
        q.refund(who)
        st.session_state["assist_notice"] = ("error", "Something went wrong. Please try again.")
        history.pop()
        st.session_state["assist_msgs"] = history
        return
    if not text.strip():
        q.refund(who)
        st.session_state["assist_notice"] = ("warning", "I couldn't produce an answer. Please try rephrasing.")
        history.pop()
        st.session_state["assist_msgs"] = history
        return
    history.append({"role": "assistant", "content": text})
    st.session_state["assist_msgs"] = history[-30:]


def _cap() -> int | None:
    try:
        return int(licensing.get_setting("ASSISTANT_DAILY_CAP") or 0) or None
    except ValueError:
        return None


@st.dialog("💬 StatsPro Assistant", width="large")
def _window(tier: str, who: str, df, stats_results) -> None:
    limit = assistant.tier_limit(tier)
    left = assistant.quota(_cap()).remaining(who, limit)
    plan = PLAN_NAMES.get(tier, "Free")
    st.caption(f"{plan} plan · " + ("unlimited questions" if left is None else f"{left} of {limit} questions left today") +
               " · Only column names and test results are shared with the AI, never your data rows.")
    kind, info = st.session_state.pop("assist_notice", (None, None))
    if kind in ("limit", "limit_paused"):
        _limit_card(tier if kind == "limit" else info, paused=kind == "limit_paused")
    elif kind:
        (st.error if kind == "error" else st.warning)(info)
    for m in st.session_state.get("assist_msgs", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if not st.session_state.get("assist_msgs"):
        st.markdown("Ask about any statistics term, your results, or what to do next. "
                    "You can also attach a **screenshot** of the part you don't understand.")
        for i, s in enumerate(STARTERS):
            if st.button(s, key=f"assist_starter_{i}"):
                st.session_state["assist_pending"] = s
                st.rerun(scope="fragment")
    pending = st.session_state.pop("assist_pending", None)
    if pending:
        _answer(pending, None, tier, who, df, stats_results)
        st.rerun(scope="fragment")
    with st.form("assist_form", clear_on_submit=True):
        question = st.text_area("Your question", key="assist_q", height=90,
                                placeholder="e.g. What is a t-test? Why is my p-value 0.03?")
        upload = st.file_uploader("Screenshot (optional)", type=["png", "jpg", "jpeg", "webp"], key="assist_img")
        sent = st.form_submit_button("Ask", type="primary")
    if sent:
        _answer(question, upload, tier, who, df, stats_results)
        st.rerun(scope="fragment")
    c1, c2 = st.columns(2)
    if st.session_state.get("assist_msgs") and c1.button("Clear conversation", key="assist_clear"):
        st.session_state["assist_msgs"] = []
        st.rerun(scope="fragment")
    c2.caption("AI can make mistakes. Double-check important decisions.")


def render_button(has_pro: bool, pro_key: str | None, df=None, stats_results=None) -> None:
    """Call from the sidebar. Shows nothing if the assistant is not configured."""
    if not assistant.available(licensing.get_setting("GROQ_API_KEY")):
        return
    tier = tier_for(has_pro, pro_key)
    who = _identity(tier, pro_key)
    if st.button("💬 Ask the assistant", key="assist_open", width="stretch", type="primary"):
        _window(tier, who, df, stats_results)
