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
    with st.chat_message("user", avatar="🧑"):
        st.markdown(history[-1]["content"])
    messages = assistant.build_messages(
        shown, assistant.build_context(df, stats_results), [h for h in history[:-1]],
        assistant.find_terms(shown), image_b64, mime or "image/jpeg")
    text = ""
    try:
        with st.chat_message("assistant", avatar="📊"):
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


def _starters(df, stats_results) -> list[str]:
    if stats_results:
        return ["Explain my results in simple words", "Which result matters most?", "What should I do next?",
                "What is a p-value?"]
    if df is not None:
        return ["Is my data ready for analysis?", "Which test should I use?", "What is a t-test?",
                "How do I handle missing values?"]
    return ["What is a t-test?", "What does a p-value mean?", "Which test should I use?", "What can StatsPro do?"]


def _pick() -> None:
    st.session_state["assist_pending"] = st.session_state.get("assist_pill")
    st.session_state["assist_pill"] = None


def _header(tier: str, left, limit) -> None:
    plan = PLAN_NAMES.get(tier, "Free")
    if left is None:
        pill, pct, tone = "Unlimited questions", 100, "#34D399"
    else:
        pct = int(100 * left / limit) if limit else 0
        pill, tone = f"{left} of {limit} questions left today", ("#34D399" if left > 1 else "#fbbf24" if left == 1 else "#f87171")
    st.markdown(f"""
    <div class="sp-assist-head">
      <div><div class="sp-assist-title">📊 StatsPro Assistant</div>
      <div class="sp-assist-sub">Ask about any statistics term, your results, or attach a screenshot.</div></div>
      <div class="sp-assist-plan"><span>{plan}</span><b style="color:{tone}">{pill}</b>
      <div class="sp-assist-bar"><i style="width:{pct}%;background:{tone}"></i></div></div>
    </div>""", unsafe_allow_html=True)


@st.dialog("Assistant", width="large")
def _window(tier: str, who: str, df, stats_results) -> None:
    limit = assistant.tier_limit(tier)
    left = assistant.quota(_cap()).remaining(who, limit)
    _header(tier, left, limit)

    kind, info = st.session_state.pop("assist_notice", (None, None))
    if kind in ("limit", "limit_paused"):
        _limit_card(tier if kind == "limit" else info, paused=kind == "limit_paused")
    elif kind:
        (st.error if kind == "error" else st.warning)(info)

    msgs = st.session_state.get("assist_msgs", [])
    if not msgs:
        st.markdown("**Hi! I'm here to help you understand your analysis.** Tap a question, or type your own below. "
                    "To ask about a part of the screen, attach a screenshot with the 📎 button.")
        st.pills("Try asking", _starters(df, stats_results), key="assist_pill", on_change=_pick,
                 label_visibility="collapsed")
    for m in msgs:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📊"):
            st.markdown(m["content"])

    pending = st.session_state.pop("assist_pending", None)
    if pending:
        _answer(pending, None, tier, who, df, stats_results)
        st.rerun(scope="fragment")

    box = st.chat_input("Ask anything, or attach a screenshot…", key="assist_box", accept_file=True,
                        file_type=["png", "jpg", "jpeg", "webp"], max_chars=assistant.MAX_QUESTION_CHARS)
    if box:
        files = getattr(box, "files", None) or []
        _answer(getattr(box, "text", "") or "", files[0] if files else None, tier, who, df, stats_results)
        st.rerun(scope="fragment")
    c1, c2 = st.columns([1, 2])
    if msgs and c1.button("Clear conversation", key="assist_clear", icon=":material/delete:"):
        st.session_state["assist_msgs"] = []
        st.rerun(scope="fragment")
    c2.caption("AI can make mistakes. Double-check important decisions. Your data rows are never shared.")


def render_button(has_pro: bool, pro_key: str | None, df=None, stats_results=None) -> None:
    """Call from the sidebar. Shows nothing if the assistant is not configured."""
    if not assistant.available(licensing.get_setting("GROQ_API_KEY")):
        return
    tier = tier_for(has_pro, pro_key)
    who = _identity(tier, pro_key)
    if st.button("💬 Ask the assistant", key="assist_open", width="stretch", type="primary", help="Plain-language help with statistics, your results, and screenshots"):
        _window(tier, who, df, stats_results)
