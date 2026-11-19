"""StatsPro AI assistant: plain-language help for statistics, grounded in the user's current analysis.

Design rules
  * Only a short summary of the analysis goes to the model (column names, types, test results).
    Raw rows are never sent.
  * Basic definitions come from a built-in glossary, so they do not depend on the model's memory.
  * Daily question limits: free 5, short (7-day) plan 15, month/lifetime unlimited. Counted per day (Dhaka time).
  * A global daily cap protects the shared free Groq key. When it is reached the assistant pauses; StatsPro works on.
  * Every failure becomes a friendly message. The assistant can never break the rest of the app.

Configuration (Streamlit secrets or environment variables): GROQ_API_KEY (required),
ASSISTANT_DAILY_CAP (optional, default 500 questions per day across all users).
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import logging
import re
import threading
from typing import Iterator, Optional

log = logging.getLogger(__name__)

# ── limits ──────────────────────────────────────────────────────────────
FREE_LIMIT = 5
WEEK_LIMIT = 15
DEFAULT_GLOBAL_CAP = 500
_DHAKA = dt.timezone(dt.timedelta(hours=6))
MAX_QUESTION_CHARS = 1500
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1600


def tier_limit(tier: str) -> Optional[int]:
    """Questions per day for a tier ('free', 'week', 'full'). None means unlimited."""
    return {"free": FREE_LIMIT, "week": WEEK_LIMIT}.get(tier)


def today() -> dt.date:
    return dt.datetime.now(_DHAKA).date()


class Quota:
    """In-memory daily counters. They live as long as the app process, which is fine for a daily limit."""

    def __init__(self, global_cap: int = DEFAULT_GLOBAL_CAP, clock=today):
        self.global_cap, self._clock = global_cap, clock
        self._lock = threading.Lock()
        self._day = clock()
        self._used: dict[str, int] = {}
        self._total = 0

    def _roll(self) -> None:
        d = self._clock()
        if d != self._day:
            self._day, self._used, self._total = d, {}, 0

    def used(self, who: str) -> int:
        with self._lock:
            self._roll()
            return self._used.get(who, 0)

    def remaining(self, who: str, limit: Optional[int]) -> Optional[int]:
        if limit is None:
            return None
        return max(0, limit - self.used(who))

    def paused(self) -> bool:
        with self._lock:
            self._roll()
            return self._total >= self.global_cap

    def consume(self, who: str, limit: Optional[int]) -> str:
        """Try to use one question. Returns 'ok', 'limit' (this user is out) or 'paused' (global cap reached)."""
        with self._lock:
            self._roll()
            if self._total >= self.global_cap:
                return "paused"
            if limit is not None and self._used.get(who, 0) >= limit:
                return "limit"
            self._used[who] = self._used.get(who, 0) + 1
            self._total += 1
            return "ok"

    def refund(self, who: str) -> None:
        """Give the question back when the model failed, so an error never costs the user a question."""
        with self._lock:
            self._roll()
            if self._used.get(who, 0) > 0:
                self._used[who] -= 1
                self._total = max(0, self._total - 1)


# ── glossary ────────────────────────────────────────────────────────────
GLOSSARY: dict[str, tuple[tuple[str, ...], str]] = {
    "t-test": (("t-test", "t test", "ttest", "student's t"),
               "A t-test checks whether the averages of two groups differ by more than chance would explain. "
               "A small p-value (usually below 0.05) says the difference is unlikely to be luck."),
    "p-value": (("p-value", "p value", "pvalue"),
                "The p-value is the chance of seeing a result at least this extreme if nothing real were going on. "
                "Below 0.05 is usually called 'statistically significant'. It does not tell you how big or important the effect is."),
    "significance": (("significant", "significance", "alpha"),
                     "'Significant' means the p-value is below the chosen cut-off (alpha, usually 0.05). "
                     "It means 'unlikely to be chance', not 'important'."),
    "anova": (("anova", "analysis of variance"),
              "ANOVA compares the averages of three or more groups at once. A small p-value says at least one group differs; "
              "it does not say which one (that needs a follow-up test)."),
    "mann-whitney": (("mann-whitney", "mann whitney", "wilcoxon rank"),
                     "Mann-Whitney U compares two groups without assuming a bell-shaped (normal) distribution. "
                     "Use it instead of a t-test when the data are skewed or ranked."),
    "levene": (("levene",),
               "Levene's test checks whether groups have similar spread (variance). A small p-value means the spreads differ, "
               "which matters because many tests assume they are equal."),
    "normality": (("normal distribution", "normality", "shapiro", "anderson", "d'agostino", "bell curve"),
                  "Normality tests ask whether data look bell-shaped. A small p-value (below 0.05) means the data are probably "
                  "NOT normal. With big samples tiny deviations become 'significant', so also look at the histogram."),
    "correlation": (("correlation", "pearson", "spearman", "correlated"),
                    "Correlation measures how strongly two numbers move together, from -1 to +1. Near 0 means no linear link. "
                    "Correlation does not prove that one causes the other."),
    "effect size": (("effect size", "cohen", "eta squared", "eta-squared", "cramer", "cramér"),
                    "Effect size says how big a difference or link is, which the p-value does not. "
                    "Rough labels: negligible, small, medium, large."),
    "chi-square": (("chi-square", "chi square", "chi2", "χ²"),
                   "The chi-square test checks whether two categories are related (for example, gender and product choice). "
                   "A small p-value suggests they are related."),
    "vif": (("vif", "multicollinearity", "variance inflation"),
            "VIF shows whether input columns overlap too much. Below 5 is fine, 5-10 is a warning, above 10 means a column is "
            "largely repeating the others, which makes regression results unstable."),
    "regression": (("regression", "ols", "r-squared", "r squared", "r2", "coefficient"),
                   "Regression predicts one number from others. R-squared says how much of the variation the model explains "
                   "(0 to 1). Each coefficient says how much the outcome changes when that input rises by one unit."),
    "fdr": (("fdr", "false discovery", "benjamini", "multiple testing", "multiple comparison"),
            "When many tests are run, some look significant by luck. FDR (false discovery rate) correction adjusts the p-values "
            "to account for that. 'Sig. after FDR' is the safer column to trust."),
    "outlier": (("outlier", "outliers", "iqr"),
                "An outlier is a value far from the rest. It may be a typo or a real extreme. Check it before deleting, "
                "because outliers can distort averages and correlations."),
    "missing values": (("missing value", "missing values", "imputation", "impute", "null", "nan"),
                       "Missing values are blanks in the data. Common fixes: fill with the median or most frequent value, "
                       "or drop rows or columns that are mostly empty."),
    "skewness": (("skew", "skewness", "kurtosis"),
                 "Skewness says if a distribution leans to one side (a long tail). Kurtosis says how heavy the tails are. "
                 "Strong skew often means a rank-based test is safer."),
    "confidence interval": (("confidence interval", " ci ", "95% ci"),
                            "A 95% confidence interval is a range that would contain the true value in 95 out of 100 repeats of the study. "
                            "A narrow range means a precise estimate."),
    "standard deviation": (("standard deviation", "std", "variance"),
                           "Standard deviation shows how spread out values are around the average. "
                           "A small value means values cluster near the mean."),
    "overfitting": (("overfit", "overfitting", "cross-validation", "cross validation"),
                    "Overfitting means a model memorised the training data and does poorly on new data. "
                    "Cross-validation tests the model on data it has not seen to catch this."),
    "rmse": (("rmse", "mae", "mse", "accuracy", "f1", "precision", "recall", "roc", "auc"),
             "These are model scores. For predicting numbers, RMSE and MAE measure the average error (lower is better). "
             "For classes, accuracy, F1 and AUC (higher is better) measure how often the model is right."),
    "data leakage": (("leakage", "data leak"),
                     "Data leakage is when the model sees information it should not have, such as the answer hidden in a column. "
                     "It gives unrealistically good scores that collapse on real data."),
    "sample size": (("sample size", "how many rows", "enough data", "power"),
                    "Bigger samples give steadier results and let you detect smaller effects. "
                    "With very small samples, even real differences may look non-significant."),
}


def find_terms(text: str, limit: int = 3) -> list[tuple[str, str]]:
    """Glossary entries mentioned in the text (term, definition)."""
    low = f" {text.lower()} "
    found = []
    for term, (aliases, definition) in GLOSSARY.items():
        if any(a in low for a in aliases):
            found.append((term, definition))
        if len(found) >= limit:
            break
    return found


# ── context (a summary only, never raw rows) ────────────────────────────
def _table_text(df, max_rows: int = 8, max_cols: int = 8) -> str:
    try:
        d = df.head(max_rows)
        d = d.iloc[:, :max_cols]
        return d.round(4).to_string(index=False)[:1800]
    except Exception:
        return ""


def build_context(df=None, stats_results: Optional[dict] = None, page: str = "") -> str:
    """Short description of what the user is looking at. Only column names, types, sizes and test results."""
    lines = []
    if page:
        lines.append(f"The user is on this part of StatsPro: {page}.")
    if df is not None:
        try:
            cols = ", ".join(f"{c} ({str(t)})" for c, t in list(df.dtypes.items())[:40])
            more = f" and {len(df.columns) - 40} more columns" if len(df.columns) > 40 else ""
            lines.append(f"Dataset: {len(df):,} rows, {len(df.columns)} columns. Columns: {cols}{more}.")
        except Exception:
            pass
    if stats_results:
        for name, value in list(stats_results.items())[:8]:
            try:
                if hasattr(value, "empty"):
                    if value.empty:
                        continue
                    lines.append(f"Result table '{name}' (first rows):\n{_table_text(value)}")
                elif isinstance(value, dict) and value:
                    brief = {k: v for k, v in list(value.items())[:12] if isinstance(v, (int, float, str))}
                    if brief:
                        lines.append(f"Result '{name}': {brief}")
            except Exception:
                continue
    return "\n".join(lines)[:6000]


SYSTEM_PROMPT = (
    "You are the StatsPro assistant, a friendly, patient statistics tutor inside a statistical analysis app. "
    "The user may be a complete beginner, so avoid jargon; if you must use a term, explain it in a few words. "
    "Reply in the user's language (English, Bangla or Banglish).\n"
    "FORMAT: start with a one-sentence direct answer in **bold**. Then 2 to 5 short sentences of explanation, using a tiny "
    "everyday example when it helps. If the user has results, finish with a line starting 'Next step:' that says one concrete "
    "thing to do in StatsPro. Never write more than about 150 words. Use short bullet points only for lists.\n"
    "RULES: (1) Use only numbers that appear in the context or in the user's message; never invent results. "
    "(2) If something is not in the context, or an image is unclear, say so and ask what is missing. "
    "(3) For screenshots: say what the image shows, explain the part the user is confused about, then give the plain-language meaning. "
    "(4) You cannot see the user's raw data, only the summary below. (5) For decisions that matter (medical, legal, financial, "
    "research claims) remind the user to double-check with an expert. (6) Stay on statistics, data analysis and StatsPro; "
    "politely decline other topics.\n"
)


def build_messages(question: str, context: str, history: list[dict], glossary: list[tuple[str, str]],
                   image_b64: Optional[str] = None, image_mime: str = "image/jpeg") -> list[dict]:
    system = SYSTEM_PROMPT
    if glossary:
        system += "\nTrusted definitions you may rely on:\n" + "\n".join(f"- {t}: {d}" for t, d in glossary)
    if context:
        system += "\nCurrent analysis (summary only):\n" + context
    messages = [{"role": "system", "content": system}]
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": str(m["content"])[:1200]})
    question = (question or "").strip()[:MAX_QUESTION_CHARS] or "Please explain what this screenshot shows."
    if image_b64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
        ]})
    else:
        messages.append({"role": "user", "content": question})
    return messages


# ── screenshots ─────────────────────────────────────────────────────────
class ImageError(ValueError):
    pass


def prepare_image(data: bytes) -> tuple[str, str]:
    """Validate and shrink a screenshot. Returns (base64, mime). Raises ImageError with a friendly message."""
    if not data:
        raise ImageError("That image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("That image is too large. Please crop the screenshot or use one under 8 MB.")
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ImageError("I couldn't read that image. Please upload a PNG or JPG screenshot.") from None
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, "white")
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)
    return base64.b64encode(out.getvalue()).decode(), "image/jpeg"


# ── model client ────────────────────────────────────────────────────────
# Groq changes its model line-up often, so models are chosen from the LIVE list by family, best first.
TEXT_PREFERENCE = ("gpt-oss-120b", "llama-3.3-70b", "qwen", "llama-3.1-8b", "gpt-oss-20b", "llama")
VISION_PREFERENCE = ("qwen", "llama-4", "vision", "scout", "maverick")
FALLBACK_TEXT = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
FALLBACK_VISION = ("qwen/qwen3.8-27b",)
_NOT_CHAT = ("whisper", "guard", "safeguard", "tts", "orpheus", "playai", "embed", "transcribe", "compound", "allam")


def order_by(ids: list[str], preference: tuple[str, ...]) -> list[str]:
    chat = [m for m in ids if not any(bad in m.lower() for bad in _NOT_CHAT)]
    ranked = []
    for rank, pattern in enumerate(preference):
        ranked += [(rank, m) for m in sorted(chat) if pattern in m.lower() and m not in {r[1] for r in ranked}]
    return [m for _, m in ranked]


class _ThinkFilter:
    """Hides <think>...</think> reasoning that some models stream inside the answer."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, piece: str) -> str:
        self.buf += piece
        out = ""
        while True:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[end + 8:], False
            else:
                start = self.buf.find("<think>")
                if start >= 0:
                    out += self.buf[:start]
                    self.buf, self.inside = self.buf[start + 7:], True
                    continue
                keep = 0
                for n in range(min(6, len(self.buf)), 0, -1):          # a partial "<think" may arrive in pieces
                    if "<think>".startswith(self.buf[-n:]):
                        keep = n
                        break
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out

    def flush(self) -> str:
        return "" if self.inside else self.buf


class AssistantError(Exception):
    """Friendly, user-facing failure. `refund` says the user should not lose a question."""

    def __init__(self, message: str, refund: bool = True):
        super().__init__(message)
        self.message, self.refund = message, refund


class GroqAssistant:
    def __init__(self, api_key: str, client=None):
        self.api_key = api_key
        self._client = client
        self.last_model = ""

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=45.0, max_retries=1)
        return self._client

    def available_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []

    def pick_models(self, vision: bool) -> list[str]:
        live = self.available_models()
        if live:
            ordered = order_by(live, VISION_PREFERENCE if vision else TEXT_PREFERENCE)
            if ordered:
                return ordered[:3]
            if vision:
                raise AssistantError("Screenshot reading isn't available right now. Please type your question instead.")
        return list(FALLBACK_VISION if vision else FALLBACK_TEXT)

    def stream(self, messages: list[dict], vision: bool = False, max_tokens: int = 900) -> Iterator[str]:
        import groq

        last: Optional[AssistantError] = None
        for name in self.pick_models(vision):
            started = False
            think = _ThinkFilter()
            try:
                response = self.client.chat.completions.create(
                    model=name, messages=messages, temperature=0.3, max_tokens=max_tokens, stream=True)
                self.last_model = name
                for event in response:
                    piece = event.choices[0].delta.content if event.choices else None
                    if piece:
                        visible = think.feed(piece)
                        if visible:
                            started = True
                            yield visible
                tail = think.flush()
                if tail:
                    yield tail
                return
            except groq.AuthenticationError:
                raise AssistantError("The assistant is not set up correctly right now. Please try again later.") from None
            except groq.RateLimitError:
                last = AssistantError("The assistant is very busy right now. Please try again in a minute.")
                if started:
                    raise last from None
            except groq.APIConnectionError:
                raise AssistantError("Could not reach the assistant. Check your internet connection and try again.") from None
            except (groq.NotFoundError, groq.BadRequestError) as exc:
                if started:
                    raise AssistantError("The answer was cut off. Please ask again.") from None
                last = AssistantError("I couldn't read that screenshot. Try a smaller, clearer crop." if vision
                                      else "The assistant couldn't process that question. Try rephrasing it.")
                log.warning("Model %s rejected the request: %s", name, str(exc)[:200])
                continue                                  # try the next model
            except groq.APIStatusError as exc:
                raise AssistantError(f"The assistant had a problem ({exc.status_code}). Please try again shortly.") from None
        raise last or AssistantError("The assistant is unavailable right now. Please try again shortly.")


def available(api_key: str) -> bool:
    return bool((api_key or "").strip())


_QUOTA: Optional[Quota] = None


def quota(global_cap: Optional[int] = None) -> Quota:
    """The one shared counter for this app process."""
    global _QUOTA
    if _QUOTA is None:
        _QUOTA = Quota(global_cap or DEFAULT_GLOBAL_CAP)
    return _QUOTA
