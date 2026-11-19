import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from modules import assistant, assistant_ui, licensing


class Day:
    def __init__(self):
        self.d = dt.date(2026, 11, 19)

    def __call__(self):
        return self.d


def test_limits_per_plan():
    assert assistant.tier_limit("free") == 5 and assistant.tier_limit("week") == 15 and assistant.tier_limit("full") is None


def test_free_user_gets_five_then_is_stopped():
    q = assistant.Quota(clock=Day())
    assert [q.consume("a", 5) for _ in range(6)] == ["ok"] * 5 + ["limit"]
    assert q.remaining("a", 5) == 0 and q.consume("b", 5) == "ok"           # other users are unaffected


def test_week_plan_gets_fifteen_and_full_is_unlimited():
    q = assistant.Quota(clock=Day())
    assert [q.consume("w", 15) for _ in range(16)].count("ok") == 15
    assert all(q.consume("m", None) == "ok" for _ in range(200))
    assert q.remaining("m", None) is None


def test_counters_reset_the_next_day():
    day = Day()
    q = assistant.Quota(clock=day)
    for _ in range(5):
        q.consume("a", 5)
    assert q.consume("a", 5) == "limit"
    day.d += dt.timedelta(days=1)
    assert q.consume("a", 5) == "ok"


def test_global_cap_pauses_everyone_even_unlimited_users():
    q = assistant.Quota(global_cap=3, clock=Day())
    assert [q.consume(f"u{i}", None) for i in range(4)] == ["ok", "ok", "ok", "paused"]
    assert q.paused()


def test_refund_returns_a_question():
    q = assistant.Quota(clock=Day())
    q.consume("a", 5)
    q.refund("a")
    assert q.used("a") == 0


def test_tier_for_each_kind_of_key(monkeypatch):
    secret = "a-long-random-secret-for-tests"
    monkeypatch.setenv("BKASH_NUMBER", "01700000000")
    monkeypatch.setenv("LICENSE_SECRET", secret)
    assert assistant_ui.tier_for(False, None) == "free"
    assert assistant_ui.tier_for(True, licensing.make_key(secret, 7)) == "week"
    assert assistant_ui.tier_for(True, licensing.make_key(secret, 30)) == "full"
    assert assistant_ui.tier_for(True, "lemon-style-key-123") == "full"
    for n in ("BKASH_NUMBER", "LICENSE_SECRET"):
        monkeypatch.delenv(n)
    assert assistant_ui.tier_for(False, None) == "full"                     # licensing off: self-hosted copy


def test_glossary_finds_terms():
    assert assistant.find_terms("What is a t-test?")[0][0] == "t-test"
    assert {t for t, _ in assistant.find_terms("explain p value and anova")} == {"p-value", "anova"}
    assert assistant.find_terms("what is the weather") == []


def test_context_has_columns_and_results_but_never_raw_rows():
    df = pd.DataFrame({"salary": [111111.5, 222222.5, 333333.5], "dept": ["a", "b", "c"]})
    stats = {"normality": pd.DataFrame({"Column": ["salary"], "p-value": [0.031]})}
    ctx = assistant.build_context(df, stats, "Statistics")
    assert "salary" in ctx and "dept" in ctx and "3 rows" in ctx and "0.031" in ctx
    assert "111111" not in ctx and "222222" not in ctx


def test_messages_include_glossary_context_history_and_image():
    msgs = assistant.build_messages("What is a t-test?", "Dataset: 3 rows", [{"role": "user", "content": "hi"}],
                                    assistant.find_terms("t-test"), "QUJD", "image/jpeg")
    assert "Trusted definitions" in msgs[0]["content"] and "Dataset: 3 rows" in msgs[0]["content"]
    assert msgs[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    plain = assistant.build_messages("q", "", [], [])
    assert plain[-1] == {"role": "user", "content": "q"}


def png(size=(3000, 2000), mode="RGB"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_screenshots_are_validated_and_shrunk():
    import base64, io
    from PIL import Image
    b64, mime = assistant.prepare_image(png())
    assert mime == "image/jpeg" and max(Image.open(io.BytesIO(base64.b64decode(b64))).size) <= 1600
    assert assistant.prepare_image(png((50, 50), "RGBA"))[1] == "image/jpeg"
    for bad in (b"", b"not an image", b"x" * (9 * 1024 * 1024)):
        with pytest.raises(assistant.ImageError):
            assistant.prepare_image(bad)


# ── model client with a fake Groq ───────────────────────────────────────
class FakeGroq:
    def __init__(self, models, pieces=("Hello", " there"), fail=None):
        self.models, self.pieces, self.fail, self.calls = models, pieces, fail, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models_api = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in self.models]))

    @property
    def models_attr(self):
        return self.models_api

    def _create(self, model, **kw):
        self.calls.append(model)
        if self.fail and model in self.fail:
            raise self.fail[model]
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in self.pieces])


def make(models, **kw):
    fake = FakeGroq(models, **kw)
    fake.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in models]))
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=fake._create))
    return assistant.GroqAssistant("k", client=fake), fake


def test_text_question_uses_a_text_model():
    a, fake = make(["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"])
    assert "".join(a.stream([{"role": "user", "content": "x"}])) == "Hello there"
    assert fake.calls == ["llama-3.3-70b-versatile"]


def test_screenshot_question_uses_a_vision_model():
    a, fake = make(["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"])
    list(a.stream([{"role": "user", "content": "x"}], vision=True))
    assert fake.calls == ["meta-llama/llama-4-scout-17b-16e-instruct"]


def test_no_vision_model_gives_a_clear_message():
    a, _ = make(["llama-3.3-70b-versatile"])
    with pytest.raises(assistant.AssistantError, match="Screenshot"):
        list(a.stream([{"role": "user", "content": "x"}], vision=True))


def test_rate_limit_falls_back_to_next_model_and_errors_are_friendly():
    import groq, httpx
    req = httpx.Request("POST", "https://api.groq.com")
    limited = groq.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
    a, fake = make(["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], fail={"llama-3.3-70b-versatile": limited})
    assert "".join(a.stream([{"role": "user", "content": "x"}])) == "Hello there"
    assert fake.calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    a, _ = make(["llama-3.3-70b-versatile"], fail={"llama-3.3-70b-versatile": limited})
    with pytest.raises(assistant.AssistantError, match="busy"):
        list(a.stream([{"role": "user", "content": "x"}]))
    bad = groq.AuthenticationError("no", response=httpx.Response(401, request=req), body=None)
    a, _ = make(["llama-3.3-70b-versatile"], fail={"llama-3.3-70b-versatile": bad})
    with pytest.raises(assistant.AssistantError) as e:
        list(a.stream([{"role": "user", "content": "x"}]))
    assert "key" not in e.value.message.lower()                              # never leaks key details to users
