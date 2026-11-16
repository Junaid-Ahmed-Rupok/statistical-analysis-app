"""Drives the real app: unconfigured = everything open; configured = Pro features gated."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_to_ml(at):
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    for key in ("preprocess_btn", "run_stats_btn", "ml_train_btn"):
        next(b for b in at.button if b.key == key).click(); at.run(timeout=240)
    assert not at.exception, [e.value for e in at.exception]


def test_everything_unlocked_when_not_configured(monkeypatch):
    for n in ("CHECKOUT_URL", "LS_STORE_ID"):
        monkeypatch.delenv(n, raising=False)
    at = AppTest.from_file(APP, default_timeout=240).run()
    _run_to_ml(at)
    assert not [m for m in at.markdown if "is a Pro feature" in m.value]


def test_gated_when_configured_and_unlocks_with_valid_key(monkeypatch):
    monkeypatch.setenv("CHECKOUT_URL", "https://example.lemonsqueezy.com/buy/test")
    monkeypatch.setenv("LS_STORE_ID", "7")
    at = AppTest.from_file(APP, default_timeout=240).run()
    _run_to_ml(at)
    assert [m for m in at.markdown if "Batch prediction and model download is a Pro feature" in m.value]
    assert not [b for b in at.button if b.key == "pdf_btn"]
    assert [m for m in at.markdown if "Cleaned CSV download is a Pro feature" in m.value]

    at.session_state["pro_active"] = True      # what a successful activation sets
    at.run(timeout=240)
    assert not [m for m in at.markdown if "is a Pro feature" in m.value]


def test_bkash_key_unlocks_pro(monkeypatch):
    from modules import licensing
    secret = "a-long-random-secret-for-tests"
    for n in ("CHECKOUT_URL", "LS_STORE_ID"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("BKASH_NUMBER", "01700000000")
    monkeypatch.setenv("LICENSE_SECRET", secret)
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    next(b for b in at.button if b.key == "preprocess_btn").click(); at.run(timeout=240)
    assert [m for m in at.markdown if "Cleaned CSV download is a Pro feature" in m.value]
    assert any("Pay with bKash" in m.value for m in at.markdown)

    at.text_input(key="license_key_input").set_value("SP1-20991231-AAAA-BBBBBBBBBBBB"); at.run()
    next(b for b in at.button if b.key == "license_activate_btn").click(); at.run(timeout=240)
    assert not at.session_state["pro_active"]

    good = licensing.make_key(secret, days=7)
    at.text_input(key="license_key_input").set_value(good); at.run()
    next(b for b in at.button if b.key == "license_activate_btn").click(); at.run(timeout=240)
    assert at.session_state["pro_active"]
    assert not [m for m in at.markdown if "Cleaned CSV download is a Pro feature" in m.value]


def test_autopay_redeem_unlocks_pro(monkeypatch):
    from modules import autopay
    for k, v in {"BKASH_NUMBER": "01700000000", "LICENSE_SECRET": "a-long-random-secret-for-tests",
                 "AUTOPAY_URL": "https://example.com/exec", "AUTOPAY_TOKEN": "t"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CHECKOUT_URL", raising=False)
    monkeypatch.delenv("LS_STORE_ID", raising=False)
    monkeypatch.setattr(autopay, "claim", lambda trx, **kw: {"ok": True, "amount": 99})
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    next(b for b in at.button if b.key == "preprocess_btn").click(); at.run(timeout=240)
    assert not at.exception, [e.value for e in at.exception]
    at.text_input(key="autopay_trx_sb").set_value("9AB1C2D3EF"); at.run()
    next(b for b in at.button if b.key == "autopay_btn_sb").click(); at.run(timeout=240)
    assert at.session_state["pro_active"] and at.session_state["autopay_key"].startswith("SP1-")
    assert not [m for m in at.markdown if "is a Pro feature" in m.value]


def test_revoking_a_key_locks_an_active_session(monkeypatch):
    from modules import licensing
    secret = "a-long-random-secret-for-tests"
    for n in ("CHECKOUT_URL", "LS_STORE_ID", "REVOKED_KEYS"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("BKASH_NUMBER", "01700000000")
    monkeypatch.setenv("LICENSE_SECRET", secret)
    key = licensing.make_key(secret, days=7)
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    next(b for b in at.button if b.key == "preprocess_btn").click(); at.run(timeout=240)
    at.text_input(key="license_key_input").set_value(key); at.run()
    next(b for b in at.button if b.key == "license_activate_btn").click(); at.run(timeout=240)
    assert at.session_state["pro_active"]

    monkeypatch.setenv("REVOKED_KEYS", key)          # owner cancels the key
    at.run(timeout=240)                              # the user's very next click
    assert not at.session_state["pro_active"]
    assert [m for m in at.markdown if "is a Pro feature" in m.value]
