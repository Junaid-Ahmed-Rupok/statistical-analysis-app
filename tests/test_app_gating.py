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

    at.session_state["pro_active"] = True      # what a successful activation sets
    at.run(timeout=240)
    assert not [m for m in at.markdown if "is a Pro feature" in m.value]
