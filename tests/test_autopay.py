import io
import json
import urllib.error

import pytest

from modules import autopay, licensing

SECRET = "a-long-random-secret-for-tests"


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def opener(payload):
    def f(req, timeout=None):
        f.last = req.full_url
        return _Resp(json.dumps(payload).encode())
    return f


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    for k, v in {"BKASH_NUMBER": "01700000000", "LICENSE_SECRET": SECRET,
                 "AUTOPAY_URL": "https://script.google.com/macros/s/X/exec", "AUTOPAY_TOKEN": "claimtoken",
                 "BKASH_PRICE_7": "50", "BKASH_PRICE_30": "99"}.items():
        monkeypatch.setenv(k, v)


def test_plan_for_amount():
    assert autopay.plan_for_amount(49) is None
    assert autopay.plan_for_amount(50) == 7
    assert autopay.plan_for_amount(98.5) == 7
    assert autopay.plan_for_amount(99) == 30
    assert autopay.plan_for_amount(500) == 30


def test_enabled_needs_everything(monkeypatch):
    assert autopay.enabled()
    monkeypatch.delenv("AUTOPAY_TOKEN")
    assert not autopay.enabled()


def test_redeem_issues_valid_key_for_amount():
    ok, msg, key = autopay.redeem("9ab1 c2d3-ef", opener=opener({"ok": True, "amount": 99}))
    assert ok and licensing.verify_signed_key(key, SECRET)[0]
    assert "30 days" in msg


def test_redeem_sends_normalised_trx_and_token():
    o = opener({"ok": True, "amount": 50})
    autopay.redeem("9ab1c2d3ef", opener=o)
    assert "trx=9AB1C2D3EF" in o.last and "token=claimtoken" in o.last and "action=claim" in o.last


@pytest.mark.parametrize("err", ["not_found", "already_used", "expired", "bad_token"])
def test_redeem_failures_give_friendly_messages_and_no_key(err):
    ok, msg, key = autopay.redeem("9AB1C2D3EF", opener=opener({"ok": False, "error": err}))
    assert not ok and key is None and msg


def test_underpayment_gets_no_key():
    ok, msg, key = autopay.redeem("9AB1C2D3EF", opener=opener({"ok": True, "amount": 20}))
    assert not ok and key is None and "less than" in msg


@pytest.mark.parametrize("bad", ["", "abc", "!!!!!!!!!!", "x" * 40])
def test_malformed_trx_never_reaches_network(bad):
    def boom(*a, **k):
        raise AssertionError("network used")
    ok, _, key = autopay.redeem(bad, opener=boom)
    assert not ok and key is None


def test_network_error_raises_autopay_error():
    def down(req, timeout=None):
        raise urllib.error.URLError("down")
    with pytest.raises(autopay.AutopayError):
        autopay.redeem("9AB1C2D3EF", opener=down)


def test_garbage_amount_gets_no_key():
    ok, _, key = autopay.redeem("9AB1C2D3EF", opener=opener({"ok": True, "amount": "lots"}))
    assert not ok and key is None
