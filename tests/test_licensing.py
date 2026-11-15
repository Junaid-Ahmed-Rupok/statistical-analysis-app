import io
import json
import urllib.error

import pytest

from modules import licensing


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _opener(payload, status=200):
    def f(req, timeout=None):
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, "x", {}, io.BytesIO(json.dumps(payload).encode()))
        return _Resp(json.dumps(payload).encode())
    return f


KEY = "38b1460a-5104-4067-a91d-77b872934d51"


def test_valid_key_for_our_store():
    ok, msg = licensing.verify_key(KEY, store_id="7", product_id="9",
        opener=_opener({"valid": True, "meta": {"store_id": 7, "product_id": 9}}))
    assert ok


def test_key_from_another_store_is_rejected():
    ok, _ = licensing.verify_key(KEY, store_id="7",
        opener=_opener({"valid": True, "meta": {"store_id": 99, "product_id": 9}}))
    assert not ok


def test_invalid_key_reports_server_message():
    ok, msg = licensing.verify_key(KEY, opener=_opener({"valid": False, "error": "license_key not found."}, status=404))
    assert not ok and "not found" in msg


def test_network_failure_raises_instead_of_looking_invalid():
    def boom(req, timeout=None): raise urllib.error.URLError("down")
    with pytest.raises(licensing.LicenseError):
        licensing.verify_key(KEY, opener=boom)


def test_garbage_key_never_hits_the_network():
    def boom(req, timeout=None): raise AssertionError("network used")
    assert licensing.verify_key("abc", opener=boom)[0] is False


def test_licensing_is_off_until_configured(monkeypatch):
    for n in ("CHECKOUT_URL", "LS_STORE_ID"):
        monkeypatch.delenv(n, raising=False)
    assert licensing.pro_enabled() is False
    monkeypatch.setenv("CHECKOUT_URL", "https://x.lemonsqueezy.com/buy/abc")
    assert licensing.pro_enabled() is False          # needs the store id too
    monkeypatch.setenv("LS_STORE_ID", "7")
    assert licensing.pro_enabled() is True
