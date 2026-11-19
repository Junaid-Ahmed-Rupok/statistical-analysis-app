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


# ── signed offline keys (bKash) ─────────────────────────────────────────
import datetime as dt

SECRET = "a-long-random-secret-for-tests"


def test_signed_key_roundtrip_and_expiry():
    today = dt.date(2026, 11, 16)
    k = licensing.make_key(SECRET, days=7, today=today)
    assert licensing.verify_signed_key(k, SECRET, today=today)[0]
    assert licensing.verify_signed_key(k, SECRET, today=dt.date(2026, 11, 23))[0]      # last valid day
    ok, msg = licensing.verify_signed_key(k, SECRET, today=dt.date(2026, 11, 24))
    assert not ok and "expired" in msg


def test_forged_or_tampered_keys_rejected():
    k = licensing.make_key(SECRET, days=7)
    assert not licensing.verify_signed_key(k, "another-long-random-secret-x")[0]
    parts = k.split("-")
    parts[1] = "20991231"                                     # try to extend the expiry
    assert not licensing.verify_signed_key("-".join(parts), SECRET)[0]
    assert not licensing.verify_signed_key("SP1-20991231-AAAA-BBBBBBBBBBBB", SECRET)[0]
    assert not licensing.verify_signed_key("", SECRET)[0]
    assert not licensing.verify_signed_key(k, "")[0]


def test_lifetime_revocation_and_case_insensitivity():
    k = licensing.make_key(SECRET, days=None)
    assert licensing.verify_signed_key(k.lower(), SECRET)[0]
    assert not licensing.verify_signed_key(k, SECRET, revoked=[k])[0]


def test_weak_secret_and_bad_days_refused():
    with pytest.raises(ValueError):
        licensing.make_key("short")
    with pytest.raises(ValueError):
        licensing.make_key(SECRET, days=0)


def test_keys_are_unique():
    assert len({licensing.make_key(SECRET, 7) for _ in range(50)}) == 50


def test_bkash_settings_switch_licensing_on(monkeypatch):
    for n in ("CHECKOUT_URL", "LS_STORE_ID", "BKASH_NUMBER", "LICENSE_SECRET"):
        monkeypatch.delenv(n, raising=False)
    assert not licensing.pro_enabled()
    monkeypatch.setenv("BKASH_NUMBER", "01700000000")
    assert not licensing.pro_enabled()                         # needs the secret too
    monkeypatch.setenv("LICENSE_SECRET", SECRET)
    assert licensing.pro_enabled() and licensing.bkash_enabled() and not licensing.lemon_enabled()
    k = licensing.make_key(SECRET, 7)
    assert licensing.verify_any(k)[0]
    monkeypatch.setenv("REVOKED_KEYS", k)
    assert not licensing.verify_any(k)[0]


# ── plan letter inside SP2 keys ─────────────────────────────────────────
def test_plan_is_signed_into_the_key():
    week, month, life = (licensing.make_key(SECRET, 7), licensing.make_key(SECRET, 30), licensing.make_key(SECRET, None))
    assert week.startswith("SP2-W") and month.startswith("SP2-M") and life.startswith("SP2-L")
    assert [licensing.key_plan(k) for k in (week, month, life)] == ["week", "full", "full"]
    assert all(licensing.verify_signed_key(k, SECRET)[0] for k in (week, month, life))


def test_cannot_upgrade_a_week_key_by_editing_the_plan_letter():
    parts = licensing.make_key(SECRET, 7).split("-")
    parts[1] = "M" + parts[1][1:]
    assert not licensing.verify_signed_key("-".join(parts), SECRET)[0]


def test_old_sp1_keys_still_work_and_count_as_full():
    import base64, hashlib, hmac
    expiry = (dt.date.today() + dt.timedelta(days=5)).strftime("%Y%m%d")
    mac = hmac.new(SECRET.encode(), f"SP1|{expiry}|ABCD".encode(), hashlib.sha256).digest()
    old = f"SP1-{expiry}-ABCD-{base64.b32encode(mac).decode().rstrip('=')[:12]}"
    assert licensing.is_signed_key(old) and licensing.verify_signed_key(old, SECRET)[0]
    assert licensing.key_plan(old) == "full"
