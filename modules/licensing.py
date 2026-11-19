"""
Optional "Pro" licensing for StatsPro, backed by Lemon Squeezy's public
license API (no API key or server of your own needed).

SAFE BY DEFAULT: until you configure the checkout (see MONETIZATION.md),
`pro_enabled()` is False and every feature stays unlocked, so a fresh clone or
self-hosted copy behaves exactly as before.

Configuration (Streamlit secrets or environment variables):
    CHECKOUT_URL      your Lemon Squeezy checkout link   (turns licensing on)
    LS_STORE_ID       numeric store id    } a key is only accepted if it was
    LS_PRODUCT_ID     numeric product id  } issued for YOUR store/product
    SPONSOR_URL       optional, e.g. https://github.com/sponsors/<you>
    HIRE_EMAIL        optional, defaults to the developer's address

Manual bKash option (no payment API needed; works alongside Lemon Squeezy):
    BKASH_NUMBER      your bKash number shown to buyers      } both required to
    LICENSE_SECRET    long random string, kept ONLY in secrets } turn it on
    BKASH_PLANS       optional text, e.g. "7 days: 199 BDT | 30 days: 299 BDT"
    BKASH_CONTACT     optional email for payment proofs (defaults to HIRE_EMAIL)
    REVOKED_KEYS      optional comma-separated keys to switch off (refunds)

Keys for bKash buyers are made offline with `python tools/make_key.py --days 7`.
They are signed (HMAC) and carry an expiry date, so the app verifies them without
any server, and nobody can forge one without LICENSE_SECRET.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

VALIDATE_URL = "https://api.lemonsqueezy.com/v1/licenses/validate"
PRO_FEATURES = ("Advanced statistical tests (ANOVA, Mann-Whitney, Levene, VIF, OLS regression)", "PDF report export", "Cleaned CSV download", "Thesis write-up and reproduce script", "Full Assumption Doctor results", "Statistician Review details", "ML Trust Report", "Live and batch prediction", "Model download", "Higher AI assistant limits")


class LicenseError(Exception):
    """The licence server could not be reached or answered unexpectedly."""


def get_setting(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets, falling back to the environment."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def lemon_enabled() -> bool:
    return bool(get_setting("CHECKOUT_URL") and get_setting("LS_STORE_ID"))


def bkash_enabled() -> bool:
    return bool(get_setting("BKASH_NUMBER") and get_setting("LICENSE_SECRET"))


def pro_enabled() -> bool:
    """True once the owner configured Lemon Squeezy (checkout link + store id) or bKash."""
    return lemon_enabled() or bkash_enabled()


# ── Signed offline keys (for bKash / manual payments) ───────────────────
# SP1-<expiry>-<nonce>-<sig>        older keys: expiry only (no plan recorded)
# SP2-<plan><expiry>-<nonce>-<sig>  newer keys: plan letter W (short plan) / M (month) / L (lifetime)
KEY_PREFIX = "SP2"
LEGACY_PREFIX = "SP1"
LIFETIME = "00000000"
SHORT_PLAN_MAX_DAYS = 10          # keys of up to 10 days are the "short" (7-day) plan


def _sig(secret: str, field: str, nonce: str, prefix: str = KEY_PREFIX) -> str:
    mac = hmac.new(secret.encode(), f"{prefix}|{field}|{nonce}".encode(), hashlib.sha256).digest()
    return base64.b32encode(mac).decode().rstrip("=")[:12]


def make_key(secret: str, days: Optional[int] = 30, today: Optional[dt.date] = None) -> str:
    """Create a key valid for *days* days from today (None = lifetime). The plan is signed into the key."""
    if not secret or len(secret) < 16:
        raise ValueError("LICENSE_SECRET must be at least 16 characters.")
    if days is None:
        plan, expiry = "L", LIFETIME
    else:
        if days < 1:
            raise ValueError("days must be at least 1")
        plan = "W" if days <= SHORT_PLAN_MAX_DAYS else "M"
        expiry = ((today or dt.date.today()) + dt.timedelta(days=int(days))).strftime("%Y%m%d")
    field = plan + expiry
    nonce = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
    return f"{KEY_PREFIX}-{field}-{nonce}-{_sig(secret, field, nonce)}"


def is_signed_key(key: str) -> bool:
    k = (key or "").strip().upper()
    return k.startswith(KEY_PREFIX + "-") or k.startswith(LEGACY_PREFIX + "-")


def key_plan(key: str) -> str:
    """'week' for a short-plan key, otherwise 'full'. Only trust this after verify_signed_key said the key is valid:
    the plan letter is covered by the signature, so it cannot be edited."""
    parts = (key or "").strip().upper().split("-")
    if len(parts) == 4 and parts[0] == KEY_PREFIX and parts[1][:1] == "W":
        return "week"
    return "full"


def verify_signed_key(key: str, secret: str, *, revoked=(), today: Optional[dt.date] = None) -> tuple[bool, str]:
    """Check a signed key offline. Returns (is_valid, message)."""
    key = (key or "").strip().upper()
    parts = key.split("-")
    if not secret or len(parts) != 4 or parts[0] not in (KEY_PREFIX, LEGACY_PREFIX):
        return False, "That licence key is not valid."
    prefix, field, nonce, sig = parts
    if not hmac.compare_digest(sig, _sig(secret, field, nonce, prefix)):
        return False, "That licence key is not valid."
    if key in {r.strip().upper() for r in revoked if r.strip()}:
        return False, "This licence key has been cancelled."
    expiry = field
    if prefix == KEY_PREFIX:
        if field[:1] not in ("W", "M", "L"):
            return False, "That licence key is not valid."
        expiry = field[1:]
    if expiry != LIFETIME:
        try:
            end = dt.datetime.strptime(expiry, "%Y%m%d").date()
        except ValueError:
            return False, "That licence key is not valid."
        if (today or dt.date.today()) > end:
            return False, f"This key expired on {end:%d %b %Y}."
        return True, f"Pro unlocked until {end:%d %b %Y}. Thank you for supporting StatsPro!"
    return True, "Pro unlocked. Thank you for supporting StatsPro!"


def verify_any(key: str) -> tuple[bool, str]:
    """Route a key to the right checker using the app's settings."""
    if is_signed_key(key):
        if not bkash_enabled():
            return False, "That licence key is not valid."
        revoked = get_setting("REVOKED_KEYS").split(",")
        return verify_signed_key(key, get_setting("LICENSE_SECRET"), revoked=revoked)
    if not lemon_enabled():
        return False, "That licence key is not valid."
    return verify_key(key, store_id=get_setting("LS_STORE_ID") or None,
                      product_id=get_setting("LS_PRODUCT_ID") or None)


def verify_key(key: str, *, store_id: Optional[str] = None,
               product_id: Optional[str] = None, timeout: float = 10.0,
               opener=urllib.request.urlopen) -> tuple[bool, str]:
    """Check a licence key with Lemon Squeezy.

    Returns (is_valid, message). Raises LicenseError on network trouble so the
    caller can avoid caching a failure as "invalid".
    """
    key = (key or "").strip()
    if len(key) < 10:
        return False, "That doesn't look like a licence key."

    body = urllib.parse.urlencode({"license_key": key}).encode()
    req = urllib.request.Request(
        VALIDATE_URL, data=body, method="POST",
        headers={"Accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:       # 4xx for unknown keys carries JSON
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except Exception as inner:
            raise LicenseError(f"Licence server error ({exc.code}).") from inner
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise LicenseError("Could not reach the licence server. Try again shortly.") from exc

    if not data.get("valid"):
        return False, str(data.get("error") or "This licence key is not valid.")

    meta = data.get("meta") or {}
    if store_id and str(meta.get("store_id")) != str(store_id):
        return False, "This key was not issued for StatsPro."
    if product_id and str(meta.get("product_id")) != str(product_id):
        return False, "This key was not issued for StatsPro Pro."
    return True, "Pro unlocked. Thank you for supporting StatsPro!"
