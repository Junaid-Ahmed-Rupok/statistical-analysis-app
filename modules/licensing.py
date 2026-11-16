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
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

VALIDATE_URL = "https://api.lemonsqueezy.com/v1/licenses/validate"
PRO_FEATURES = ("PDF report export", "Cleaned CSV download", "Reproducible script export", "Batch prediction", "Model download")


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


def pro_enabled() -> bool:
    """True only once the owner has configured a checkout link AND a store id."""
    return bool(get_setting("CHECKOUT_URL") and get_setting("LS_STORE_ID"))


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
