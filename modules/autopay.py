"""
Automatic key delivery for bKash payments.

How it works
------------
1. Your Android phone forwards the *bKash payment SMS* to a small Google Apps Script
   (tools/apps_script/Code.gs) that stores {TrxID, amount} in a Google Sheet.
2. The buyer types their TrxID into StatsPro.
3. This module asks the script to "claim" that TrxID. The script answers with the
   amount paid exactly once, so a TrxID can never be used twice.
4. StatsPro signs a licence key for the matching plan (see licensing.make_key) and
   shows it to the buyer, who is unlocked immediately.

Settings (Streamlit secrets / environment):
    AUTOPAY_URL      the Apps Script web-app URL   } both required to turn
    AUTOPAY_TOKEN    shared secret for claiming    } auto-delivery on
    BKASH_PRICE_7    price of the 7-day pass, default 50
    BKASH_PRICE_30   price of the 30-day plan, default 99

Without them everything falls back to the manual flow (you make keys by hand).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from . import licensing

TRX_RE = re.compile(r"^[A-Z0-9]{8,14}$")


class AutopayError(Exception):
    """The payment service could not be reached or answered unexpectedly."""


def enabled() -> bool:
    return bool(licensing.bkash_enabled() and licensing.get_setting("AUTOPAY_URL")
                and licensing.get_setting("AUTOPAY_TOKEN"))


def prices() -> tuple[float, float]:
    def _num(name, default):
        try:
            return float(licensing.get_setting(name) or default)
        except ValueError:
            return float(default)
    return _num("BKASH_PRICE_7", 50), _num("BKASH_PRICE_30", 99)


def plan_for_amount(amount: float) -> Optional[int]:
    """Days of Pro that *amount* BDT buys: the best plan it fully pays for, else None."""
    p7, p30 = prices()
    if amount >= p30:
        return 30
    if amount >= p7:
        return 7
    return None


def normalise_trx(raw: str) -> str:
    return re.sub(r"[\s\-]", "", (raw or "")).upper()


def claim(trx: str, *, url: Optional[str] = None, token: Optional[str] = None,
          timeout: float = 15.0, opener=urllib.request.urlopen) -> dict:
    """Ask the payment sheet to hand over a TrxID. Returns the script's JSON answer."""
    url = url or licensing.get_setting("AUTOPAY_URL")
    token = token or licensing.get_setting("AUTOPAY_TOKEN")
    query = urllib.parse.urlencode({"action": "claim", "trx": trx, "token": token})
    req = urllib.request.Request(f"{url}{'&' if '?' in url else '?'}{query}", headers={"Accept": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise AutopayError("Could not reach the payment service. Please try again in a minute.") from exc


_MESSAGES = {
    "not_found": "We can't see that payment yet. bKash messages can take a minute or two. "
                 "Check the TrxID and try again shortly.",
    "already_used": "That TrxID has already been used to get a key.",
    "expired": "That payment is too old to claim automatically. Please email us the TrxID.",
    "bad_token": "The payment service is misconfigured. Please email us your TrxID.",
}


def redeem(trx: str, **kw) -> tuple[bool, str, Optional[str]]:
    """Turn a paid TrxID into a signed key. Returns (ok, message, key)."""
    trx = normalise_trx(trx)
    if not TRX_RE.match(trx):
        return False, "That does not look like a bKash TrxID (8-14 letters and numbers).", None
    data = claim(trx, **kw)
    if not data.get("ok"):
        return False, _MESSAGES.get(str(data.get("error")), "We could not confirm that payment."), None
    try:
        amount = float(data.get("amount"))
    except (TypeError, ValueError):
        return False, "We could not read the payment amount. Please email us your TrxID.", None
    days = plan_for_amount(amount)
    if days is None:
        p7, _ = prices()
        return False, (f"The payment was {amount:g} BDT, which is less than the cheapest plan "
                       f"({p7:g} BDT). Please email us your TrxID."), None
    key = licensing.make_key(licensing.get_setting("LICENSE_SECRET"), days=days)
    return True, f"Payment confirmed: {amount:g} BDT for {days} days of Pro.", key
