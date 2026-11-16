"""Privacy helpers: spot columns that probably hold personal data."""

from __future__ import annotations

import re

import pandas as pd

_NAME_HINTS = (
    ("email", "email addresses"), ("e-mail", "email addresses"), ("phone", "phone numbers"),
    ("mobile", "phone numbers"), ("contact", "contact details"), ("address", "street addresses"),
    ("full_name", "personal names"), ("fullname", "personal names"), ("first_name", "personal names"),
    ("last_name", "personal names"), ("surname", "personal names"), ("student_name", "personal names"),
    ("nid", "national ID numbers"), ("national_id", "national ID numbers"), ("passport", "passport numbers"),
    ("ssn", "national ID numbers"), ("dob", "dates of birth"), ("birth", "dates of birth"),
    ("ip_address", "IP addresses"), ("iban", "bank account numbers"), ("card_number", "card numbers"),
)
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_PHONE = re.compile(r"^\+?\d[\d\s\-()]{8,16}\d$")


def _token_match(col: str, hint: str) -> bool:
    c = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
    if hint in ("nid", "ssn", "dob"):                      # short hints must be a whole word
        return hint in c.split("_")
    return hint in c


def pii_columns(df: pd.DataFrame, sample: int = 200) -> list[tuple[str, str]]:
    """Return [(column, what it looks like)] for columns that may hold personal data."""
    out: dict[str, str] = {}
    for col in df.columns:
        name = str(col)
        for hint, label in _NAME_HINTS:
            if _token_match(name, hint):
                out[name] = label
                break
        if name in out or not (df[col].dtype == object or str(df[col].dtype).startswith("str")):
            continue
        vals = df[col].dropna().astype(str).head(sample)
        if len(vals) < 5:
            continue
        if (vals.map(lambda v: bool(_EMAIL.match(v.strip())))).mean() > 0.6:
            out[name] = "email addresses"
        elif (vals.map(lambda v: bool(_PHONE.match(v.strip())))).mean() > 0.6:
            out[name] = "phone numbers"
    return list(out.items())
