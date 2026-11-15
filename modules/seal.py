"""
Integrity Seal and Honesty Ledger for the PDF report.

* The SEAL is a SHA-256 fingerprint of the exact cleaned data the analysis ran
  on. It is the hash of the bytes of `cleaned_data.csv` as StatsPro exports it,
  so anyone can recompute it with `sha256sum cleaned_data.csv` and see whether
  the report and the data still belong together.
* The LEDGER states, in numbers, how many hypotheses were tested, how many look
  significant by chance alone, and how many survive multiple-testing correction.
  Most reports hide this; StatsPro prints it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform

import pandas as pd

ALPHA = 0.05
# Tables whose rows are real research hypotheses (not assumption diagnostics).
HYPOTHESIS_TABLES = ("pearson_correlation", "spearman_correlation", "chi_square",
                     "anova", "mann_whitney")


def csv_bytes(df: pd.DataFrame) -> bytes:
    """The exact bytes StatsPro offers as 'cleaned_data.csv'."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")   # same bytes on every OS


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def group4(hexstr: str) -> str:
    return " ".join(hexstr[i:i + 4] for i in range(0, len(hexstr), 4))


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "?"


def honesty_ledger(stats_results: dict, raw_df: pd.DataFrame | None,
                   cleaned_df: pd.DataFrame | None) -> dict:
    """Count the hypotheses tested and what the corrections did."""
    n = nominal = survive = 0
    for key in HYPOTHESIS_TABLES:
        t = stats_results.get(key)
        if t is None or getattr(t, "empty", True) or "P-Value" not in t.columns:
            continue
        n += len(t)
        nominal += int((t["P-Value"] < ALPHA).sum())
        if "Sig. after FDR" in t.columns:
            survive += int(t["Sig. after FDR"].fillna(False).astype(bool).sum())
    reg = stats_results.get("regression")
    if reg is not None and not getattr(reg, "empty", True):
        coefs = reg[reg["Variable"] != "const"]
        n += len(coefs)
        nominal += int((coefs["P-Value"] < ALPHA).sum())
        if "Sig. after FDR" in coefs.columns:
            survive += int(coefs["Sig. after FDR"].fillna(False).astype(bool).sum())
        else:
            survive += int((coefs["P-Value"] < ALPHA).sum())

    choices = {}
    if raw_df is not None and cleaned_df is not None:
        choices = {
            "rows_removed": int(len(raw_df) - len(cleaned_df)),
            "columns_removed": int(raw_df.shape[1] - cleaned_df.shape[1]),
            "missing_before": int(raw_df.isna().sum().sum()),
            "missing_after": int(cleaned_df.isna().sum().sum()),
        }
    expected_by_chance = ALPHA * n
    return {"tests": n, "nominal": nominal, "survive": survive,
            "expected_by_chance": expected_by_chance,
            "removed_by_correction": max(0, nominal - survive),
            "choices": choices}


def build_seal(cleaned_df: pd.DataFrame) -> dict:
    digest = fingerprint(csv_bytes(cleaned_df))
    return {
        "sha256": digest,
        "id": digest[:12].upper(),
        "rows": int(len(cleaned_df)),
        "columns": int(cleaned_df.shape[1]),
        "versions": (f"Python {platform.python_version()}, pandas {_version('pandas')}, "
                     f"SciPy {_version('scipy')}, statsmodels {_version('statsmodels')}"),
    }


def verify(file_bytes: bytes, claimed_hash: str) -> bool:
    """Does this file match the fingerprint printed in a report?"""
    claimed = "".join(claimed_hash.split()).lower()
    return len(claimed) == 64 and fingerprint(file_bytes) == claimed
