"""Forgiving CSV reader: tolerates BOMs, Excel encodings, other delimiters and empty files."""

from __future__ import annotations

import io

import pandas as pd

MAX_COLUMNS = 500


class CSVError(ValueError):
    """A problem with the uploaded file, worded so a non-technical user can act on it."""


def read_csv(file) -> pd.DataFrame:
    raw = file.read() if hasattr(file, "read") else bytes(file)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw or not raw.strip():
        raise CSVError("The file is empty.")
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:                                      # latin-1 never fails, but be safe
        raise CSVError("The file's text encoding could not be read. Save it as UTF-8 CSV and try again.")

    df = None
    for kwargs in ({}, {"sep": None, "engine": "python"}, {"sep": ";"}, {"sep": "\t"}):
        try:
            cand = pd.read_csv(io.StringIO(text), **kwargs)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, csv_error()):
            continue
        if df is None or cand.shape[1] > df.shape[1]:
            df = cand
        if df.shape[1] > 1:
            break
    if df is None:
        raise CSVError("This does not look like a CSV file. Check that it is a plain-text table with a header row.")
    if df.shape[0] == 0:
        raise CSVError("The file has column headings but no data rows.")
    if df.shape[1] > MAX_COLUMNS:
        raise CSVError(f"The file has {df.shape[1]:,} columns; the limit is {MAX_COLUMNS}.")
    df = df.dropna(axis=1, how="all") if df.shape[1] > 1 else df
    if df.shape[1] == 0:
        raise CSVError("Every column in the file is empty.")
    df.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)]
    return df


def csv_error():
    import csv
    return csv.Error
