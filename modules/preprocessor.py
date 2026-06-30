"""
preprocessor.py — Production Data Preprocessing Pipeline for StatsPro

Responsibilities:
  - Dataset overview & quality scoring
  - Type inference and correction
  - Constant-column pruning
  - Exact duplicate removal
  - Missing-value imputation  (KNN for numeric, mode for categorical)
  - IQR-based outlier Winsorization
  - Optional label-encoding + standard-scaling for ML pipelines

Design principles:
  - Immutable inputs  : every method works on a copy; the caller's DataFrame
                        is never mutated.
  - Pandas 2.x safe   : no chained inplace operations; all assignments use
                        the explicit `df[col] = …` form.
  - Fail-soft         : imputation failures fall back gracefully and are
                        recorded in the cleaning log.
  - Single source of truth : cleaning_log captures every action with
                        counts so the UI can render it without re-computing.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KNN_NEIGHBORS       = 5          # k for KNNImputer
_KNN_MAX_COLS        = 50         # skip KNN for very wide numeric blocks
_KNN_MAX_ROWS        = 100_000    # skip KNN for very large datasets (speed)
_OUTLIER_IQR_FACTOR  = 1.5        # standard Tukey fence
_QUALITY_MAX_MISSING = 40         # max penalty points for missing data
_QUALITY_MAX_DUPS    = 20         # max penalty points for duplicates
_QUALITY_CONST_PEN   = 5          # penalty per constant numeric column
_QUALITY_NUMERIC_BON = 5          # bonus for having numeric columns


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _log_entry(action: str, detail: str, status: str = "completed") -> dict[str, str]:
    return {"action": action, "detail": detail, "status": status}


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------
class DataPreprocessor:
    """
    Stateful preprocessing pipeline.

    After calling `clean()` the instance exposes:
      - cleaning_log   : ordered list of action dicts (action / detail / status)
      - outlier_bounds : per-column IQR fence values (set by handle_outliers)
      - label_encoders : fitted LabelEncoders (set by encode_and_scale)
    """

    def __init__(self) -> None:
        self.label_encoders:  dict[str, LabelEncoder] = {}
        self.scaler:          StandardScaler           = StandardScaler()
        self.outlier_bounds:  dict[str, dict]          = {}
        self.cleaning_log:    list[dict[str, str]]     = []
        self._imputer:        KNNImputer | None         = None

    # -------------------------------------------------------------------------
    # Public: overview
    # -------------------------------------------------------------------------

    def get_overview(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute a lightweight summary of the DataFrame.
        Returns a plain dict; does not mutate df.
        """
        if df is None or df.empty:
            return self._empty_overview()

        n_rows, n_cols = df.shape

        missing       = df.isnull().sum()
        missing_pct   = (missing / n_rows * 100).round(2)
        total_missing = int(missing.sum())

        missing_table = (
            pd.DataFrame({
                "Column":             missing.index,
                "Missing Count":      missing.values,
                "Missing Percentage": missing_pct.values,
            })
            .sort_values("Missing Percentage", ascending=False)
            .reset_index(drop=True)
        )

        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols       = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()

        return {
            "shape":                df.shape,
            "rows":                 n_rows,
            "columns":              n_cols,
            "memory_usage":         round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
            "duplicates":           int(df.duplicated().sum()),
            "duplicate_percentage": round(df.duplicated().sum() / n_rows * 100, 2),
            "missing_table":        missing_table,
            "total_missing":        total_missing,
            "missing_percentage":   round(total_missing / (n_rows * n_cols) * 100, 2),
            "numeric_columns":      numeric_cols,
            "categorical_columns":  cat_cols,
            "datetime_columns":     datetime_cols,
            "numeric_count":        len(numeric_cols),
            "categorical_count":    len(cat_cols),
            "quality_score":        self.get_quality_score(df),
        }

    def _empty_overview(self) -> dict[str, Any]:
        return {
            "shape": (0, 0), "rows": 0, "columns": 0,
            "memory_usage": 0.0, "duplicates": 0, "duplicate_percentage": 0.0,
            "missing_table": pd.DataFrame(), "total_missing": 0,
            "missing_percentage": 0.0, "numeric_columns": [],
            "categorical_columns": [], "datetime_columns": [],
            "numeric_count": 0, "categorical_count": 0, "quality_score": 0.0,
        }

    # -------------------------------------------------------------------------
    # Public: quality score
    # -------------------------------------------------------------------------

    def get_quality_score(self, df: pd.DataFrame) -> float:
        """
        Composite quality score (0–100).

        Scoring breakdown:
          - Start at 100
          - Deduct up to 40 pts for missing values  (2 pts per 1%)
          - Deduct up to 20 pts for duplicate rows  (3 pts per 1%)
          - Deduct  5 pts per constant numeric column
          - Add     5 pts bonus when numeric columns are present
        """
        if df is None or df.empty:
            return 0.0

        n_cells = df.shape[0] * df.shape[1]
        score   = 100.0

        missing_pct = df.isnull().sum().sum() / n_cells * 100
        score -= min(missing_pct * 2, _QUALITY_MAX_MISSING)

        dup_pct = df.duplicated().sum() / len(df) * 100
        score -= min(dup_pct * 3, _QUALITY_MAX_DUPS)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                score -= _QUALITY_CONST_PEN

        if len(numeric_cols) > 0:
            score += _QUALITY_NUMERIC_BON

        return round(float(np.clip(score, 0, 100)), 1)

    # -------------------------------------------------------------------------
    # Public: clean
    # -------------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """
        Full cleaning pipeline (returns a new DataFrame; original is untouched):

          1. Drop zero-variance (constant) columns
          2. Remove exact duplicate rows
          3. Impute missing values
               - Numeric  : KNNImputer (k=5, distance-weighted)
                            Falls back to per-column median on failure or when
                            the block is too large for KNN to be practical.
               - Categorical : mode imputation; 'Unknown' when mode is empty.

        Returns
        -------
        df_clean   : pd.DataFrame  — cleaned copy
        cleaning_log : list[dict]  — audit trail
        """
        if df is None or df.empty:
            self.cleaning_log = [_log_entry("Clean", "Empty DataFrame — nothing to do.", "skipped")]
            return df.copy() if df is not None else pd.DataFrame(), self.cleaning_log

        df_clean = df.copy()
        self.cleaning_log = []
        initial_rows, initial_cols = df_clean.shape

        # ── Step 1: Drop constant columns ─────────────────────────────────────
        df_clean = self._drop_constant_columns(df_clean)

        # ── Step 2: Remove exact duplicates ──────────────────────────────────
        df_clean = self._remove_duplicates(df_clean, initial_rows)

        # ── Step 3: Impute missing values ─────────────────────────────────────
        df_clean = self._impute_missing(df_clean)

        # ── Summary ───────────────────────────────────────────────────────────
        final_rows, final_cols = df_clean.shape
        self.cleaning_log.append(_log_entry(
            "Summary",
            f"Rows: {initial_rows:,} → {final_rows:,} | "
            f"Columns: {initial_cols} → {final_cols}",
        ))

        return df_clean.reset_index(drop=True), self.cleaning_log

    # ── Clean sub-steps ──────────────────────────────────────────────────────

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with zero variance (including all-NaN columns)."""
        constant_cols = [
            c for c in df.columns
            if df[c].nunique(dropna=False) <= 1
        ]
        if not constant_cols:
            return df

        df = df.drop(columns=constant_cols)
        self.cleaning_log.append(_log_entry(
            "Drop Constant Columns",
            f"Removed {len(constant_cols)} constant column(s): "
            f"{', '.join(constant_cols)}",
        ))
        return df

    def _remove_duplicates(self, df: pd.DataFrame, initial_rows: int) -> pd.DataFrame:
        """Drop exact duplicate rows and record how many were removed."""
        dup_count = int(df.duplicated().sum())
        if dup_count == 0:
            return df

        df = df.drop_duplicates()
        self.cleaning_log.append(_log_entry(
            "Remove Duplicates",
            f"Removed {dup_count:,} duplicate row(s) "
            f"({dup_count / initial_rows * 100:.1f}% of original).",
        ))
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute all missing values:
          - Numeric columns   : KNNImputer → median fallback
          - Categorical cols  : mode → 'Unknown' fallback
        """
        total_missing = int(df.isnull().sum().sum())
        if total_missing == 0:
            self.cleaning_log.append(_log_entry(
                "Missing Values", "No missing values detected.", "skipped"
            ))
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── Numeric ──────────────────────────────────────────────────────────
        if numeric_cols and df[numeric_cols].isnull().any().any():
            df, method = self._impute_numeric(df, numeric_cols)
            self.cleaning_log.append(_log_entry(
                "Impute Numeric Missing",
                f"Filled missing values in {len(numeric_cols)} numeric "
                f"column(s) using {method}.",
            ))

        # ── Categorical ───────────────────────────────────────────────────────
        for col in cat_cols:
            missing_count = int(df[col].isnull().sum())
            if missing_count == 0:
                continue

            mode_series = df[col].mode()
            fill_value  = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col]     = df[col].fillna(fill_value)

            self.cleaning_log.append(_log_entry(
                "Impute Categorical Missing",
                f'Column "{col}": {missing_count:,} value(s) filled '
                f'with mode ("{fill_value}").',
            ))

        return df

    def _impute_numeric(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> tuple[pd.DataFrame, str]:
        """
        Attempt KNN imputation on the numeric block.
        Falls back to per-column median when:
          - The numeric block has more columns than _KNN_MAX_COLS
          - The DataFrame has more rows than _KNN_MAX_ROWS
          - KNNImputer raises any exception (e.g. all-NaN column)
        """
        n_rows, n_numeric = len(df), len(numeric_cols)
        use_knn = (n_numeric <= _KNN_MAX_COLS and n_rows <= _KNN_MAX_ROWS)

        if use_knn:
            try:
                imputer             = KNNImputer(n_neighbors=_KNN_NEIGHBORS, weights="distance")
                imputed             = imputer.fit_transform(df[numeric_cols])
                df[numeric_cols]    = imputed
                self._imputer       = imputer
                return df, f"KNN (k={_KNN_NEIGHBORS}, distance-weighted)"
            except Exception as exc:
                logger.warning("KNNImputer failed (%s); falling back to median.", exc)

        # Median fallback — column by column, pandas 2.x safe
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val  = df[col].median()
                df[col]     = df[col].fillna(median_val)

        reason = "median (dataset too large for KNN)" if not use_knn else "median (KNN fallback)"
        return df, reason

    # -------------------------------------------------------------------------
    # Public: handle_outliers
    # -------------------------------------------------------------------------

    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Detect and Winsorize outliers in all numeric columns.

        Strategy: cap values at [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
        Rows are never dropped — the distribution is compressed, not truncated.

        Parameters
        ----------
        df     : DataFrame to process (a copy is made; original unchanged).
        method : reserved for future extension ('iqr' is the only option now).

        Returns
        -------
        df_clean      : Winsorized DataFrame
        outlier_counts: {column_name: n_outliers_capped}  (includes zeros)
        """
        df_clean      = df.copy()
        outlier_counts: dict[str, int] = {}
        numeric_cols  = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df_clean[col].dropna()
            if series.empty:
                outlier_counts[col] = 0
                continue

            q1, q3  = series.quantile(0.25), series.quantile(0.75)
            iqr     = q3 - q1

            # Zero-IQR columns (constant after cleaning): skip Winsorization
            if iqr == 0:
                outlier_counts[col] = 0
                continue

            lower = q1 - _OUTLIER_IQR_FACTOR * iqr
            upper = q3 + _OUTLIER_IQR_FACTOR * iqr

            self.outlier_bounds[col] = {
                "lower": lower, "upper": upper,
                "Q1": q1,       "Q3": q3,  "IQR": iqr,
            }

            n_outliers          = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
            outlier_counts[col] = n_outliers
            df_clean[col]       = df_clean[col].clip(lower=lower, upper=upper)

        total_capped     = sum(outlier_counts.values())
        affected_cols    = sum(1 for v in outlier_counts.values() if v > 0)

        if total_capped > 0:
            self.cleaning_log.append(_log_entry(
                "Outlier Capping",
                f"Winsorized {total_capped:,} outlier value(s) across "
                f"{affected_cols} column(s) using IQR × {_OUTLIER_IQR_FACTOR}.",
            ))
        else:
            self.cleaning_log.append(_log_entry(
                "Outlier Check", "No outliers detected across numeric columns.", "skipped"
            ))

        return df_clean, outlier_counts

    # -------------------------------------------------------------------------
    # Public: encode_and_scale  (ML preparation only)
    # -------------------------------------------------------------------------

    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a DataFrame for classic ML pipelines that require purely
        numeric input with zero-mean, unit-variance features.

        Actions:
          1. Label-encode every object / category column.
          2. Standard-scale every numeric column.

        Note: the MLEngine has its own encoding logic that preserves original
        labels for the prediction UI.  Use this method only when you need a
        fully numeric, scaled copy for downstream tasks outside MLEngine.
        """
        df_out = df.copy()

        # 1. Label-encode categoricals
        cat_cols = df_out.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            df_out[col]            = le.fit_transform(df_out[col].astype(str))
            self.label_encoders[col] = le

        # 2. Standard-scale numerics
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_out[numeric_cols] = self.scaler.fit_transform(df_out[numeric_cols])

        return df_out

    # -------------------------------------------------------------------------
    # Public: column-level utilities
    # -------------------------------------------------------------------------

    def infer_and_cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to coerce columns stored as object/string to their natural
        numeric or datetime types.  Leaves columns unchanged if coercion
        would introduce more than 5% new NaN values, signalling mixed content.

        Returns a new DataFrame with improved dtypes.
        """
        df_out  = df.copy()
        n       = len(df_out)

        for col in df_out.select_dtypes(include=["object"]).columns:
            # Try numeric first
            coerced_num = pd.to_numeric(df_out[col], errors="coerce")
            new_nans    = int(coerced_num.isna().sum()) - int(df_out[col].isna().sum())
            if new_nans / max(n, 1) <= 0.05:
                df_out[col] = coerced_num
                continue

            # Try datetime
            try:
                coerced_dt  = pd.to_datetime(df_out[col], errors="coerce", infer_datetime_format=True)
                new_nans_dt = int(coerced_dt.isna().sum()) - int(df_out[col].isna().sum())
                if new_nans_dt / max(n, 1) <= 0.05:
                    df_out[col] = coerced_dt
            except Exception:
                pass  # leave as object

        return df_out

    def column_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a per-column diagnostic DataFrame with:
          dtype, n_unique, n_missing, missing_pct, mean/median/std (numeric),
          top_value/top_freq (categorical), skewness, suggested_action.
        """
        rows = []
        n    = len(df)

        for col in df.columns:
            series      = df[col]
            n_missing   = int(series.isna().sum())
            missing_pct = round(n_missing / max(n, 1) * 100, 2)
            n_unique    = int(series.nunique(dropna=True))
            dtype_str   = str(series.dtype)

            row: dict[str, Any] = {
                "Column":       col,
                "Dtype":        dtype_str,
                "Unique":       n_unique,
                "Missing":      n_missing,
                "Missing %":    missing_pct,
            }

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                row.update({
                    "Mean":      round(float(clean.mean()), 4)   if len(clean) else np.nan,
                    "Median":    round(float(clean.median()), 4) if len(clean) else np.nan,
                    "Std":       round(float(clean.std()), 4)    if len(clean) else np.nan,
                    "Min":       round(float(clean.min()), 4)    if len(clean) else np.nan,
                    "Max":       round(float(clean.max()), 4)    if len(clean) else np.nan,
                    "Skewness":  round(float(stats.skew(clean)), 3) if len(clean) > 2 else np.nan,
                    "Top Value": np.nan,
                    "Top Freq":  np.nan,
                    "Suggested Action": self._suggest_action_numeric(series, missing_pct),
                })
            else:
                vc        = series.value_counts(dropna=True)
                top_val   = str(vc.index[0]) if not vc.empty else "N/A"
                top_freq  = int(vc.iloc[0])  if not vc.empty else 0
                row.update({
                    "Mean":      np.nan, "Median": np.nan, "Std": np.nan,
                    "Min":       np.nan, "Max":    np.nan, "Skewness": np.nan,
                    "Top Value": top_val,
                    "Top Freq":  top_freq,
                    "Suggested Action": self._suggest_action_categorical(series, missing_pct, n_unique),
                })

            rows.append(row)

        return pd.DataFrame(rows)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _suggest_action_numeric(series: pd.Series, missing_pct: float) -> str:
        if series.nunique() <= 1:
            return "Drop — constant column"
        if missing_pct > 50:
            return "Consider dropping (>50% missing)"
        if missing_pct > 0:
            return "Impute missing values"
        skew = abs(float(stats.skew(series.dropna())))
        if skew > 2:
            return "Consider log/sqrt transform (high skew)"
        return "OK"

    @staticmethod
    def _suggest_action_categorical(
        series: pd.Series, missing_pct: float, n_unique: int
    ) -> str:
        n = len(series)
        if n_unique <= 1:
            return "Drop — constant column"
        if missing_pct > 50:
            return "Consider dropping (>50% missing)"
        if missing_pct > 0:
            return "Impute with mode"
        if n_unique / max(n, 1) > 0.9:
            return "High cardinality — consider hashing or dropping"
        return "OK"
