"""
ml_engine.py — Production ML Pipeline for StatsPro
Auto-detects regression vs classification, trains 6 models,
returns metrics, feature importance, and best model for predictions.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_SCALING_REQUIRED = frozenset({"Linear Regression", "Logistic Regression"})


def _make_pipeline(name: str, model) -> Pipeline:
    if name in _SCALING_REQUIRED:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])


class MLEngine:

    def __init__(self):
        self._reset()

    def _reset(self):
        self.task: str | None = None
        self.target_col: str | None = None
        self.feature_cols: list[str] = []
        self.pipelines: dict[str, Pipeline] = {}
        self.results: pd.DataFrame | None = None
        self.best_pipeline: Pipeline | None = None
        self.best_model_name: str | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        # Maps col -> list of original string labels (in label-encoded order).
        # Only populated for columns that are categorical (object/category or
        # reclassified mixed-type). Used by the UI to show human-readable
        # choices instead of raw integer codes.
        self.categorical_labels: dict[str, list[str]] = {}
        self.confusion_matrices: dict[str, np.ndarray] = {}
        self._reclassified_cols: set[str] = set()

    # ── Task detection ────────────────────────────────────────────────────────

    def detect_task(self, df: pd.DataFrame, target: str) -> str:
        col = df[target].dropna()
        dtype = col.dtype

        # Unambiguously categorical dtypes
        if dtype == "bool" or hasattr(dtype, "categories"):
            return "classification"
        if dtype == "object":
            return "classification"

        # Numeric columns: only apply cardinality heuristics when the ratio of
        # unique values is truly low, to avoid misclassifying continuous
        # columns that happen to have few distinct values in a small sample.
        n = len(col)
        n_unique = col.nunique()

        # Hard ceiling: if there are ≤ 10 unique numeric values treat as
        # classification (e.g. ratings 1-5, binary 0/1).
        if n_unique <= 10:
            return "classification"

        # Ratio heuristic only for reasonably-sized datasets and a tight
        # threshold, to avoid false positives on large continuous columns.
        if n >= 200 and (n_unique / n) < 0.02:
            return "classification"

        return "regression"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs: mode for categoricals, median for numerics."""
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ["object", "category"]:
                    fill_val = df[col].mode()
                    df[col] = df[col].fillna(fill_val.iloc[0] if not fill_val.empty else "Unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())
        return df

    def _resolve_mixed_type_columns(self, df: pd.DataFrame, *, training: bool) -> pd.DataFrame:
        """Detect columns that contain non-numeric values mixed with numbers
        and reclassify them as strings so they are encoded consistently."""
        df = df.copy()
        if training:
            self._reclassified_cols = set()
            for col in self.feature_cols:
                if df[col].dtype in ["object", "category"]:
                    continue
                coerced = pd.to_numeric(df[col], errors="coerce")
                new_nans = coerced.isna().sum() - df[col].isna().sum()
                if new_nans > 0:
                    example = df[col][coerced.isna() & df[col].notna()].iloc[0]
                    logger.warning(
                        "Column '%s' has mixed types (%d value(s) could not be "
                        "parsed as numeric, e.g. %r). Reclassifying as categorical.",
                        col, new_nans, example,
                    )
                    df[col] = df[col].astype(str)
                    self._reclassified_cols.add(col)
        else:
            for col in self._reclassified_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        return df

    def _encode_inference_df(self, input_df: pd.DataFrame) -> np.ndarray:
        """Encode a user-supplied prediction DataFrame using fitted encoders.

        Accepts *original* (human-readable) string values for categorical
        columns — the caller does NOT need to pre-encode anything.
        """
        if self.best_pipeline is None:
            raise RuntimeError("No trained model available. Call .run() first.")

        missing = set(self.feature_cols) - set(input_df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        df_clean = input_df[self.feature_cols].copy()
        df_clean = self._resolve_mixed_type_columns(df_clean, training=False)
        df_clean = self._impute(df_clean)

        for col in self.feature_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                raw = df_clean[col].astype(str)
                known = raw.isin(le.classes_)
                if not known.all():
                    unknown = raw[~known].unique().tolist()
                    warnings.warn(
                        f"Column '{col}' contains values unseen during training: "
                        f"{unknown}. Falling back to the most frequent training class.",
                        UserWarning, stacklevel=3,
                    )
                    raw = raw.where(known, le.classes_[0])
                df_clean[col] = le.transform(raw)
            elif df_clean[col].dtype in ["object", "category"]:
                warnings.warn(
                    f"Column '{col}' is categorical but has no fitted encoder. Substituting 0.",
                    UserWarning, stacklevel=3,
                )
                df_clean[col] = 0

        return df_clean.values.astype(float)

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_data(self, df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
        self.target_col = target
        self.feature_cols = [c for c in df.columns if c != target]

        df_clean = df.dropna(subset=[target]).copy()
        df_clean = self._resolve_mixed_type_columns(df_clean, training=True)
        df_clean = self._impute(df_clean)

        # Encode the target column for classification.
        # Guard against any non-numeric dtype (object, StringDtype, category, bool).
        if self.task == "classification" and not pd.api.types.is_integer_dtype(df_clean[target]):
            le = LabelEncoder()
            df_clean[target] = le.fit_transform(df_clean[target].astype(str)).astype(int)
            self.label_encoders[target] = le
        elif self.task == "classification":
            # Already integer-coded; ensure plain Python int dtype for XGBoost/CatBoost
            df_clean[target] = df_clean[target].astype(int)

        # First pass: encode known object/category feature columns
        for col in self.feature_cols:
            if df_clean[col].dtype in ["object", "category", "string"]:
                le = LabelEncoder()
                le.fit(df_clean[col].astype(str))
                # Store the human-readable labels BEFORE transforming
                self.categorical_labels[col] = list(le.classes_)
                df_clean[col] = le.transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        # Second pass: safety net for any remaining non-numeric columns
        for col in self.feature_cols:
            if col not in self.label_encoders and not pd.api.types.is_numeric_dtype(df_clean[col]):
                le = LabelEncoder()
                le.fit(df_clean[col].astype(str))
                self.categorical_labels[col] = list(le.classes_)
                df_clean[col] = le.transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        # Reclassified mixed-type columns were cast to str before encoding,
        # so they are already captured in the passes above via label_encoders.
        # Ensure their labels are also reflected in categorical_labels.
        for col in self._reclassified_cols:
            if col in self.label_encoders and col not in self.categorical_labels:
                self.categorical_labels[col] = list(self.label_encoders[col].classes_)

        X = df_clean[self.feature_cols].values.astype(float)
        y = df_clean[target].values
        return X, y

    # ── Model building ────────────────────────────────────────────────────────

    def _build_models(self) -> dict:
        if self.task == "regression":
            return {
                "Linear Regression":  LinearRegression(),
                "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
                "XGBoost":            XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
                "CatBoost":           CatBoostRegressor(n_estimators=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0),
                "AdaBoost":           AdaBoostRegressor(n_estimators=100, random_state=42),
            }
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "XGBoost":             XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0, eval_metric="logloss"),
            "CatBoost":            CatBoostClassifier(n_estimators=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0),
            "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        self._reset()
        self.task = self.detect_task(df, target)
        X, y = self.prepare_data(df, target)

        test_size = 0.15 if len(X) < 100 else 0.2
        stratify = y if self.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        cv = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if self.task == "classification"
            else KFold(n_splits=5, shuffle=True, random_state=42)
        )

        models = self._build_models()
        self.pipelines = {name: _make_pipeline(name, model) for name, model in models.items()}

        results = []
        for name, pipe in self.pipelines.items():
            try:
                row = self._train_and_evaluate(name, pipe, X_train, X_test, y_train, y_test, cv)
            except Exception as exc:
                logger.exception("Model '%s' failed during training.", name)
                row = {"Model": name, "Error": str(exc)}
            results.append(row)

        self.results = pd.DataFrame(results)
        self._select_best_model()
        self._extract_feature_importance()
        return self.results

    def _train_and_evaluate(self, name, pipe, X_train, X_test, y_train, y_test, cv) -> dict:
        if self.task == "regression":
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            return {
                "Model":          name,
                "CV R² (mean)":   round(float(cv_scores.mean()), 4),
                "CV R² (std)":    round(float(cv_scores.std()), 4),
                "Test R²":        round(float(r2_score(y_test, y_pred)), 4),
                "Test RMSE":      round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "Test MAE":       round(float(mean_absolute_error(y_test, y_pred)), 4),
            }
        else:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            # Store a proper sklearn confusion matrix (ndarray), not a crosstab
            self.confusion_matrices[name] = confusion_matrix(y_test, y_pred)
            return {
                "Model":                name,
                "CV Accuracy (mean)":   round(float(cv_scores.mean()), 4),
                "CV Accuracy (std)":    round(float(cv_scores.std()), 4),
                "Test Accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
                "Test Precision":       round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "Test Recall":          round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "Test F1":              round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }

    # ── Model selection & feature importance ──────────────────────────────────

    def _select_best_model(self):
        metric = "Test R²" if self.task == "regression" else "Test F1"
        valid = self.results.dropna(subset=[metric])
        if valid.empty:
            raise RuntimeError("All models failed during training.")
        best_idx = valid[metric].idxmax()
        self.best_model_name = self.results.loc[best_idx, "Model"]
        self.best_pipeline = self.pipelines[self.best_model_name]

    def _extract_feature_importance(self):
        if self.best_pipeline is None:
            return
        model = self.best_pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        else:
            return
        self.feature_importance = (
            pd.DataFrame({"Feature": self.feature_cols, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Return predictions in the original label space (strings for
        classification when the target was encoded)."""
        X = self._encode_inference_df(input_df)
        preds = self.best_pipeline.predict(X)
        if self.task == "classification" and self.target_col in self.label_encoders:
            preds = self.label_encoders[self.target_col].inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities. Raises for regression or models that
        do not expose predict_proba (checked on the inner model, not the
        Pipeline wrapper, which always proxies the attribute)."""
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification tasks.")
        # Check the INNER model, not the Pipeline (Pipeline always has the
        # attribute as long as the final estimator does).
        inner_model = self.best_pipeline.named_steps["model"]
        if not hasattr(inner_model, "predict_proba"):
            raise RuntimeError(f"{self.best_model_name} does not support probability estimates.")
        X = self._encode_inference_df(input_df)
        return self.best_pipeline.predict_proba(X)

    def get_categorical_options(self, col: str) -> list[str] | None:
        """Return the original string labels for a categorical feature column,
        or None if the column is numeric.

        Use this in the prediction UI so that selectboxes show human-readable
        values (e.g. 'good', 'bad') instead of their integer codes (0, 1).
        """
        return self.categorical_labels.get(col, None)

    def is_categorical_feature(self, col: str) -> bool:
        """True when *col* was encoded as categorical during training."""
        return col in self.categorical_labels

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MLEngine":
        engine = joblib.load(path)
        if not isinstance(engine, cls):
            raise TypeError(f"Expected MLEngine, got {type(engine).__name__}.")
        return engine
