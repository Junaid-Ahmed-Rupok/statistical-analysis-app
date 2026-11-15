"""
ml_engine.py — ML Studio engine for StatsPro

Auto-detects regression vs classification (the user can override), trains six
models with cross-validation, picks the best model by CROSS-VALIDATION score
(not by the holdout set, so the holdout stays an honest final check), explains
it with permutation importance, and serves predictions.

Guard rails: validates the target, drops ID/constant columns by default,
warns about likely target leakage and class imbalance, and handles rare classes
instead of crashing.
"""

import logging
import re
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Linear models need scaled numeric inputs and one-hot categoricals (integer
# codes would wrongly imply "North" < "South" < "West").
_LINEAR_MODELS = frozenset({"Linear Regression", "Logistic Regression"})

MIN_ROWS = 30          # below this, results are meaningless
MAX_CLASSES = 20       # above this a "classification" target is really an ID / free text
_ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|index|idx|key)$", re.IGNORECASE)


def _make_pipeline(name: str, model, cat_idx: list[int], n_features: int) -> Pipeline:
    if name in _LINEAR_MODELS:
        cat_set = set(cat_idx)
        num_idx = [i for i in range(n_features) if i not in cat_set]
        parts = []
        if cat_idx:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx))
        if num_idx:
            parts.append(("num", StandardScaler(), num_idx))
        return Pipeline([("prep", ColumnTransformer(parts)), ("model", model)])
    return Pipeline([("model", model)])


def _is_text_like(series: pd.Series) -> bool:
    """True for object / category / pandas-3 'str' / string columns."""
    return not pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)


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
        self.selection_metric: str | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.importance_method: str | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        # col -> original string labels, for categorical FEATURES (used by the UI
        # to show human-readable choices instead of integer codes).
        self.categorical_labels: dict[str, list[str]] = {}
        self.confusion_matrices: dict[str, np.ndarray] = {}
        self.class_labels: list[str] = []
        self.holdout: dict[str, np.ndarray] | None = None   # best model's holdout y_true / y_pred
        self.notes: list[str] = []                          # human-readable heads-ups from the last run
        self._reclassified_cols: set[str] = set()
        self._X_test: np.ndarray | None = None
        self._clean_index = None        # original row labels of the rows used for training/testing
        self._test_pos: np.ndarray | None = None
        self._cal_scores: np.ndarray | None = None   # conformal calibration scores
        self.conformal: dict | None = None
        self._y_test: np.ndarray | None = None
        self._cat_idx: list[int] = []

    # ── Task detection ────────────────────────────────────────────────────────

    def detect_task(self, df: pd.DataFrame, target: str) -> str:
        col = df[target].dropna()
        dtype = col.dtype

        # Anything that is not a plain number (object, category, bool, and the
        # pandas-3 'str' dtype) is a class label.
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            return "classification"

        n = len(col)
        n_unique = col.nunique()

        # ≤ 10 distinct numeric values (ratings 1-5, binary 0/1) → classes.
        if n_unique <= 10:
            return "classification"

        # Low unique ratio on a reasonably large dataset → classes.
        if n >= 200 and (n_unique / n) < 0.02:
            return "classification"

        return "regression"

    # ── Feature suggestions & checks ──────────────────────────────────────────

    def suggest_features(self, df: pd.DataFrame, target: str) -> dict:
        """Pick sensible default features and flag likely mistakes.

        Returns {"features": [...], "excluded": {col: reason}, "warnings": [...]}.
        ID-like, constant and date columns are excluded by default (they cannot
        generalise). Likely target leakage is only WARNED about, because a high
        correlation can be legitimate and the user should decide.
        """
        n = len(df)
        features, excluded, warns = [], {}, []
        target_numeric = pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target])

        for col in df.columns:
            if col == target:
                continue
            s = df[col].dropna()
            nunique = s.nunique()
            if nunique <= 1:
                excluded[col] = "constant (one value only)"
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                excluded[col] = "date/time column"
                continue
            numeric = pd.api.types.is_numeric_dtype(df[col])
            if not numeric and n >= MIN_ROWS and nunique / max(len(s), 1) > 0.95:
                excluded[col] = "looks like an ID or free text (almost every value is unique)"
                continue
            if (pd.api.types.is_integer_dtype(df[col]) and n >= MIN_ROWS
                    and (_ID_NAME.search(str(col)) or (s.is_monotonic_increasing and nunique == len(s)))):
                excluded[col] = "looks like a row ID / index"
                continue
            features.append(col)

            if target_numeric and numeric and not pd.api.types.is_bool_dtype(df[col]):
                try:
                    r = df[[col, target]].dropna().corr().iloc[0, 1]
                except Exception:
                    r = np.nan
                if pd.notna(r) and abs(r) >= 0.95:
                    warns.append(
                        f"'{col}' is almost identical to the target (r = {r:.2f}). "
                        f"It may be leaking the answer — consider unticking it."
                    )
        return {"features": features, "excluded": excluded, "warnings": warns}

    def _validate_target(self, df: pd.DataFrame, target: str):
        y = df[target].dropna()
        if len(y) < MIN_ROWS:
            raise ValueError(
                f"Need at least {MIN_ROWS} rows with a value for '{target}' to train "
                f"reliable models (found {len(y)})."
            )
        n_unique = y.nunique()
        if n_unique < 2:
            raise ValueError(f"'{target}' has only one distinct value — there is nothing to predict.")
        if self.task == "regression":
            if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
                raise ValueError(
                    f"Regression needs a numeric target, but '{target}' is not numeric. "
                    f"Switch the task to Classification."
                )
        elif n_unique > MAX_CLASSES:
            raise ValueError(
                f"'{target}' has {n_unique} distinct values — too many to treat as classes "
                f"(max {MAX_CLASSES}). It looks like an ID or free text. Pick a different "
                f"target{', or switch the task to Regression' if pd.api.types.is_numeric_dtype(y) else ''}."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs: mode for text columns, median for numeric ones."""
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                if _is_text_like(df[col]):
                    fill_val = df[col].mode()
                    df[col] = df[col].fillna(fill_val.iloc[0] if not fill_val.empty else "Unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())
        return df

    def _resolve_mixed_type_columns(self, df: pd.DataFrame, *, training: bool) -> pd.DataFrame:
        """Columns that mix numbers and text (e.g. 5, 'N/A', 7) are converted to
        text so they encode consistently. Plain text columns are left alone."""
        df = df.copy()
        if training:
            self._reclassified_cols = set()
            for col in self.feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if pd.api.types.infer_dtype(df[col], skipna=True) in ("mixed", "mixed-integer"):
                    logger.warning("Column '%s' mixes numbers and text; treating it as categorical.", col)
                    self._reclassified_cols.add(col)
        for col in self._reclassified_cols:
            if col in df.columns:
                mask = df[col].notna()
                df[col] = df[col].astype(object)
                df.loc[mask, col] = df.loc[mask, col].astype(str)
        return df

    def _encode_inference_df(self, input_df: pd.DataFrame) -> np.ndarray:
        """Encode a user-supplied prediction DataFrame using the fitted encoders.

        Accepts *original* (human-readable) values for categorical columns — the
        caller does NOT need to pre-encode anything.
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
                        f"{unknown}. Falling back to the first known category.",
                        UserWarning, stacklevel=3,
                    )
                    raw = raw.where(known, le.classes_[0])
                df_clean[col] = le.transform(raw)
            elif _is_text_like(df_clean[col]):
                warnings.warn(
                    f"Column '{col}' is categorical but has no fitted encoder. Substituting 0.",
                    UserWarning, stacklevel=3,
                )
                df_clean[col] = 0

        return df_clean.values.astype(float)

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_data(self, df: pd.DataFrame, target: str,
                     features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.target_col = target
        self.feature_cols = list(features) if features else [c for c in df.columns if c != target]

        df_clean = df.dropna(subset=[target]).copy()

        # Classes with a single example can be neither split nor cross-validated.
        if self.task == "classification":
            counts = df_clean[target].value_counts()
            rare = counts[counts < 2].index.tolist()
            if rare:
                df_clean = df_clean[~df_clean[target].isin(rare)]
                self.notes.append(
                    f"Removed {len(rare)} class(es) with only one example "
                    f"({', '.join(map(str, rare[:5]))}{'…' if len(rare) > 5 else ''}) — "
                    f"a single example can't be learned or evaluated."
                )

        df_clean = self._resolve_mixed_type_columns(df_clean, training=True)
        df_clean = self._impute(df_clean)

        # Always label-encode classification targets so class codes are
        # 0..k-1 (XGBoost/CatBoost require this, e.g. ratings 1-5 would fail).
        if self.task == "classification":
            le = LabelEncoder()
            tgt = df_clean[target]
            numeric = pd.api.types.is_numeric_dtype(tgt) and not pd.api.types.is_bool_dtype(tgt)
            df_clean[target] = le.fit_transform(tgt if numeric else tgt.astype(str)).astype(int)
            self.label_encoders[target] = le
            self.class_labels = [str(c) for c in le.classes_]

        # Encode text/categorical features to integer codes (kept human-readable
        # via categorical_labels; linear models one-hot them inside the pipeline).
        for col in self.feature_cols:
            if _is_text_like(df_clean[col]):
                le = LabelEncoder()
                le.fit(df_clean[col].astype(str))
                self.categorical_labels[col] = list(le.classes_)
                df_clean[col] = le.transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        self._cat_idx = [i for i, c in enumerate(self.feature_cols) if c in self.categorical_labels]
        X = df_clean[self.feature_cols].values.astype(float)
        y = df_clean[target].values
        self._clean_index = df_clean.index.to_numpy()
        return X, y

    # ── Model building ────────────────────────────────────────────────────────

    def _build_models(self) -> dict:
        if self.task == "regression":
            return {
                "Linear Regression":  LinearRegression(),
                "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
                "XGBoost":            XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
                "CatBoost":           CatBoostRegressor(n_estimators=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0, allow_writing_files=False),
                "AdaBoost":           AdaBoostRegressor(n_estimators=100, random_state=42),
            }
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "XGBoost":             XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0, eval_metric="logloss"),
            "CatBoost":            CatBoostClassifier(n_estimators=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0, allow_writing_files=False),
            "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, target: str, task: str | None = None,
            features: list[str] | None = None, progress_callback=None) -> pd.DataFrame:
        """Train all models.

        task:  None/'auto' to auto-detect, or 'regression' / 'classification'.
        features: columns to use (default: every column except the target).
        progress_callback(done, total, model_name): called after each model.
        """
        self._reset()
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        if task in (None, "auto"):
            task = self.detect_task(df, target)
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'auto', 'regression' or 'classification'.")
        self.task = task

        self._validate_target(df, target)
        feats = [c for c in (features or df.columns) if c in df.columns and c != target]
        if not feats:
            raise ValueError("Select at least one feature column to train on.")

        X, y = self.prepare_data(df, target, feats)
        if len(X) < MIN_ROWS:
            raise ValueError(f"Only {len(X)} usable rows left after cleaning — need at least {MIN_ROWS}.")
        if len(X) < 100:
            self.notes.append(f"Only {len(X)} rows: treat these scores as rough estimates.")

        if self.task == "classification":
            counts = np.bincount(y)
            share = counts.min() / counts.sum()
            if len(counts) >= 2 and share < 0.10:
                self.notes.append(
                    f"Imbalanced classes: the smallest class is {share:.0%} of the data. "
                    f"Accuracy can look high while that class is ignored — judge by F1 / ROC-AUC."
                )
            n_splits = int(max(2, min(5, counts.min())))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify = y
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            stratify = None

        test_size = 0.15 if len(X) < 100 else 0.2
        pos = np.arange(len(X))
        X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
            X, y, pos, test_size=test_size, random_state=42, stratify=stratify
        )
        self._test_pos = pos_test
        self._X_test, self._y_test = X_test, y_test

        models = self._build_models()
        self.pipelines = {
            name: _make_pipeline(name, model, self._cat_idx, len(self.feature_cols))
            for name, model in models.items()
        }

        results = []
        total = len(self.pipelines)
        for i, (name, pipe) in enumerate(self.pipelines.items(), start=1):
            t0 = time.time()
            try:
                row = self._train_and_evaluate(name, pipe, X_train, X_test, y_train, y_test, cv)
                row["Time (s)"] = round(time.time() - t0, 2)
            except Exception as exc:
                logger.exception("Model '%s' failed during training.", name)
                row = {"Model": name, "Error": str(exc)[:200]}
            results.append(row)
            if progress_callback:
                progress_callback(i, total, name)

        self.results = pd.DataFrame(results)
        self._select_best_model()
        self.holdout = {"y_true": y_test, "y_pred": self.best_pipeline.predict(X_test)}
        self._extract_feature_importance()
        self._fit_conformal(X_train, y_train)
        self._sanity_notes()
        return self.results

    def _train_and_evaluate(self, name, pipe, X_train, X_test, y_train, y_test, cv) -> dict:
        if self.task == "regression":
            cvr = cross_validate(pipe, X_train, y_train, cv=cv, scoring={"r2": "r2"}, error_score="raise")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            return {
                "Model":          name,
                "CV R² (mean)":   round(float(cvr["test_r2"].mean()), 4),
                "CV R² (std)":    round(float(cvr["test_r2"].std()), 4),
                "Test R²":        round(float(r2_score(y_test, y_pred)), 4),
                "Test RMSE":      round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "Test MAE":       round(float(mean_absolute_error(y_test, y_pred)), 4),
            }

        cvr = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring={"accuracy": "accuracy", "f1": "f1_weighted"}, error_score="raise")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        self.confusion_matrices[name] = confusion_matrix(y_test, y_pred, labels=range(len(self.class_labels)))
        try:
            proba = pipe.predict_proba(X_test)
            if proba.shape[1] == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted",
                                    labels=range(proba.shape[1]))
            auc = round(float(auc), 4)
        except Exception:
            auc = np.nan
        return {
            "Model":                name,
            "CV Accuracy (mean)":   round(float(cvr["test_accuracy"].mean()), 4),
            "CV Accuracy (std)":    round(float(cvr["test_accuracy"].std()), 4),
            "CV F1 (mean)":         round(float(cvr["test_f1"].mean()), 4),
            "Test Accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
            "Test Precision":       round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "Test Recall":          round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "Test F1":              round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "Test ROC-AUC":         auc,
        }

    # ── Model selection & feature importance ──────────────────────────────────

    def _select_best_model(self):
        """Pick the best model by cross-validation score on the TRAINING data, so
        the holdout test set is never used to choose — it stays an honest check."""
        metric = "CV R² (mean)" if self.task == "regression" else "CV F1 (mean)"
        self.selection_metric = metric
        if metric not in self.results.columns:
            raise RuntimeError("All models failed during training.")
        valid = self.results.dropna(subset=[metric])
        if valid.empty:
            raise RuntimeError("All models failed during training.")
        best_idx = valid[metric].idxmax()
        self.best_model_name = self.results.loc[best_idx, "Model"]
        self.best_pipeline = self.pipelines[self.best_model_name]

    def _extract_feature_importance(self):
        """Permutation importance on the holdout set: how much the score drops when
        a feature is shuffled. Works for every model type and is not biased toward
        high-cardinality features the way impurity importance is."""
        if self.best_pipeline is None or self._X_test is None:
            return
        X_t, y_t = self._X_test, self._y_test
        if len(X_t) > 2000:
            idx = np.random.default_rng(42).choice(len(X_t), 2000, replace=False)
            X_t, y_t = X_t[idx], y_t[idx]
        scoring = "r2" if self.task == "regression" else "f1_weighted"
        try:
            r = permutation_importance(self.best_pipeline, X_t, y_t, n_repeats=5,
                                       random_state=42, scoring=scoring, n_jobs=1)
        except Exception:
            logger.exception("Permutation importance failed.")
            return
        self.feature_importance = (
            pd.DataFrame({
                "Feature": self.feature_cols,
                "Importance": np.clip(r.importances_mean, 0, None),
                "Std": r.importances_std,
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        self.importance_method = (
            f"Permutation importance — drop in {'R²' if self.task == 'regression' else 'F1'} "
            f"on the holdout set when a feature is shuffled"
        )

    # ── Trust layer: uncertainty, sanity checks, fairness ────────────────────

    def _fit_conformal(self, X_train, y_train, calib_frac: float = 0.25):
        """Split-conformal calibration. A clone of the best model is fitted on part
        of the training data; its errors on the remaining part set the interval
        width. Coverage is then CHECKED on the untouched holdout set."""
        from sklearn.base import clone
        try:
            Xa, Xc, ya, yc = train_test_split(
                X_train, y_train, test_size=calib_frac, random_state=7,
                stratify=y_train if self.task == "classification" else None)
        except ValueError:                      # too few rows in a class to stratify
            Xa, Xc, ya, yc = train_test_split(X_train, y_train, test_size=calib_frac, random_state=7)
        if len(yc) < 10:
            return
        try:
            tmp = clone(self.best_pipeline).fit(Xa, ya)
            if self.task == "regression":
                self._cal_scores = np.abs(yc - tmp.predict(Xc))
            else:
                proba = tmp.predict_proba(Xc)
                idx = np.searchsorted(tmp.classes_, yc)
                self._cal_scores = 1.0 - proba[np.arange(len(yc)), idx]
        except Exception:
            logger.exception("Conformal calibration failed.")
            self._cal_scores = None
            return
        self.conformal = {"n_cal": int(len(yc))}
        self.conformal_summary(0.10)

    def conformal_quantile(self, alpha: float) -> float:
        n = len(self._cal_scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(self._cal_scores, level, method="higher"))

    def conformal_summary(self, alpha: float = 0.10) -> dict | None:
        """Target vs observed coverage on the holdout set (and average interval
        width / set size)."""
        if self._cal_scores is None or self._X_test is None:
            return None
        q = self.conformal_quantile(alpha)
        yt = self._y_test
        if self.task == "regression":
            pred = self.best_pipeline.predict(self._X_test)
            covered = np.abs(yt - pred) <= q
            extra = {"avg_width": float(2 * q)}
        else:
            proba = self.best_pipeline.predict_proba(self._X_test)
            sets = proba >= (1 - q)
            idx = np.searchsorted(self.best_pipeline.classes_, yt)
            covered = sets[np.arange(len(yt)), idx]
            extra = {"avg_set_size": float(sets.sum(axis=1).mean())}
        self.conformal.update({"alpha": alpha, "q": q, "target": 1 - alpha,
                               "observed": float(covered.mean()), **extra})
        return dict(self.conformal)

    def predict_interval(self, input_df: pd.DataFrame, alpha: float = 0.10):
        """Regression: (prediction, lower, upper) with ~(1-alpha) coverage."""
        if self.task != "regression" or self._cal_scores is None:
            raise RuntimeError("Prediction intervals need a trained regression model.")
        pred = self.predict(input_df).astype(float)
        q = self.conformal_quantile(alpha)
        return pred, pred - q, pred + q

    def predict_set(self, input_df: pd.DataFrame, alpha: float = 0.10) -> list[list[str]]:
        """Classification: the set of plausible classes per row, containing the
        true class ~(1-alpha) of the time. A 1-class set = confident; a big set =
        the model is unsure."""
        if self.task != "classification" or self._cal_scores is None:
            raise RuntimeError("Prediction sets need a trained classification model.")
        proba = self.predict_proba(input_df)
        q = self.conformal_quantile(alpha)
        out = []
        for row in proba:
            keep = [self.class_labels[i] for i in np.argsort(-row) if row[i] >= 1 - q]
            out.append(keep or [self.class_labels[int(np.argmax(row))]])
        return out

    def _sanity_notes(self):
        """Flag results that look too good (usually leakage) or unstable."""
        if self.results is None or self.best_model_name is None:
            return
        row = self.results[self.results["Model"] == self.best_model_name].iloc[0]
        if self.task == "regression":
            test, cv = row.get("Test R²"), row.get("CV R² (mean)")
            if pd.notna(test) and test > 0.98:
                self.notes.append(f"Test R² is {test:.3f} — almost perfect. Real data rarely behaves like this; "
                                  f"check whether a feature contains or is derived from the target.")
        else:
            test, cv = row.get("Test Accuracy"), row.get("CV F1 (mean)")
            if pd.notna(test) and test >= 0.99:
                self.notes.append(f"Test accuracy is {test:.1%} — suspiciously high. Check for a feature that "
                                  f"gives the answer away (an ID, a copy of the target, or a later-stage value).")
        key = "Test R²" if self.task == "regression" else "Test F1"
        if pd.notna(cv) and pd.notna(row.get(key)) and abs(row[key] - cv) > 0.15:
            self.notes.append(f"Cross-validation ({cv:.2f}) and holdout ({row[key]:.2f}) disagree by more than "
                              f"0.15 — the estimate is unstable, probably because there is little data.")

    def fairness_report(self, df: pd.DataFrame, group_cols: list[str] | None = None,
                        min_group: int = 10) -> tuple[pd.DataFrame, list[str]]:
        """Per-group performance on the holdout set.

        `df` is the data passed to run(). Groups can be ANY categorical column,
        including ones left out of the model (the usual case for sensitive
        attributes). Returns (table, warnings)."""
        if self._test_pos is None or self._clean_index is None:
            raise RuntimeError("Train a model first.")
        rows_idx = self._clean_index[self._test_pos]
        test_df = df.loc[rows_idx]
        y_true, y_pred = self.holdout["y_true"], self.holdout["y_pred"]
        if group_cols is None:
            group_cols = [c for c in df.columns if c != self.target_col and _is_text_like(df[c])
                          and 2 <= df[c].nunique() <= 8]
        reg = self.task == "regression"
        overall_err = float(np.mean(np.abs(y_true - y_pred))) if reg else float(np.mean(y_true == y_pred))
        pos = 1 if (not reg and len(self.class_labels) == 2) else None
        recs, warn = [], []
        for col in group_cols:
            vals = test_df[col].astype(str).to_numpy()
            per = []
            for g in sorted(set(vals)):
                m = vals == g
                if m.sum() < min_group:
                    continue
                yt, yp = y_true[m], y_pred[m]
                rec = {"Group column": col, "Group": g, "n": int(m.sum())}
                if reg:
                    rec.update({"MAE": float(np.mean(np.abs(yt - yp))),
                                "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
                                "Bias (pred − actual)": float(np.mean(yp - yt))})
                else:
                    rec["Accuracy"] = float(np.mean(yt == yp))
                    if pos is not None:
                        rec["Positive rate"] = float(np.mean(yp == pos))
                        has_pos = (yt == pos).any()
                        rec["True-positive rate"] = float(np.mean(yp[yt == pos] == pos)) if has_pos else np.nan
                per.append(rec)
            recs.extend(per)
            if len(per) < 2:
                continue
            if reg:
                worst = max(per, key=lambda r: r["MAE"]); best = min(per, key=lambda r: r["MAE"])
                if worst["MAE"] > 1.25 * overall_err and worst["MAE"] > 1.25 * best["MAE"]:
                    warn.append(f"Errors are {worst['MAE'] / best['MAE']:.1f}× larger for {col} = {worst['Group']} "
                                f"than for {best['Group']} (MAE {worst['MAE']:.2f} vs {best['MAE']:.2f}).")
            else:
                accs = [r["Accuracy"] for r in per]
                if max(accs) - min(accs) > 0.10:
                    lo = per[int(np.argmin(accs))]; hi = per[int(np.argmax(accs))]
                    warn.append(f"Accuracy differs by {max(accs) - min(accs):.0%} across {col}: "
                                f"{hi['Group']} {hi['Accuracy']:.0%} vs {lo['Group']} {lo['Accuracy']:.0%}.")
                if pos is not None:
                    pr = [r["Positive rate"] for r in per]
                    if max(pr) - min(pr) > 0.10:
                        warn.append(f"The model predicts '{self.class_labels[1]}' at very different rates across {col} "
                                    f"({min(pr):.0%} to {max(pr):.0%}) — check demographic parity.")
                    tp = [r["True-positive rate"] for r in per if pd.notna(r["True-positive rate"])]
                    if len(tp) >= 2 and max(tp) - min(tp) > 0.10:
                        warn.append(f"True-positive rates for '{self.class_labels[1]}' differ by {max(tp) - min(tp):.0%} across {col} — "
                                    f"it is missed more often in some groups (equal opportunity).")
        return pd.DataFrame(recs), warn

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Return predictions in the original label space (the target's own
        values for classification)."""
        X = self._encode_inference_df(input_df)
        preds = self.best_pipeline.predict(X)
        if self.task == "classification" and self.target_col in self.label_encoders:
            preds = self.label_encoders[self.target_col].inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        """Class probabilities (columns follow `self.class_labels`). Raises for
        regression or models without predict_proba."""
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification tasks.")
        inner_model = self.best_pipeline.named_steps["model"]
        if not hasattr(inner_model, "predict_proba"):
            raise RuntimeError(f"{self.best_model_name} does not support probability estimates.")
        X = self._encode_inference_df(input_df)
        return self.best_pipeline.predict_proba(X)

    def get_categorical_options(self, col: str) -> list[str] | None:
        """Original string labels for a categorical feature, or None if numeric."""
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
