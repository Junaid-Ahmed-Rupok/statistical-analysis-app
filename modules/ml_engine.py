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
    accuracy_score, f1_score, mean_absolute_error,
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

# Models that are sensitive to feature scale and need a StandardScaler.
# Tree-based models are scale-invariant and should NOT be scaled.
_SCALING_REQUIRED = frozenset({"Linear Regression", "Logistic Regression"})


def _make_pipeline(name: str, model) -> Pipeline:
    """Wrap a model in a scaling Pipeline only when the algorithm needs it."""
    if name in _SCALING_REQUIRED:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])


class MLEngine:
    """
    AutoML pipeline supporting regression and classification.

    Typical usage::

        engine = MLEngine()
        results_df = engine.run(df, target="price")
        predictions = engine.predict(new_df)
        probabilities = engine.predict_proba(new_df)  # classification only
    """

    # ------------------------------------------------------------------
    # Construction & reset
    # ------------------------------------------------------------------

    def __init__(self):
        self._reset()

    def _reset(self):
        """Wipe all fitted state so the engine can be re-used cleanly."""
        self.task: str | None = None
        self.target_col: str | None = None
        self.feature_cols: list[str] = []
        self.pipelines: dict[str, Pipeline] = {}
        self.results: pd.DataFrame | None = None
        self.best_pipeline: Pipeline | None = None
        self.best_model_name: str | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.confusion_matrices: dict[str, np.ndarray] = {}
        # Tracks which feature columns were reclassified from numeric →
        # categorical during training, so predict() applies the same logic
        # without re-detecting from the inference DataFrame's dtypes.
        self._reclassified_cols: set[str] = set()

    # ------------------------------------------------------------------
    # Task detection
    # ------------------------------------------------------------------

    def detect_task(self, df: pd.DataFrame, target: str) -> str:
        """
        Infer whether *target* is a regression or classification problem.

        Rules (in priority order):
        1. Boolean or category dtype → classification.
        2. Object (string) dtype     → classification.
        3. ≤ 15 unique values        → classification.
        4. Unique-value ratio < 5 %  → classification (e.g. integer codes).
        5. Otherwise                 → regression.
        """
        col   = df[target].dropna()
        dtype = col.dtype

        if dtype == "bool" or hasattr(dtype, "categories"):
            return "classification"
        if dtype == "object":
            return "classification"
        if col.nunique() <= 15:
            return "classification"
        if col.nunique() / len(col) < 0.05:
            return "classification"
        return "regression"

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        In-place-safe imputation:
        - Numeric columns  → median (robust to outliers).
        - Categorical cols → mode (most frequent value).
        """
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ["object", "category"]:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        return df

    def _resolve_mixed_type_columns(
        self, df: pd.DataFrame, *, training: bool
    ) -> pd.DataFrame:
        """
        Handle columns whose declared dtype is numeric but which contain
        non-numeric strings (e.g. "Level 1", "N/A").

        During **training** (`training=True`):
            Detects affected columns by attempting coercion, logs a warning
            for each, records them in `self._reclassified_cols`, and casts
            them to str so downstream LabelEncoder picks them up correctly.

        During **inference** (`training=False`):
            Uses the set recorded at training time — does NOT re-detect from
            the inference DataFrame's dtypes, which may differ from training.
        """
        df = df.copy()

        if training:
            self._reclassified_cols = set()
            for col in self.feature_cols:
                if df[col].dtype in ["object", "category"]:
                    continue
                coerced  = pd.to_numeric(df[col], errors="coerce")
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
            # Apply the same reclassification decided at training time.
            for col in self._reclassified_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        return df

    def _encode_inference_df(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Shared preprocessing path used by both predict() and predict_proba().
        Validates columns, resolves mixed types, imputes, encodes, and
        returns a float numpy array ready for pipeline.predict / predict_proba.
        """
        if self.best_pipeline is None:
            raise RuntimeError("No trained model available. Call .run() first.")

        missing = set(self.feature_cols) - set(input_df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        df_clean = input_df[self.feature_cols].copy()
        # Use training-recorded reclassification — not re-detected from dtypes.
        df_clean = self._resolve_mixed_type_columns(df_clean, training=False)
        df_clean = self._impute(df_clean)

        for col in self.feature_cols:
            if col in self.label_encoders:
                le  = self.label_encoders[col]
                raw = df_clean[col].astype(str)
                known = raw.isin(le.classes_)
                if not known.all():
                    unknown = raw[~known].unique().tolist()
                    warnings.warn(
                        f"Column '{col}' contains values unseen during training: "
                        f"{unknown}. Encoding as the most frequent training class.",
                        UserWarning,
                        stacklevel=3,
                    )
                    raw = raw.where(known, le.classes_[0])
                df_clean[col] = le.transform(raw)
            elif df_clean[col].dtype in ["object", "category"]:
                warnings.warn(
                    f"Column '{col}' is categorical but has no fitted encoder. "
                    "Substituting 0.",
                    UserWarning,
                    stacklevel=3,
                )
                df_clean[col] = 0

        return df_clean.values.astype(float)

    def prepare_data(
        self, df: pd.DataFrame, target: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolve mixed types → impute → encode → return (X, y).

        Label encoders are stored on self so predict() applies the same
        transformation at inference time.
        """
        self.target_col  = target
        self.feature_cols = [c for c in df.columns if c != target]

        # Drop rows where the target is missing — we cannot train on them.
        df_clean = df.dropna(subset=[target]).copy()

        # Reclassify mixed-type feature columns BEFORE imputation so the
        # correct strategy (mode vs median) is used for each column.
        df_clean = self._resolve_mixed_type_columns(df_clean, training=True)
        df_clean = self._impute(df_clean)

        if self.task == "classification" and df_clean[target].dtype == "object":
            le = LabelEncoder()
            df_clean[target] = le.fit_transform(df_clean[target].astype(str))
            self.label_encoders[target] = le

        for col in self.feature_cols:
            if df_clean[col].dtype in ["object", "category"]:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le

        X = df_clean[self.feature_cols].values.astype(float)
        y = df_clean[target].values
        return X, y

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------

    def _build_models(self) -> dict:
        if self.task == "regression":
            return {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0,
                ),
                "CatBoost": CatBoostRegressor(
                    n_estimators=100, depth=6, learning_rate=0.1,
                    random_seed=42, verbose=0,
                ),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
            }
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0, eval_metric="logloss",
            ),
            "CatBoost": CatBoostClassifier(
                n_estimators=100, depth=6, learning_rate=0.1,
                random_seed=42, verbose=0,
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, random_state=42, algorithm="SAMME"
            ),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Train all models and return a comparison DataFrame.

        The engine is fully reset at the start of each call so it is safe
        to call run() multiple times on the same instance.
        """
        self._reset()

        self.task = self.detect_task(df, target)
        logger.info("Detected task: %s (target=%r)", self.task, target)

        X, y = self.prepare_data(df, target)

        test_size = 0.15 if len(X) < 100 else 0.2
        stratify  = y if self.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        cv = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if self.task == "classification"
            else KFold(n_splits=5, shuffle=True, random_state=42)
        )

        models = self._build_models()
        self.pipelines = {
            name: _make_pipeline(name, model) for name, model in models.items()
        }

        results = []
        for name, pipe in self.pipelines.items():
            try:
                row = self._train_and_evaluate(
                    name, pipe, X_train, X_test, y_train, y_test, cv
                )
            except Exception as exc:
                logger.error("Model %r failed: %s", name, exc, exc_info=True)
                row = {"Model": name, "Error": str(exc)}
            results.append(row)

        self.results = pd.DataFrame(results)
        self._select_best_model()
        self._extract_feature_importance()
        return self.results

    def _train_and_evaluate(
        self, name, pipe, X_train, X_test, y_train, y_test, cv
    ) -> dict:
        """Fit pipe, compute CV + held-out metrics, return a result dict."""
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
            self.confusion_matrices[name] = pd.crosstab(
                y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
            )
            return {
                "Model":               name,
                "CV Accuracy (mean)":  round(float(cv_scores.mean()), 4),
                "CV Accuracy (std)":   round(float(cv_scores.std()), 4),
                "Test Accuracy":       round(float(accuracy_score(y_test, y_pred)), 4),
                "Test Precision":      round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "Test Recall":         round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "Test F1":             round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }

    def _select_best_model(self):
        """Pick the best pipeline based on the primary held-out metric."""
        metric = "Test R²" if self.task == "regression" else "Test F1"
        valid  = self.results.dropna(subset=[metric])

        if valid.empty:
            raise RuntimeError(
                "All models failed during training. Check the logs for details."
            )

        best_idx             = valid[metric].idxmax()
        self.best_model_name = self.results.loc[best_idx, "Model"]
        self.best_pipeline   = self.pipelines[self.best_model_name]
        logger.info(
            "Best model: %s (held-out %s = %s)",
            self.best_model_name, metric, self.results.loc[best_idx, metric],
        )

    def _extract_feature_importance(self):
        """Pull feature importances / coefficients from the best model step."""
        if self.best_pipeline is None:
            return

        model = self.best_pipeline.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef        = model.coef_
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        else:
            return

        self.feature_importance = (
            pd.DataFrame({"Feature": self.feature_cols, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model found by run().

        Parameters
        ----------
        input_df:
            DataFrame with the same feature columns used during training.
            The target column should not be present.

        Returns
        -------
        np.ndarray
            Predicted values. Classification targets are decoded back to
            their original string labels when applicable.
        """
        X     = self._encode_inference_df(input_df)
        preds = self.best_pipeline.predict(X)

        if self.task == "classification" and self.target_col in self.label_encoders:
            preds = self.label_encoders[self.target_col].inverse_transform(
                preds.astype(int)
            )
        return preds

    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Return class-probability estimates (classification only).

        Raises RuntimeError for regression tasks or models that do not
        support probability estimates (e.g. some AdaBoost configurations).
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks."
            )
        if not hasattr(self.best_pipeline, "predict_proba"):
            raise RuntimeError(
                f"{self.best_model_name} does not support probability estimates."
            )

        # _encode_inference_df handles validation, reclassification, imputation,
        # and encoding in one place — no duplicated preprocessing.
        X = self._encode_inference_df(input_df)
        return self.best_pipeline.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Persist the entire engine to path using joblib."""
        joblib.dump(self, path)
        logger.info("Engine saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MLEngine":
        """
        Load a previously saved engine from path.

        Raises TypeError if the file does not contain an MLEngine instance,
        guarding against accidentally loading an unrelated pickle.
        """
        engine = joblib.load(path)
        if not isinstance(engine, cls):
            raise TypeError(
                f"Expected MLEngine, got {type(engine).__name__}. "
                "The file may be corrupt or from an incompatible version."
            )
        logger.info("Engine loaded from %s", path)
        return engine
