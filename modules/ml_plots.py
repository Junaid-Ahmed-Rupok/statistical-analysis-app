"""Charts for ML Studio: comparison, ROC / precision-recall / calibration, residuals, learning curve.

Every function takes a trained MLEngine and returns a matplotlib Figure (or None when
the chart cannot be drawn for that data). They never raise on awkward data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import learning_curve

GREEN, GREY, RED = "#34D399", "#999999", "#E5534B"
_MAX_CLASSES = 8


def _clf_ready(engine) -> bool:
    h = engine.holdout or {}
    return engine.task == "classification" and h.get("y_proba") is not None and len(h["y_true"]) > 1


def _present_classes(engine):
    y, proba = engine.holdout["y_true"], engine.holdout["y_proba"]
    k = proba.shape[1]
    return [i for i in range(min(k, _MAX_CLASSES)) if 0 < int((y == i).sum()) < len(y)]


def model_comparison(engine):
    res = engine.results
    if res is None or "Error" in res.columns and res.get("Model") is None:
        return None
    reg = engine.task == "regression"
    mean_c, std_c, label = (("CV R² (mean)", "CV R² (std)", "Cross-validated R²") if reg
                            else ("CV Accuracy (mean)", "CV Accuracy (std)", "Cross-validated accuracy"))
    if mean_c not in res.columns:
        return None
    d = res.dropna(subset=[mean_c]).sort_values(mean_c)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(2.8, 0.55 * len(d) + 1.2)))
    colors = [GREEN if m == engine.best_model_name else "#4B6B5E" for m in d["Model"]]
    ax.barh(d["Model"], d[mean_c], xerr=d[std_c] if std_c in d else None, color=colors, capsize=3)
    ax.set_xlabel(label)
    ax.set_title("Model comparison (error bars = spread across folds)", fontweight="bold")
    fig.tight_layout()
    return fig


def roc(engine):
    if not _clf_ready(engine):
        return None
    y, proba = engine.holdout["y_true"], engine.holdout["y_proba"]
    classes = _present_classes(engine)
    if not classes:
        return None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y, proba[:, 1])
        ax.plot(fpr, tpr, color=GREEN, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        for i in classes:
            fpr, tpr, _ = roc_curve((y == i).astype(int), proba[:, i])
            ax.plot(fpr, tpr, lw=1.6, label=f"{engine.class_labels[i]} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (holdout)", fontweight="bold"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def precision_recall(engine):
    if not _clf_ready(engine):
        return None
    y, proba = engine.holdout["y_true"], engine.holdout["y_proba"]
    classes = _present_classes(engine)
    if not classes:
        return None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if proba.shape[1] == 2:
        p, r, _ = precision_recall_curve(y, proba[:, 1])
        ax.plot(r, p, color=GREEN, lw=2, label=f"AUC = {auc(r, p):.3f}")
        ax.axhline(float(np.mean(y)), ls="--", color=GREY, label="Base rate")
    else:
        for i in classes:
            p, r, _ = precision_recall_curve((y == i).astype(int), proba[:, i])
            ax.plot(r, p, lw=1.6, label=str(engine.class_labels[i]))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve (holdout)", fontweight="bold"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def calibration(engine):
    """Do predicted probabilities match how often the class really happens? (binary only)"""
    if not _clf_ready(engine) or engine.holdout["y_proba"].shape[1] != 2:
        return None
    y, p = engine.holdout["y_true"], engine.holdout["y_proba"][:, 1]
    if len(y) < 20 or len(set(y)) < 2:
        return None
    try:
        frac, mean_pred = calibration_curve(y, p, n_bins=min(8, max(3, len(y) // 10)), strategy="quantile")
    except ValueError:
        return None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfectly calibrated")
    ax.plot(mean_pred, frac, "o-", color=GREEN, lw=2, label="Model")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (holdout)", fontweight="bold"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def residuals(engine):
    """Residual histogram and normal Q-Q plot (regression)."""
    h = engine.holdout
    if engine.task != "regression" or h is None or len(h["y_true"]) < 8:
        return None
    r = np.asarray(h["y_true"], float) - np.asarray(h["y_pred"], float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r, bins=min(30, max(8, len(r) // 5)), color=GREEN, alpha=0.8, edgecolor="white")
    axes[0].axvline(0, ls="--", color=GREY)
    axes[0].set_xlabel("Residual (actual − predicted)"); axes[0].set_ylabel("Rows")
    axes[0].set_title("Residual distribution", fontweight="bold")
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(GREEN); axes[1].get_lines()[1].set_color(GREY)
    axes[1].set_title("Residual Q-Q plot", fontweight="bold")
    fig.tight_layout()
    return fig


def learning_curve_data(engine, points: int = 5):
    """Training vs cross-validated score as the training set grows (the best model only)."""
    X, y = getattr(engine, "_X_train", None), getattr(engine, "_y_train", None)
    if X is None or y is None or engine.best_pipeline is None or len(y) < 40:
        return None
    scoring = "r2" if engine.task == "regression" else "f1_weighted"
    sizes, tr, va = learning_curve(engine.best_pipeline, X, y, cv=engine._cv, scoring=scoring,
                                   train_sizes=np.linspace(0.2, 1.0, points), n_jobs=1, error_score=np.nan)
    return {"sizes": sizes, "train": tr, "val": va, "metric": "R²" if scoring == "r2" else "F1"}


def learning_curve_fig(data):
    if not data:
        return None
    s = data["sizes"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, color, name in (("train", RED, "Training score"), ("val", GREEN, "Cross-validated score")):
        m, sd = np.nanmean(data[key], axis=1), np.nanstd(data[key], axis=1)
        ax.plot(s, m, "o-", color=color, lw=2, label=name)
        ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("Training rows"); ax.set_ylabel(data["metric"])
    ax.set_title("Learning curve", fontweight="bold"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def learning_curve_advice(data) -> str:
    """One sentence of plain-English reading of the learning curve."""
    if not data:
        return ""
    tr, va = np.nanmean(data["train"][-1]), np.nanmean(data["val"][-1])
    gap = tr - va
    slope = np.nanmean(data["val"][-1]) - np.nanmean(data["val"][-2]) if len(data["val"]) > 1 else 0
    if va < 0.5 * max(tr, 1e-9) and gap > 0.2:
        return "Large gap between training and validation: the model is **overfitting**. More data or a simpler model should help."
    if gap > 0.1 and slope > 0.01:
        return "The validation score is still rising: **more data would likely improve** this model."
    if gap <= 0.1 and va < 0.6:
        return "Both scores are low and close together: the model is **underfitting**. Try better features or a more flexible model."
    return "Training and validation scores are close and flat: **more data is unlikely to help much**."
