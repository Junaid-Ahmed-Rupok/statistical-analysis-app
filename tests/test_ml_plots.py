import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from modules import ml_plots as P
from modules.ml_engine import MLEngine

rng = np.random.default_rng(0)


def _engine(kind):
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "g": rng.choice(list("xy"), n)})
    if kind == "reg":
        df["y"] = 2 * df["a"] + rng.normal(size=n)
    elif kind == "bin":
        df["y"] = (df["a"] + rng.normal(size=n) > 0).astype(int)
    else:
        df["y"] = pd.cut(df["a"] + rng.normal(size=n), [-9, -.5, .5, 9], labels=list("abc")).astype(str)
    e = MLEngine(); e.run(df, "y"); return e


@pytest.fixture(scope="module")
def engines():
    return {k: _engine(k) for k in ("reg", "bin", "multi")}


def test_classification_charts(engines):
    for k in ("bin", "multi"):
        for fn in (P.roc, P.precision_recall, P.model_comparison):
            assert fn(engines[k]) is not None
    assert P.calibration(engines["bin"]) is not None
    assert P.calibration(engines["multi"]) is None           # binary only
    assert P.residuals(engines["bin"]) is None


def test_regression_charts(engines):
    assert P.residuals(engines["reg"]) is not None
    assert P.model_comparison(engines["reg"]) is not None
    assert P.roc(engines["reg"]) is None and P.calibration(engines["reg"]) is None


@pytest.mark.parametrize("kind", ["reg", "bin", "multi"])
def test_learning_curve(engines, kind):
    d = P.learning_curve_data(engines[kind])
    assert d and d["train"].shape == d["val"].shape and len(d["sizes"]) == 5
    assert P.learning_curve_fig(d) is not None and P.learning_curve_advice(d)


def test_charts_survive_a_reloaded_engine_without_training_rows(engines):
    e = MLEngine.adopt(engines["bin"])
    e._X_train = None
    assert P.learning_curve_data(e) is None and P.learning_curve_fig(None) is None
    assert P.roc(e) is not None


def test_export_excludes_training_rows(engines):
    assert "_X_train" not in engines["bin"]._payload()["state"]
