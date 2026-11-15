import numpy as np
import pandas as pd
import pytest

from modules.ml_engine import MLEngine


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    g = rng.choice(["a", "b"], n)
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "g": g, "y": y,
                         "cls": (x + rng.normal(0, .8, n) > 0).astype(int)})


def test_conformal_interval_coverage_is_close_to_target(data):
    e = MLEngine(); e.run(data, "y", features=["x"])
    s = e.conformal_summary(0.10)
    assert abs(s["observed"] - 0.90) < 0.08
    pred, lo, hi = e.predict_interval(data.head(5))
    assert (lo < pred).all() and (pred < hi).all()
    assert e.conformal_summary(0.2)["q"] < s["q"]          # lower coverage -> narrower


def test_conformal_sets_cover_and_vary(data):
    e = MLEngine(); e.run(data, "cls", features=["x"])
    s = e.conformal_summary(0.10)
    assert s["observed"] >= 0.80
    sets = e.predict_set(data.head(200))
    assert all(len(x) >= 1 for x in sets) and {len(x) for x in sets} >= {1, 2}


def test_fairness_flags_a_group_with_worse_errors():
    rng = np.random.default_rng(1)
    n = 800
    g = rng.choice(["a", "b"], n)
    x = rng.normal(size=n)
    y = 3 * x + np.where(g == "a", rng.normal(0, .3, n), rng.normal(0, 3, n))   # group b is noisy
    df = pd.DataFrame({"x": x, "g": g, "y": y})
    e = MLEngine(); e.run(df, "y", features=["x"])           # group column NOT a feature
    table, warns = e.fairness_report(df)
    assert set(table["Group"]) == {"a", "b"}
    assert any("larger" in w and "g = b" in w for w in warns)


def test_too_good_to_be_true_note(data):
    d = data.copy(); d["copy"] = d["y"] * 1.0001
    e = MLEngine(); e.run(d, "y", features=["x", "copy"])
    assert any("almost perfect" in n for n in e.notes)
