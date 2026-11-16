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


def test_model_export_survives_module_reload(tmp_path):
    """Streamlit reloads modules; exporting must not depend on class identity."""
    import importlib
    import numpy as np, pandas as pd
    from modules import ml_engine as me
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    df["y"] = 2 * df["a"] + rng.normal(size=120)
    eng = me.MLEngine(); eng.run(df, "y")
    importlib.reload(me)                         # old instance's class is now a different object
    path = tmp_path / "m.joblib"
    path.write_bytes(eng.to_bytes())
    loaded = me.MLEngine.load(path)
    row = df.drop(columns="y").iloc[:5]
    assert np.allclose(loaded.predict(row), eng.predict(row))


def test_stale_engine_is_adopted_with_state_kept():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    df["y"] = df["a"] + rng.normal(size=100)
    eng = MLEngine(); eng.run(df, "y")

    class Old:                                   # an engine from an older deploy: no to_bytes
        pass
    old = Old(); old.__dict__.update(eng.__dict__)
    new = MLEngine.adopt(old)
    assert isinstance(new, MLEngine) and new.best_model_name == eng.best_model_name
    assert len(new.to_bytes()) > 0
