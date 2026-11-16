import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from modules import review as R

from pathlib import Path
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_sample_size_matches_textbook():
    assert R.sample_size("t", 0.5) == 64          # Cohen: d=.5, 80% power, alpha .05
    assert R.sample_size("corr", 0.3) == 85
    assert R.sample_size("anova", 0.25, groups=3) == 53


def test_min_detectable_is_inverse_of_sample_size():
    assert R.min_detectable_effect("t", 64) == pytest.approx(0.5, abs=0.01)
    assert R.min_detectable_effect("corr", 85) == pytest.approx(0.3, abs=0.01)


def test_bad_effect_rejected():
    with pytest.raises(ValueError):
        R.sample_size("t", 0)


def test_flags_simpson_and_repeated_measures():
    rng = np.random.default_rng(1)
    g = np.repeat(["a", "b", "c"], 60)
    x = rng.normal(0, 1, 180) + np.repeat([0, 4, 8], 60)
    y = -x + np.repeat([0, 12, 24], 60) + rng.normal(0, .5, 180)
    df = pd.DataFrame({"x": x, "y": y, "grp": g, "patient_id": np.tile(np.arange(30), 6).astype(str)})
    f = R.review(df)["findings"]
    assert {"Independence", "Confounding"} <= set(f["Area"])
    assert f.iloc[0]["Severity"] == "High"


def test_flags_tiny_sample_and_non_random_missingness():
    assert R.review(pd.DataFrame({"a": range(10), "b": range(10)}))["findings"].iloc[0]["Area"] == "Sample size"
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0, 1, 300)
    b[a > 0.8] = np.nan                     # missing depends on a
    f = R.review(pd.DataFrame({"a": a, "b": b, "c": rng.normal(0, 1, 300)}))["findings"]
    assert (f["Area"] == "Missing data").any() and (f["Severity"] == "High").any()


def test_clean_random_data_has_no_serious_findings():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400),
                       "g": rng.choice(["x", "y"], 400)})
    assert R.review(df)["counts"].get("High", 0) == 0


def test_tab_renders():
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    next(b for b in at.button if b.key == "preprocess_btn").click(); at.run(timeout=240)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Statistician Review" in m.value for m in at.markdown)
    assert any("You need about" in s.value for s in at.success)
