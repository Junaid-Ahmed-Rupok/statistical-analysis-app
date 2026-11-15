import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modules import advisor


def _df(seed=0):
    rng = np.random.default_rng(seed)
    n = 120
    g = np.repeat(["a", "b"], n // 2)
    return pd.DataFrame({
        "normal_y": rng.normal(0, 1, n) + (g == "b") * 0.8,
        "skewed_y": rng.exponential(1, n) + (g == "b") * 1.0,
        "unequal_y": np.where(g == "a", rng.normal(0, 1, n), rng.normal(0.5, 6, n)),
        "grp": g,
    })


def test_chooses_test_by_assumptions():
    out = advisor.group_comparisons(_df()).set_index("Numeric Variable")
    assert out.loc["normal_y", "Recommended Test"] == "One-way ANOVA"
    assert out.loc["unequal_y", "Recommended Test"] == "Welch's ANOVA"
    assert out.loc["skewed_y", "Recommended Test"] in ("Mann-Whitney U", "One-way ANOVA")  # n=60/group, mild skew


def test_small_skewed_groups_get_nonparametric_test():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": np.r_[rng.exponential(1, 12), rng.exponential(3, 12)],
                       "g": ["x"] * 12 + ["z"] * 12})
    (row,) = advisor.group_comparisons(df).to_dict("records")
    assert row["Recommended Test"] == "Mann-Whitney U"
    assert "departs from normality" in row["Why"]


def test_three_groups_use_kruskal_when_not_normal():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": np.r_[rng.lognormal(0, 1.2, 10), rng.lognormal(0.5, 1.2, 10), rng.lognormal(1, 1.2, 10)],
                       "g": ["a"] * 10 + ["b"] * 10 + ["c"] * 10})
    (row,) = advisor.group_comparisons(df).to_dict("records")
    assert row["Recommended Test"] == "Kruskal-Wallis" and row["Effect Measure"] == "ε²"


def test_p_value_matches_scipy_and_fdr_is_applied():
    df = _df()
    out = advisor.group_comparisons(df).set_index("Numeric Variable")
    a, b = df[df.grp == "a"].normal_y, df[df.grp == "b"].normal_y
    assert out.loc["normal_y", "P-Value"] == pytest.approx(stats.f_oneway(a, b).pvalue)
    assert (out["P-Value (FDR)"] >= out["P-Value"] - 1e-12).all()


def test_correlation_switches_to_spearman_on_skew():
    rng = np.random.default_rng(3)
    x = rng.exponential(1, 100)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0, .3, 100), "z": rng.normal(size=100), "w": rng.normal(size=100)})
    out = advisor.correlation_advice(df).set_index(["Variable 1", "Variable 2"])
    assert out.loc[("x", "y"), "Recommended"] == "Spearman"
    assert out.loc[("z", "w"), "Recommended"] == "Pearson"


def test_empty_when_nothing_to_compare():
    assert advisor.group_comparisons(pd.DataFrame({"a": [1, 2, 3]})).empty
    assert advisor.correlation_advice(pd.DataFrame({"a": [1, 2, 3]})).empty
