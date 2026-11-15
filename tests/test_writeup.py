import subprocess
import sys

import pandas as pd
import pytest

from modules import advisor, writeup
from modules.statistics import StatisticalAnalyzer


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    df = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "tips.csv")
    res = StatisticalAnalyzer().run_all(df)
    return df, res, advisor.group_comparisons(df), advisor.correlation_advice(df)


def test_apa_formatting():
    assert writeup.apa_p(0.0004) == "p < .001"
    assert writeup.apa_p(0.0234) == "p = .023"
    assert writeup.apa_r(-0.5) == "-.50" and writeup.apa_r(0.123) == ".12"


def test_writeup_contains_methods_and_every_section(analysed):
    df, res, g, c = analysed
    w = writeup.build_writeup(df, res, g, c)
    assert "Benjamini-Hochberg" in w["methods"] and "244 observations" in w["methods"]
    assert {"Group differences", "Correlations", "Regression"} <= set(w["results"])
    assert "p < .001" in w["markdown"] and "Check every statement" in w["markdown"]


def test_wording_follows_fdr_decision(analysed):
    df, res, g, c = analysed
    row = g.iloc[0].copy()
    row["Sig. after FDR"], row["Significant"] = False, True
    assert "did not survive" in writeup._group_sentence(row)


def test_generated_script_reproduces_the_reported_numbers(analysed, tmp_path):
    df, res, g, c = analysed
    script = writeup.build_script(df.columns, res, g, c)
    (tmp_path / "reproduce.py").write_text(script, encoding="utf-8")
    df.to_csv(tmp_path / "data.csv", index=False)
    run = subprocess.run([sys.executable, "-I", "reproduce.py", "data.csv"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    got = pd.read_csv(tmp_path / "reproduced_results.csv").set_index("analysis")["p_value"]
    for _, r in g.head(writeup.MAX_PER_SECTION).iterrows():
        key = f"{r['Numeric Variable']} by {r['Grouping Variable']}"
        assert got[key] == pytest.approx(r["P-Value"], rel=1e-6, abs=1e-12)
    for _, r in c.head(writeup.MAX_PER_SECTION).iterrows():
        key = f"{r['Variable 1']} ~ {r['Variable 2']}"
        assert got[key] == pytest.approx(r["P-Value"], rel=1e-6, abs=1e-12)
    for _, r in res["chi_square"].iterrows():
        assert got[f"{r['Variable 1']} x {r['Variable 2']}"] == pytest.approx(r["P-Value"], abs=1e-4)
    assert got[[i for i in got.index if i.startswith("OLS")][0]] == pytest.approx(res["ols_model_stats"]["F P-Value"], abs=1e-4)
