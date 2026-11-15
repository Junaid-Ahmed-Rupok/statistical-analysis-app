"""
Thesis-ready write-up and reproducible script.

`build_writeup` turns the analysis results into APA-style Methods and Results
paragraphs. `build_script` emits a standalone Python script that recomputes
every reported number from the cleaned data, so a reader (or a reviewer) can
check the text against the data.
"""

from __future__ import annotations

import importlib.metadata as md
import json
from typing import Optional

import numpy as np
import pandas as pd

ALPHA = 0.05
MAX_PER_SECTION = 12
_V = "Cram\u00e9r's V"      # column name (kept out of f-strings for Python 3.11)
_R2, _ADJ = "R\u00b2", "Adj. R\u00b2"


# ── APA formatting helpers ───────────────────────────────────────────────────

def apa_p(p: float) -> str:
    if p is None or pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


def apa_num(x: float, dec: int = 2) -> str:
    return f"{x:,.{dec}f}"


def apa_r(x: float) -> str:
    """Correlations / proportions bounded by 1 drop the leading zero."""
    s = f"{x:.2f}"
    return s.replace("0.", ".", 1).replace("-0.", "-.", 1)


def _eff(label: str) -> str:
    return f" ({str(label).lower()} effect)" if label and str(label) != "Negligible" else " (negligible effect)"


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "unknown"


# ── Sentences ────────────────────────────────────────────────────────────────

def _group_sentence(r: pd.Series) -> str:
    num, cat, test = r["Numeric Variable"], r["Grouping Variable"], r["Recommended Test"]
    sig = bool(r["Sig. after FDR"])
    verb = "differed significantly" if sig else "did not differ significantly"
    where = "between" if r["Groups"] == 2 else "across"
    if test in ("One-way ANOVA", "Welch's ANOVA"):
        name = "A one-way ANOVA" if test == "One-way ANOVA" else "Welch's ANOVA"
        d2 = apa_num(r["df2"], 0) if test == "One-way ANOVA" else apa_num(r["df2"], 1)
        stat = f"F({apa_num(r['df1'], 0)}, {d2}) = {apa_num(r['Statistic'])}"
    elif test == "Mann-Whitney U":
        name = "A Mann-Whitney U test"
        stat = f"U = {apa_num(r['Statistic'], 1)}, n₁ = {int(r['df1'])}, n₂ = {int(r['df2'])}"
    else:
        name = "A Kruskal-Wallis test"
        stat = f"H({int(r['df1'])}) = {apa_num(r['Statistic'])}"
    eff = f"{r['Effect Measure']} = {apa_r(r['Effect Size'])}"
    s = (f"{name} showed that {num} {verb} {where} levels of {cat}, {stat}, "
         f"{apa_p(r['P-Value'])}, {eff}{_eff(r['Effect Label'])}.")
    if sig != bool(r["Significant"]):
        s += f" (This did not survive false-discovery-rate correction, adjusted {apa_p(r['P-Value (FDR)'])}.)"
    elif sig and r["P-Value (FDR)"] >= 0.001:
        s = s[:-1] + f"; FDR-adjusted {apa_p(r['P-Value (FDR)'])}."
    return s


def _corr_sentence(r: pd.Series) -> str:
    sym = "r" if r["Recommended"] == "Pearson" else "rₛ"
    sig = bool(r["Sig. after FDR"])
    direction = "positive" if r["r"] > 0 else "negative"
    if not sig:
        return (f"{r['Variable 1']} and {r['Variable 2']} were not significantly correlated, "
                f"{sym}({int(r['N']) - 2}) = {apa_r(r['r'])}, {apa_p(r['P-Value'])}.")
    return (f"{r['Variable 1']} and {r['Variable 2']} showed a {str(r['Effect Label']).lower()} {direction} "
            f"{r['Recommended']} correlation, {sym}({int(r['N']) - 2}) = {apa_r(r['r'])}, "
            f"{apa_p(r['P-Value'])}.")


def _chi_sentence(r: pd.Series) -> str:
    sig = bool(r.get("Sig. after FDR", r["Significant"]))
    n = r.get("N")
    n_txt = f", N = {int(n)}" if n is not None and not pd.isna(n) else ""
    verb = "were significantly associated" if sig else "were not significantly associated"
    return (f"{r['Variable 1']} and {r['Variable 2']} {verb}, "
            f"χ²({int(r['Degrees of Freedom'])}{n_txt}) = {apa_num(r['Chi-Square'])}, "
            f"{apa_p(r['P-Value'])}, Cram\u00e9r's V = {apa_r(r[_V])}.")


def _regression_paragraph(stats_results: dict) -> Optional[str]:
    ms = stats_results.get("ols_model_stats") or {}
    reg = stats_results.get("regression")
    if not ms or reg is None or getattr(reg, "empty", True):
        return None
    k = len(ms["Predictors"])
    sig_model = ms["F P-Value"] < ALPHA
    text = (f"A multiple linear regression predicting {ms['Target']} from {k} predictor"
            f"{'s' if k != 1 else ''} was {'statistically significant' if sig_model else 'not statistically significant'}, "
            f"F({k}, {ms['N'] - k - 1}) = {apa_num(ms['F-Statistic'])}, {apa_p(ms['F P-Value'])}, "
            f"R² = {apa_r(ms['R²'])}, adjusted R² = {apa_r(ms['Adj. R²'])}.")
    coefs = reg[reg["Variable"] != "const"]
    sig_col = "Sig. after FDR" if "Sig. after FDR" in coefs.columns else "Significant"
    sig_rows = coefs[coefs[sig_col].fillna(False).astype(bool)]
    if len(sig_rows):
        parts = [f"{r['Variable']} (B = {apa_num(r['Coefficient'], 3)}, 95% CI "
                 f"[{apa_num(r['CI 95% Low'], 3)}, {apa_num(r['CI 95% High'], 3)}], {apa_p(r['P-Value'])})"
                 for _, r in sig_rows.iterrows()]
        text += " Significant predictors were " + "; ".join(parts) + "."
    else:
        text += " No individual predictor was significant."
    return text


# ── Public: text ─────────────────────────────────────────────────────────────

def build_writeup(df: pd.DataFrame, stats_results: dict, groups: pd.DataFrame,
                  corrs: pd.DataFrame) -> dict:
    """Return {'methods': str, 'results': {section: [sentences]}, 'markdown': str}."""
    n_rows, n_cols = df.shape
    n_num = df.select_dtypes(include=[np.number]).shape[1]
    tests = []
    if groups is not None and not groups.empty:
        tests.append("one-way ANOVA, Welch's ANOVA, Mann-Whitney U or Kruskal-Wallis tests for group "
                     "comparisons (chosen per comparison from the assumption checks)")
    if corrs is not None and not corrs.empty:
        tests.append("Pearson or Spearman correlations")
    tests.append("chi-square tests of independence for categorical pairs")
    reg_ms = stats_results.get("ols_model_stats")
    if reg_ms:
        tests.append("an ordinary least squares regression")

    methods = (
        f"The analysis used {n_rows:,} observations on {n_cols} variables ({n_num} numeric). "
        f"Assumptions were checked for every comparison: normality of each group (Shapiro-Wilk for small "
        f"groups; skewness for groups of 30 or more, where mean-based tests are robust to mild departures) "
        f"and homogeneity of variance (Levene's test, median-centred). Where assumptions held, classical "
        f"tests were used; otherwise Welch's ANOVA or rank-based tests were used. The analyses comprised "
        f"{'; '.join(tests)}. Pairwise deletion was used for missing values. Effect sizes are reported with "
        f"each test (η² or ε² for group differences, r for correlations, Cramér's V "
        f"for associations). Because many tests were run, p-values were adjusted within each family with the "
        f"Benjamini-Hochberg false-discovery-rate procedure, and statements of significance refer to adjusted "
        f"values at α = .05. Analyses were run in Python (pandas {_version('pandas')}, "
        f"SciPy {_version('scipy')}, statsmodels {_version('statsmodels')})."
    )

    results: dict[str, list[str]] = {}
    if groups is not None and not groups.empty:
        results["Group differences"] = [_group_sentence(r) for _, r in groups.head(MAX_PER_SECTION).iterrows()]
        extra = len(groups) - MAX_PER_SECTION
        if extra > 0:
            results["Group differences"].append(f"{extra} further comparisons are in the accompanying table.")
    if corrs is not None and not corrs.empty:
        results["Correlations"] = [_corr_sentence(r) for _, r in corrs.head(MAX_PER_SECTION).iterrows()]
    chi = stats_results.get("chi_square")
    if chi is not None and not getattr(chi, "empty", True):
        ordered = chi.sort_values("P-Value").head(MAX_PER_SECTION)
        results["Associations between categorical variables"] = [_chi_sentence(r) for _, r in ordered.iterrows()]
    reg = _regression_paragraph(stats_results)
    if reg:
        results["Regression"] = [reg]

    md_lines = ["## Methods", "", methods, "", "## Results", ""]
    for section, sents in results.items():
        md_lines += [f"### {section}", "", " ".join(sents), ""]
    md_lines += ["*Draft generated by StatsPro. Check every statement against your research question "
                 "before submitting; automated analysis cannot judge causality or study design.*"]
    return {"methods": methods, "results": results, "markdown": "\n".join(md_lines)}


# ── Public: reproducible script ──────────────────────────────────────────────

_HEADER = '''"""
Reproduces the numbers in the StatsPro write-up.

Usage:   python reproduce.py cleaned_data.csv
Needs:   pip install pandas numpy scipy statsmodels
Writes:  reproduced_results.csv  (also printed)
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
import statsmodels.api as sm

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "cleaned_data.csv")
rows = []


def groups_of(num, cat, min_n=3):
    sub = df[[num, cat]].dropna()
    return [g[num].to_numpy(float) for _, g in sub.groupby(cat, observed=True) if len(g) >= min_n]

'''


def build_script(df_columns, stats_results: dict, groups: pd.DataFrame, corrs: pd.DataFrame,
                 max_items: int = MAX_PER_SECTION) -> str:
    out = [_HEADER]
    if groups is not None and not groups.empty:
        out.append("# ── Group comparisons (test chosen by the Assumption Doctor) ──")
        for _, r in groups.head(max_items).iterrows():
            num, cat, test = r["Numeric Variable"], r["Grouping Variable"], r["Recommended Test"]
            call = {
                "One-way ANOVA": "stats.f_oneway(*g)",
                "Welch's ANOVA": "anova_oneway(g, use_var='unequal')",
                "Mann-Whitney U": "stats.mannwhitneyu(g[0], g[1], alternative='two-sided')",
                "Kruskal-Wallis": "stats.kruskal(*g)",
            }[test]
            out.append(f"g = groups_of({num!r}, {cat!r})\nres = {call}\n"
                       f"rows.append(dict(analysis={'%s by %s' % (num, cat)!r}, test={test!r}, "
                       f"statistic=float(res.statistic), p_value=float(res.pvalue)))\n")
    if corrs is not None and not corrs.empty:
        out.append("# ── Correlations ──")
        for _, r in corrs.head(max_items).iterrows():
            fn = "pearsonr" if r["Recommended"] == "Pearson" else "spearmanr"
            a, b = r["Variable 1"], r["Variable 2"]
            out.append(f"sub = df[[{a!r}, {b!r}]].dropna()\nres = stats.{fn}(sub[{a!r}], sub[{b!r}])\n"
                       f"rows.append(dict(analysis={'%s ~ %s' % (a, b)!r}, test={r['Recommended']!r}, "
                       f"statistic=float(res[0]), p_value=float(res[1])))\n")
    chi = stats_results.get("chi_square")
    if chi is not None and not getattr(chi, "empty", True):
        out.append("# ── Chi-square tests of independence ──")
        for _, r in chi.sort_values("P-Value").head(max_items).iterrows():
            a, b = r["Variable 1"], r["Variable 2"]
            out.append(f"res = stats.chi2_contingency(pd.crosstab(df[{a!r}], df[{b!r}]))\n"
                       f"rows.append(dict(analysis={'%s x %s' % (a, b)!r}, test='Chi-square', "
                       f"statistic=float(res[0]), p_value=float(res[1])))\n")
    ms = stats_results.get("ols_model_stats")
    if ms:
        preds = list(ms["Predictors"])
        out.append("# ── OLS regression ──")
        out.append(f"clean = df[[{ms['Target']!r}] + {preds!r}].dropna()\n"
                   f"model = sm.OLS(clean[{ms['Target']!r}], sm.add_constant(clean[{preds!r}])).fit()\n"
                   f"rows.append(dict(analysis='OLS: {ms['Target']} ~ ' + {' + '.join(preds)!r}, test='F-test (model)', "
                   f"statistic=float(model.fvalue), p_value=float(model.f_pvalue)))\n")
    out.append("results = pd.DataFrame(rows)\nprint(results.to_string(index=False))\n"
               "results.to_csv('reproduced_results.csv', index=False)\n")
    return "\n".join(out)
