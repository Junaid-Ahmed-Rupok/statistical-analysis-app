"""
Statistician Review: the checks a senior statistician makes *before* trusting results.

Where the Assumption Doctor asks "is this test valid?", the review asks the
questions that come first: was the data collected in a way the tests assume, is
the sample big enough to find anything, and does the model fit the data? Every
finding has a severity, a plain-English explanation of why it matters and what
to do about it. Nothing here claims certainty: these are screening heuristics
that point a human at the right questions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower

HIGH, MEDIUM, LOW, OK = "High", "Medium", "Low", "OK"
_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2, OK: 3}
ALPHA = 0.05


def _f(severity, area, finding, why, action):
    return {"Severity": severity, "Area": area, "Finding": finding, "Why it matters": why, "What to do": action}


# ── power and sample size ───────────────────────────────────────────────
def sample_size(kind: str, effect: float, power: float = 0.8, alpha: float = ALPHA, groups: int = 2) -> float:
    """Required n *per group* (t / anova) or total n (correlation).

    kind: 't' (Cohen's d), 'anova' (Cohen's f), 'corr' (Pearson r).
    """
    if effect <= 0:
        raise ValueError("effect size must be positive")
    if kind == "t":
        return float(np.ceil(TTestIndPower().solve_power(effect, power=power, alpha=alpha)))
    if kind == "anova":
        total = FTestAnovaPower().solve_power(effect, power=power, alpha=alpha, k_groups=groups)
        return float(np.ceil(total / groups))
    if kind == "corr":
        z = np.arctanh(min(effect, 0.999))
        return float(np.ceil(((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / z) ** 2 + 3))
    raise ValueError(f"unknown kind {kind!r}")


def min_detectable_effect(kind: str, n: float, power: float = 0.8, alpha: float = ALPHA, groups: int = 2) -> float:
    """Smallest effect detectable with the given n (per group for t / anova, total for corr)."""
    try:
        if kind == "t":
            return float(TTestIndPower().solve_power(None, nobs1=n, power=power, alpha=alpha))
        if kind == "anova":
            return float(FTestAnovaPower().solve_power(None, nobs=n * groups, power=power, alpha=alpha, k_groups=groups))
    except (ValueError, RuntimeError):
        return float("inf")            # too few observations for the solver: nothing is detectable
    if kind == "corr":
        if n <= 3:
            return 1.0
        z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3)
        return float(np.tanh(z))
    raise ValueError(f"unknown kind {kind!r}")


def power_table(df: pd.DataFrame, max_levels: int = 10) -> pd.DataFrame:
    """For each categorical grouping and numeric correlation set, what effect could this data detect?"""
    rows = []
    cats = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
            if 2 <= df[c].nunique() <= max_levels]
    for c in cats:
        sizes = df[c].value_counts()
        sizes = sizes[sizes >= 2]
        if len(sizes) < 2:
            continue
        k, n = len(sizes), float(sizes.min())
        if k == 2:
            d = min_detectable_effect("t", n)
            rows.append({"Comparison": f"groups of {c}", "Smallest group n": int(n),
                         "Smallest effect detectable (80% power)": round(d, 2) if np.isfinite(d) else "n too small", "Measure": "Cohen's d",
                         "Verdict": _effect_verdict(d, "d")})
        else:
            f = min_detectable_effect("anova", n, groups=k)
            rows.append({"Comparison": f"groups of {c}", "Smallest group n": int(n),
                         "Smallest effect detectable (80% power)": round(f, 2) if np.isfinite(f) else "n too small", "Measure": "Cohen's f",
                         "Verdict": _effect_verdict(f, "f")})
    n_all = int(df.dropna(how="all").shape[0])
    r = min_detectable_effect("corr", n_all)
    rows.append({"Comparison": "correlations (all rows)", "Smallest group n": n_all,
                 "Smallest effect detectable (80% power)": round(r, 2), "Measure": "|r|",
                 "Verdict": _effect_verdict(r, "r")})
    return pd.DataFrame(rows)


def _effect_verdict(v: float, kind: str) -> str:
    small, medium, large = {"d": (0.2, 0.5, 0.8), "f": (0.1, 0.25, 0.4), "r": (0.1, 0.3, 0.5)}[kind]
    if not np.isfinite(v):
        return "Underpowered: even large effects may be missed"
    if v <= small * 1.25:
        return "Sensitive: can find small effects"
    if v <= medium:
        return "Adequate: finds medium effects"
    if v <= large:
        return "Weak: only large effects visible"
    return "Underpowered: even large effects may be missed"


# ── design checks ───────────────────────────────────────────────────────
def _design_checks(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    n = len(df)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 1. Sample size relative to what is being tested
    if n < 30:
        out.append(_f(HIGH, "Sample size", f"Only {n} rows.",
                      "Small samples give unstable estimates and hide real effects; most tests here assume more data.",
                      "Treat every result as exploratory. Collect more data or use exact/permutation tests."))
    elif n < 100:
        out.append(_f(MEDIUM, "Sample size", f"{n} rows is modest.",
                      "Only medium-to-large effects are reliably detectable at this size (see the power table).",
                      "Do not read a non-significant result as 'no effect'."))

    # 2. Repeated measures / clustering: an identifier that repeats
    for c in cat:
        vc = df[c].value_counts()
        if 1 < len(vc) < n * 0.5 and vc.iloc[0] >= 3 and len(vc) > 10 and any(
                k in c.lower() for k in ("id", "subject", "patient", "user", "participant", "person", "customer", "school", "class", "site")):
            out.append(_f(HIGH, "Independence", f"'{c}' repeats ({len(vc)} distinct values over {n} rows).",
                          "Rows from the same unit are correlated. Ordinary tests treat them as independent, "
                          "which makes p-values far too small.",
                          "Use a mixed-effects model or cluster-robust standard errors, or aggregate to one row per unit."))
    # exact duplicate entities
    dup = int(df.duplicated().sum())
    if dup:
        out.append(_f(MEDIUM, "Data integrity", f"{dup} exact duplicate rows remain.",
                      "Duplicates inflate the sample size and make results look more certain than they are.",
                      "Confirm whether they are genuine repeats; remove them if not."))

    # 3. Time order: autocorrelation
    for c in num:
        x = df[c].dropna().to_numpy()
        if len(x) >= 30 and np.ptp(x) > 0:
            r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            se = 1 / np.sqrt(len(x))
            if abs(r1) > 3 * se and abs(r1) > 0.3:
                out.append(_f(MEDIUM, "Independence", f"'{c}' is strongly related to its neighbour (lag-1 r = {r1:.2f}).",
                              "If rows are in time or spatial order, observations are not independent and standard errors are too small.",
                              "If the rows are ordered by time/place, use time-series or spatial methods; if the file was merely sorted, ignore."))
                break

    # 4. Group imbalance / tiny cells
    for c in cat:
        k = df[c].nunique()
        if 2 <= k <= 10:
            vc = df[c].value_counts()
            if vc.min() < 5:
                out.append(_f(MEDIUM, "Group sizes", f"'{c}' has a group with only {int(vc.min())} rows ({vc.idxmin()}).",
                              "Tiny groups produce unreliable means and break chi-square assumptions.",
                              "Merge rare levels or exclude them from that comparison."))
            elif vc.max() / vc.min() > 5:
                out.append(_f(LOW, "Group sizes", f"'{c}' is unbalanced ({int(vc.max())} vs {int(vc.min())} rows).",
                              "Strong imbalance lowers power and makes results sensitive to unequal variances.",
                              "Prefer Welch-type tests (the Assumption Doctor does) and report group sizes."))

    # 5. Multiple testing burden
    pairs = len(num) * (len(num) - 1) // 2
    group_tests = len(num) * sum(1 for c in cat if 2 <= df[c].nunique() <= 10)
    total = pairs + group_tests
    if total > 20:
        out.append(_f(MEDIUM, "Multiple testing", f"About {total} comparisons are possible in this dataset.",
                      f"At alpha = .05 roughly {total * ALPHA:.0f} would look significant by chance alone.",
                      "Use the FDR-adjusted p-values, and state your hypotheses in advance for the ones you care about."))

    # 6. Simpson's paradox screen
    sim = _simpson(df, num, cat)
    out.extend(sim)

    # 7. Missingness that depends on other variables
    out.extend(_missingness(df, num))
    return out


def _simpson(df, num, cat, max_flags: int = 2) -> list[dict]:
    flags = []
    for c in cat:
        if not 2 <= df[c].nunique() <= 6:
            continue
        for i, a in enumerate(num[:8]):
            for b in num[i + 1:8]:
                sub = df[[a, b, c]].dropna()
                if len(sub) < 40:
                    continue
                overall = stats.spearmanr(sub[a], sub[b])[0]
                if abs(overall) < 0.2 or np.isnan(overall):
                    continue
                within = []
                for _, g in sub.groupby(c, observed=True):
                    if len(g) >= 10 and g[a].nunique() > 1 and g[b].nunique() > 1:
                        within.append(stats.spearmanr(g[a], g[b])[0])
                if len(within) >= 2 and all(abs(w) >= 0.1 for w in within) and all(np.sign(w) == -np.sign(overall) for w in within):
                    flags.append(_f(HIGH, "Confounding", f"Simpson's paradox: '{a}' vs '{b}' is {'positive' if overall > 0 else 'negative'} overall "
                                    f"(rho = {overall:.2f}) but reverses inside every level of '{c}'.",
                                    "The overall relationship is produced by how the groups differ, not by the variables themselves.",
                                    f"Analyse '{a}' vs '{b}' within '{c}', or adjust for '{c}' in a regression."))
                    if len(flags) >= max_flags:
                        return flags
    return flags


def _missingness(df: pd.DataFrame, num: list[str]) -> list[dict]:
    out = []
    for c in df.columns:
        miss = df[c].isna()
        share = float(miss.mean())
        if share < 0.05 or share > 0.95:
            continue
        related = []
        for o in num:
            if o == c:
                continue
            a, b = df.loc[miss, o].dropna(), df.loc[~miss, o].dropna()
            if len(a) >= 8 and len(b) >= 8 and a.std() > 0 and b.std() > 0:
                p = stats.mannwhitneyu(a, b).pvalue
                if p < 0.001:
                    related.append(o)
        if related:
            out.append(_f(HIGH, "Missing data", f"Missingness in '{c}' ({share:.0%}) is related to {', '.join(map(repr, related[:3]))}.",
                          "Data are not missing at random, so dropping or mean-filling those rows biases the results.",
                          "Investigate why values are missing; consider multiple imputation or a sensitivity analysis."))
        elif share >= 0.2:
            out.append(_f(MEDIUM, "Missing data", f"'{c}' is {share:.0%} missing.",
                          "A large gap leaves less information than the row count suggests.",
                          "Say how missing values were handled and test whether conclusions change without that column."))
    return out


# ── regression diagnostics ──────────────────────────────────────────────
def regression_diagnostics(df: pd.DataFrame, target: str | None = None) -> list[dict]:
    """Fit OLS of one numeric target on the other numeric columns and check the model's assumptions."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
    from statsmodels.stats.stattools import durbin_watson

    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if df[c].nunique() > 2]
    if len(num) < 2:
        return []
    if target is None:
        target = num[-1]
    X = df[[c for c in num if c != target]].dropna(axis=1, how="all")
    data = pd.concat([df[target], X], axis=1).dropna()
    X = X.loc[data.index]
    X = X.loc[:, X.std() > 0]
    if len(data) < 30 or X.shape[1] == 0 or X.shape[1] > len(data) / 5:
        return []
    Xc = sm.add_constant(X)
    model = sm.OLS(data[target], Xc).fit()
    out = []
    note = f"(model: '{target}' on {X.shape[1]} numeric predictors)"

    try:
        bp = het_breuschpagan(model.resid, Xc)[1]
        if bp < 0.01:
            out.append(_f(MEDIUM, "Model fit", f"Residual spread is not constant (Breusch-Pagan p = {bp:.4f}) {note}.",
                          "Standard errors and p-values from ordinary regression are unreliable.",
                          "Use robust (HC3) standard errors or transform the outcome."))
    except Exception:
        pass
    try:
        rp = linear_reset(model, power=2, use_f=True).pvalue
        if rp < 0.01:
            out.append(_f(MEDIUM, "Model fit", f"The straight-line model misses curvature (RESET p = {rp:.4f}) {note}.",
                          "A real non-linear relationship is being forced into a line.",
                          "Add squared terms, use splines, or try a tree-based model."))
    except Exception:
        pass
    sk = float(stats.skew(model.resid))
    if abs(sk) > 1 and len(data) < 100:
        out.append(_f(LOW, "Model fit", f"Residuals are skewed (skew = {sk:.1f}) {note}.",
                      "With small samples, skewed residuals make confidence intervals unreliable.",
                      "Transform the outcome (e.g. log) or use bootstrap intervals."))
    dw = float(durbin_watson(model.resid))
    if dw < 1.3 or dw > 2.7:
        out.append(_f(LOW, "Independence", f"Residuals are serially correlated (Durbin-Watson = {dw:.2f}) {note}.",
                      "Errors that follow the row order mean observations are not independent.",
                      "If the data has a time order, use a time-series model."))
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    thresh = 4 / len(data)
    bad = int((cooks > thresh * 3).sum())
    if bad:
        out.append(_f(MEDIUM, "Influence", f"{bad} row(s) have outsized influence on the fitted model {note}.",
                      "A handful of points are steering the coefficients; conclusions may not survive their removal.",
                      "Refit without them and report both results."))
    return out


# ── entry point ─────────────────────────────────────────────────────────
def review(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return {'findings': DataFrame, 'power': DataFrame, 'verdict': str, 'counts': dict}."""
    findings = _design_checks(df) + regression_diagnostics(df, target)
    if not findings:
        findings = [_f(OK, "Overall", "No major design or model problems were detected by these screens.",
                       "Screens cannot see how the data was collected.", "Confirm the study design yourself before publishing.")]
    tbl = pd.DataFrame(findings)
    tbl["_o"] = tbl["Severity"].map(_ORDER)
    tbl = tbl.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    counts = tbl["Severity"].value_counts().to_dict()
    if counts.get(HIGH):
        verdict = (f"{counts[HIGH]} serious issue(s) found. Do not trust the headline p-values until these are addressed.")
    elif counts.get(MEDIUM):
        verdict = "No fatal problems, but several issues could change conclusions. Address them before reporting."
    elif counts.get(LOW):
        verdict = "Only minor issues. Results look reasonable; mention the caveats."
    else:
        verdict = "No problems detected by these screens."
    return {"findings": tbl, "power": power_table(df), "verdict": verdict, "counts": counts}
