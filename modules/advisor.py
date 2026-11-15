"""
Assumption Doctor — picks the *right* test for the data and says why.

For every numeric-vs-group comparison and every numeric pair it checks the
assumptions behind the classic test (normality, equal variances, outliers,
group sizes), switches to a safer alternative when they fail, runs that test,
and explains the decision in plain English. Benjamini-Hochberg FDR is applied
across all comparisons so many tests don't inflate false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

ALPHA = 0.05
MIN_GROUP = 3          # groups smaller than this are ignored
LARGE_GROUP = 30       # at this size the CLT makes mean-based tests robust to mild skew
MAX_LEVELS = 10        # more levels than this is rarely a meaningful grouping


def _group_is_ok(x: np.ndarray) -> tuple[bool, str]:
    """Is the group's distribution close enough to normal for mean-based tests?"""
    n = len(x)
    if np.ptp(x) == 0:
        return False, "constant values"
    skew = float(stats.skew(x))
    if n >= LARGE_GROUP:
        return (abs(skew) <= 2), f"skew {skew:.1f}"
    if n >= 8:
        # Shapiro has little power with so few values, so also guard on skew.
        p = stats.shapiro(x).pvalue
        return (p >= ALPHA and abs(skew) <= 1.5), f"Shapiro p = {p:.3f}, skew {skew:.1f}"
    return False, f"only {n} values"


def _eta_squared(groups) -> float:
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _size_label(value: float, kind: str) -> str:
    v = abs(value)
    cuts = {"eta": (0.01, 0.06, 0.14), "r": (0.1, 0.3, 0.5)}[kind]
    return ("Negligible", "Small", "Medium", "Large")[sum(v >= c for c in cuts)]


def group_comparisons(df: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    """Compare each numeric column across each categorical grouping."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df.columns if c not in num_cols
                and df[c].nunique(dropna=True) in range(2, MAX_LEVELS + 1)]
    rows = []
    for num in num_cols:
        for cat in cat_cols:
            sub = df[[num, cat]].dropna()
            groups = {k: g[num].to_numpy(float) for k, g in sub.groupby(cat, observed=True)
                      if len(g) >= MIN_GROUP}
            if len(groups) < 2:
                continue
            arrs = list(groups.values())
            k = len(arrs)
            n = sum(len(a) for a in arrs)

            ok_flags = [_group_is_ok(a) for a in arrs]
            bad = [(name, why) for (name, a), (ok, why) in zip(groups.items(), ok_flags) if not ok]
            normal_ok = not bad
            try:
                lev_p = float(stats.levene(*arrs, center="median").pvalue)
            except Exception:
                lev_p = np.nan
            equal_var = not (lev_p < ALPHA)

            reasons = []
            if normal_ok:
                reasons.append("each group looks close enough to normal")
            else:
                name, why = bad[0]
                reasons.append(f"group '{name}' departs from normality ({why})"
                               + (f" and {len(bad) - 1} more" if len(bad) > 1 else ""))
            reasons.append("variances are similar" if equal_var else f"variances differ (Levene p = {lev_p:.3f})")

            if normal_ok and equal_var:
                test = "One-way ANOVA"
                stat, p = stats.f_oneway(*arrs)
                df1, df2 = k - 1, n - k
                eff, eff_name = _eta_squared(arrs), "η²"
                eff_label = _size_label(eff, "eta")
            elif normal_ok:
                test = "Welch's ANOVA"
                res = anova_oneway(arrs, use_var="unequal")
                stat, p = float(res.statistic), float(res.pvalue)
                df1, df2 = float(res.df_num), float(res.df_denom)
                eff, eff_name = _eta_squared(arrs), "η²"
                eff_label = _size_label(eff, "eta")
            elif k == 2:
                test = "Mann-Whitney U"
                u = stats.mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
                stat, p = float(u.statistic), float(u.pvalue)
                df1, df2 = len(arrs[0]), len(arrs[1])      # group sizes, reported with U
                eff, eff_name = abs(1 - 2 * stat / (len(arrs[0]) * len(arrs[1]))), "|r|"
                eff_label = _size_label(eff, "r")
            else:
                test = "Kruskal-Wallis"
                h = stats.kruskal(*arrs)
                stat, p = float(h.statistic), float(h.pvalue)
                df1, df2 = k - 1, np.nan
                eff, eff_name = stat * (n + 1) / (n ** 2 - 1), "ε²"
                eff_label = _size_label(eff, "eta")

            rows.append({
                "Numeric Variable": num, "Grouping Variable": cat, "Groups": k, "N": n,
                "Recommended Test": test, "Statistic": round(float(stat), 4),
                "df1": df1, "df2": df2,
                "P-Value": float(p), "Effect Size": round(float(eff), 4),
                "Effect Measure": eff_name, "Effect Label": eff_label,
                "Why": f"Chose {test}: " + "; ".join(reasons) + ".",
                "Assumptions OK": bool(normal_ok and equal_var),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["P-Value (FDR)"] = multipletests(out["P-Value"], method="fdr_bh")[1]
    out["Sig. after FDR"] = out["P-Value (FDR)"] < ALPHA
    out["Significant"] = out["P-Value"] < ALPHA
    # most interesting first: FDR-significant, then largest effect
    out = out.sort_values(["Sig. after FDR", "Effect Size"], ascending=[False, False]).head(max_rows)
    return out.reset_index(drop=True)


def correlation_advice(df: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    """For each numeric pair: Pearson when both variables are well-behaved,
    Spearman when skew or outliers would let a few points drive r."""
    num = df.select_dtypes(include=[np.number])
    rows = []
    for i, a in enumerate(num.columns):
        for b in num.columns[i + 1:]:
            sub = num[[a, b]].dropna()
            if len(sub) < 8 or sub[a].nunique() < 3 or sub[b].nunique() < 3:
                continue
            problems = []
            for col in (a, b):
                x = sub[col].to_numpy(float)
                if abs(stats.skew(x)) > 1:
                    problems.append(f"'{col}' is skewed ({stats.skew(x):.1f})")
                z = np.abs((x - x.mean()) / (x.std() or 1))
                if (z > 4).any():
                    problems.append(f"'{col}' has extreme outliers")
            method = "Spearman" if problems else "Pearson"
            fn = stats.spearmanr if problems else stats.pearsonr
            r, p = fn(sub[a], sub[b])
            rows.append({
                "Variable 1": a, "Variable 2": b, "Recommended": method, "N": len(sub),
                "r": round(float(r), 4), "P-Value": float(p),
                "Effect Label": _size_label(r, "r"),
                "Why": (f"Chose Spearman: {'; '.join(problems)}." if problems
                        else "Chose Pearson: both variables look well-behaved (no strong skew or extreme outliers)."),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["P-Value (FDR)"] = multipletests(out["P-Value"], method="fdr_bh")[1]
    out["Sig. after FDR"] = out["P-Value (FDR)"] < ALPHA
    out = out.reindex(out["r"].abs().sort_values(ascending=False).index).head(max_rows)
    return out.reset_index(drop=True)
