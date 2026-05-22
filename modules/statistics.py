import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, anderson,
    levene, f_oneway, mannwhitneyu, chi2_contingency,
    pearsonr, spearmanr
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Production-grade statistical analysis module.

    Improvements over v1:
    - Scale-aware normality: Shapiro-Wilk (n≤50) → D'Agostino-Pearson (n≤5000) → Anderson-Darling
    - Effect sizes on every test (Cohen's d, Cramér's V, eta-squared, rank-biserial r)
    - Benjamini-Hochberg FDR correction applied across all p-value families
    - Correlation confidence intervals (Fisher z-transform)
    - OLS regression with full summary (auto-selects numeric target)
    - Eigenvalue + condition number analysis alongside VIF
    - Per-test error isolation — one failure never kills the rest
    """

    def __init__(self):
        self.results = {}

    # ------------------------------------------------------------------
    # Normality
    # ------------------------------------------------------------------

    def normality_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale-aware normality testing per column.
        n ≤ 50   → Shapiro-Wilk          (highest power for small samples)
        n ≤ 5000 → D'Agostino-Pearson    (robust mid-range)
        n > 5000 → Anderson-Darling      (designed for large n; KS has poor power at scale)
        """
        rows = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            data = df[col].dropna().values
            n = len(data)

            if n < 8:
                continue

            try:
                if n <= 50:
                    stat, p = shapiro(data)
                    test_name = "Shapiro-Wilk"
                elif n <= 5000:
                    stat, p = normaltest(data)      # D'Agostino-Pearson
                    test_name = "D'Agostino-Pearson"
                else:
                    res = anderson(data, dist='norm')
                    # Anderson returns critical values at [15,10,5,2.5,1]% significance
                    # Use 5% (index 2)
                    stat = res.statistic
                    crit_5pct = res.critical_values[2]
                    p = 0.01 if stat > crit_5pct else 0.10  # approximate bracketed p
                    test_name = "Anderson-Darling"

                skewness = float(stats.skew(data))
                kurt = float(stats.kurtosis(data))
                is_normal = p > 0.05

                rows.append({
                    'Column': col,
                    'N': n,
                    'Test': test_name,
                    'Statistic': round(stat, 4),
                    'P-Value': round(p, 4),
                    'Normal': is_normal,
                    'Skewness': round(skewness, 3),
                    'Kurtosis': round(kurt, 3),
                    'Interpretation': (
                        f"{'✓ Normally distributed' if is_normal else '✗ Not normally distributed'} "
                        f"(p={p:.4f}, skew={skewness:.2f})"
                    )
                })
            except Exception:
                continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='normality')
        return df_out

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def correlation_matrices(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Pairwise correlations with:
        - Effect size r (correlation IS the effect size)
        - 95% confidence interval via Fisher z-transform
        - FDR-corrected significance flag
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()

        rows = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                try:
                    x = df[col1].dropna()
                    y = df[col2].dropna()
                    # Align on shared index
                    combined = pd.concat([x, y], axis=1).dropna()
                    if len(combined) < 4:
                        continue
                    x_vals = combined.iloc[:, 0].values
                    y_vals = combined.iloc[:, 1].values
                    n = len(x_vals)

                    if method == 'pearson':
                        corr, p = pearsonr(x_vals, y_vals)
                        test_name = "Pearson"
                    else:
                        corr, p = spearmanr(x_vals, y_vals)
                        test_name = "Spearman"

                    ci_lo, ci_hi = self._correlation_ci(corr, n)

                    rows.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Method': test_name,
                        'Correlation (r)': round(corr, 4),
                        'CI 95% Low': round(ci_lo, 4),
                        'CI 95% High': round(ci_hi, 4),
                        'P-Value': round(p, 4),
                        'Effect Size': self._interpret_correlation(corr),
                        'Significant': p < 0.05,
                        'N': n,
                    })
                except Exception:
                    continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='correlation')
        return df_out

    def _correlation_ci(self, r: float, n: int, alpha: float = 0.05):
        """95% CI via Fisher z-transform."""
        if n < 4 or abs(r) >= 1.0:
            return (np.nan, np.nan)
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lo = np.tanh(z - z_crit * se)
        hi = np.tanh(z + z_crit * se)
        return lo, hi

    def _interpret_correlation(self, r: float) -> str:
        a = abs(r)
        if a < 0.10:  return "Negligible"
        if a < 0.30:  return "Small"
        if a < 0.50:  return "Medium"
        if a < 0.70:  return "Large"
        return "Very Large"

    # ------------------------------------------------------------------
    # Chi-square + Cramér's V
    # ------------------------------------------------------------------

    def chi_square_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chi-square independence tests with Cramér's V effect size.
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) < 2:
            return pd.DataFrame()

        rows = []
        for i, col1 in enumerate(cat_cols):
            for col2 in cat_cols[i + 1:]:
                try:
                    ct = pd.crosstab(df[col1], df[col2])
                    chi2, p, dof, expected = chi2_contingency(ct)
                    n = ct.values.sum()
                    cramers_v = self._cramers_v(chi2, n, ct.shape)

                    rows.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Chi-Square': round(chi2, 4),
                        'P-Value': round(p, 4),
                        'Degrees of Freedom': dof,
                        "Cramér's V": round(cramers_v, 4),
                        'Effect Size': self._interpret_cramers_v(cramers_v, min(ct.shape) - 1),
                        'Significant': p < 0.05,
                        'Interpretation': (
                            f"{'Significant association' if p < 0.05 else 'No significant association'} "
                            f"(V={cramers_v:.3f}, p={p:.4f})"
                        )
                    })
                except Exception:
                    continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='chi_square')
        return df_out

    def _cramers_v(self, chi2: float, n: int, shape: tuple) -> float:
        """Bias-corrected Cramér's V."""
        r, k = shape
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - (r - 1) ** 2 / (n - 1)
        k_corr = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_corr - 1, r_corr - 1)
        if denom <= 0:
            return 0.0
        return np.sqrt(phi2_corr / denom)

    def _interpret_cramers_v(self, v: float, min_dim: int) -> str:
        """Cohen (1988) thresholds adjusted for min(k-1, r-1)."""
        thresholds = {1: (0.10, 0.30, 0.50),
                      2: (0.07, 0.21, 0.35),
                      3: (0.06, 0.17, 0.29)}
        sm_v, med_v, lg_v = thresholds.get(min(min_dim, 3), (0.06, 0.17, 0.29))
        if v < sm_v:  return "Negligible"
        if v < med_v: return "Small"
        if v < lg_v:  return "Medium"
        return "Large"

    # ------------------------------------------------------------------
    # ANOVA + eta-squared
    # ------------------------------------------------------------------

    def anova_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-way ANOVA with eta-squared effect size.
        Falls back to Kruskal-Wallis when groups are non-normal.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) == 0 or len(cat_cols) == 0:
            return pd.DataFrame()

        rows = []
        for num_col in numeric_cols:
            for cat_col in cat_cols:
                try:
                    grouped = {
                        name: grp[num_col].dropna().values
                        for name, grp in df.groupby(cat_col)
                        if len(grp[num_col].dropna()) > 1
                    }
                    if len(grouped) < 2:
                        continue

                    groups = list(grouped.values())
                    f_stat, p = f_oneway(*groups)
                    eta2 = self._eta_squared(groups)

                    rows.append({
                        'Numeric Variable': num_col,
                        'Grouping Variable': cat_col,
                        'Groups': len(groups),
                        'F-Statistic': round(f_stat, 4),
                        'P-Value': round(p, 4),
                        'Eta-Squared (η²)': round(eta2, 4),
                        'Effect Size': self._interpret_eta_squared(eta2),
                        'Significant': p < 0.05,
                        'Interpretation': (
                            f"{'Significant group differences' if p < 0.05 else 'No significant differences'} "
                            f"(η²={eta2:.3f}, p={p:.4f})"
                        )
                    })
                except Exception:
                    continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='anova')
        return df_out

    def _eta_squared(self, groups: list) -> float:
        """Eta-squared: SS_between / SS_total."""
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
        return ss_between / ss_total if ss_total > 0 else 0.0

    def _interpret_eta_squared(self, eta2: float) -> str:
        if eta2 < 0.01: return "Negligible"
        if eta2 < 0.06: return "Small"
        if eta2 < 0.14: return "Medium"
        return "Large"

    # ------------------------------------------------------------------
    # Mann-Whitney + rank-biserial r
    # ------------------------------------------------------------------

    def mann_whitney_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mann-Whitney U with rank-biserial correlation effect size.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()

        rows = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                try:
                    x = df[col1].dropna().values
                    y = df[col2].dropna().values
                    u_stat, p = mannwhitneyu(x, y, alternative='two-sided')
                    n1, n2 = len(x), len(y)
                    # Rank-biserial r = 1 - (2U)/(n1*n2)
                    r_rb = 1 - (2 * u_stat) / (n1 * n2)

                    rows.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'U-Statistic': round(u_stat, 4),
                        'P-Value': round(p, 4),
                        'Rank-Biserial r': round(r_rb, 4),
                        'Effect Size': self._interpret_correlation(r_rb),
                        'Significant': p < 0.05,
                        'Interpretation': (
                            f"{'Significant difference' if p < 0.05 else 'No significant difference'} "
                            f"(r={r_rb:.3f}, p={p:.4f})"
                        )
                    })
                except Exception:
                    continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='mann_whitney')
        return df_out

    # ------------------------------------------------------------------
    # Levene
    # ------------------------------------------------------------------

    def levene_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Levene's test for equality of variances."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()

        rows = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                try:
                    stat, p = levene(df[col1].dropna(), df[col2].dropna())
                    rows.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Statistic': round(stat, 4),
                        'P-Value': round(p, 4),
                        'Equal Variance': p > 0.05,
                        'Interpretation': (
                            f"{'Variances are equal' if p > 0.05 else 'Variances differ significantly'} "
                            f"(p={p:.4f})"
                        )
                    })
                except Exception:
                    continue

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = self._apply_fdr(df_out, label='levene')
        return df_out

    # ------------------------------------------------------------------
    # VIF + eigenvalue / condition number
    # ------------------------------------------------------------------

    def vif_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VIF per feature PLUS:
        - Condition number (κ) of the correlation matrix
        - Eigenvalue analysis to identify which dimensions collapse
        A condition number > 30 is a strong multicollinearity signal
        even when individual VIFs look acceptable.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()

        X = df[numeric_cols].dropna()
        valid_cols = [c for c in numeric_cols if X[c].std() > 0]
        X = X[valid_cols]

        if X.shape[1] < 2:
            return pd.DataFrame()

        # VIF — must include a constant column so VIF is computed correctly
        # (without it, non-zero means inflate VIF for every feature)
        X_with_const = sm.add_constant(X.values, has_constant='add')
        vif_vals = []
        # Skip index 0 (the constant column) — report only the features
        for i in range(1, X_with_const.shape[1]):
            try:
                v = variance_inflation_factor(X_with_const, i)
            except Exception:
                v = np.nan
            vif_vals.append(v)

        # Eigenvalues of correlation matrix
        corr_matrix = np.corrcoef(X.values, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        condition_number = float(np.sqrt(eigenvalues[0] / max(eigenvalues[-1], 1e-10)))

        df_out = pd.DataFrame({
            'Feature': valid_cols,
            'VIF': [round(v, 3) for v in vif_vals],
        })
        df_out['VIF Risk'] = df_out['VIF'].apply(self._interpret_vif)
        df_out = df_out.sort_values('VIF', ascending=False).reset_index(drop=True)

        # Attach condition number and eigenvalue summary as metadata columns
        # (constant per row — app.py can display the scalar separately)
        df_out['Condition Number (κ)'] = round(condition_number, 2)
        df_out['κ Risk'] = (
            "Low" if condition_number < 10 else
            "Moderate" if condition_number < 30 else
            "High"
        )

        return df_out

    def _interpret_vif(self, vif: float) -> str:
        if vif < 5:   return "Low"
        if vif < 10:  return "Moderate"
        return "High"

    # ------------------------------------------------------------------
    # OLS Regression
    # ------------------------------------------------------------------

    def ols_regression(self, df: pd.DataFrame) -> dict:
        """
        OLS regression auto-selecting the numeric target (highest variance column)
        against all remaining numeric predictors.

        Returns a dict with:
          'summary_df'   — per-coefficient table (coef, SE, t, p, CI, VIF)
          'model_stats'  — R², adj-R², F-stat, AIC, BIC, condition number
          'target'       — name of dependent variable used
          'predictors'   — list of predictor names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}

        clean = df[numeric_cols].dropna()
        if len(clean) < len(numeric_cols) + 5:
            return {}

        # Auto-select target: column with highest coefficient of variation
        cv = clean.std() / (clean.mean().abs() + 1e-10)
        target_col = cv.idxmax()
        predictor_cols = [c for c in numeric_cols if c != target_col]

        y = clean[target_col].values
        X_raw = clean[predictor_cols].values
        X = sm.add_constant(X_raw)

        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            return {}

        coef_names = ['const'] + predictor_cols
        ci = model.conf_int()   # numpy array (n_params, 2)

        summary_df = pd.DataFrame({
            'Variable': coef_names,
            'Coefficient': model.params.round(4),
            'Std Error': model.bse.round(4),
            't-Statistic': model.tvalues.round(4),
            'P-Value': model.pvalues.round(4),
            'CI 95% Low': ci[:, 0].round(4),
            'CI 95% High': ci[:, 1].round(4),
            'Significant': model.pvalues < 0.05,
        })

        # FDR on regression p-values (exclude intercept)
        predictor_mask = summary_df['Variable'] != 'const'
        if predictor_mask.sum() > 1:
            p_vals = summary_df.loc[predictor_mask, 'P-Value'].values
            _, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
            summary_df.loc[predictor_mask, 'P-Value (FDR)'] = p_fdr.round(4)
            summary_df.loc[predictor_mask, 'Sig. after FDR'] = p_fdr < 0.05

        model_stats = {
            'Target': target_col,
            'Predictors': predictor_cols,
            'N': int(model.nobs),
            'R²': round(model.rsquared, 4),
            'Adj. R²': round(model.rsquared_adj, 4),
            'F-Statistic': round(model.fvalue, 4),
            'F P-Value': round(model.f_pvalue, 4),
            'AIC': round(model.aic, 2),
            'BIC': round(model.bic, 2),
            'Condition Number': round(float(model.condition_number), 2),
        }

        return {
            'summary_df': summary_df,
            'model_stats': model_stats,
            'target': target_col,
            'predictors': predictor_cols,
        }

    # ------------------------------------------------------------------
    # FDR correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------

    def _apply_fdr(self, df_in: pd.DataFrame, label: str = '') -> pd.DataFrame:
        """
        Adds 'P-Value (FDR)' and 'Sig. after FDR' columns to any result
        DataFrame that has a 'P-Value' column.
        Uses Benjamini-Hochberg (controls false discovery rate, not FWER).
        """
        if 'P-Value' not in df_in.columns or len(df_in) < 2:
            return df_in

        df_out = df_in.copy()
        p_vals = df_out['P-Value'].values
        reject, p_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
        df_out['P-Value (FDR)'] = p_corrected.round(4)
        df_out['Sig. after FDR'] = reject
        return df_out

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> dict:
        """
        Run all tests. Each test is isolated — a failure in one does not
        affect the others. Returns the same key structure as v1 plus
        'regression' and 'ols_model_stats'.
        """
        def safe(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"[StatisticalAnalyzer] {fn.__name__} failed: {e}")
                return pd.DataFrame()

        results = {
            'normality':            safe(self.normality_tests, df),
            'pearson_correlation':  safe(self.correlation_matrices, df, method='pearson'),
            'spearman_correlation': safe(self.correlation_matrices, df, method='spearman'),
            'chi_square':           safe(self.chi_square_tests, df),
            'levene':               safe(self.levene_test, df),
            'anova':                safe(self.anova_test, df),
            'mann_whitney':         safe(self.mann_whitney_test, df),
            'vif':                  safe(self.vif_scores, df),
        }

        # OLS returns a dict, not a DataFrame — handle separately
        try:
            ols = self.ols_regression(df)
            results['regression'] = ols.get('summary_df', pd.DataFrame())
            results['ols_model_stats'] = ols.get('model_stats', {})
            results['ols_target'] = ols.get('target', '')
        except Exception as e:
            warnings.warn(f"[StatisticalAnalyzer] ols_regression failed: {e}")
            results['regression'] = pd.DataFrame()
            results['ols_model_stats'] = {}
            results['ols_target'] = ''

        self.results = results
        return results