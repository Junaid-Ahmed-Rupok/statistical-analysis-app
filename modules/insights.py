import pandas as pd
import numpy as np


class InsightsGenerator:
    """
    Insights generator — updated to match statistics.py v2 column names.

    Key fixes vs v1:
    - 'Correlation' → 'Correlation (r)'
    - FDR-aware significance: reads 'Sig. after FDR' when present, falls back to 'Significant'
    - Condition number insight from VIF table
    - OLS regression insight (R², significant predictors)
    - Effect size summary across tests
    - Skewness insight from normality table
    """

    def __init__(self):
        self.insights = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        stats_results: dict,
        vif_df: pd.DataFrame,
        overview_dict: dict
    ) -> dict:

        insights = {
            'summary': [],
            'warnings': [],
            'highlights': [],
            'recommendation': ''
        }

        self._data_quality_insights(insights, overview_dict)
        self._normality_insights(insights, stats_results)
        self._correlation_insights(insights, stats_results)
        self._vif_insights(insights, vif_df)
        self._anova_insights(insights, stats_results)
        self._regression_insights(insights, stats_results)
        self._outlier_insights(insights, df)
        self._effect_size_summary(insights, stats_results)

        insights['recommendation'] = self._generate_recommendations(
            insights, overview_dict['quality_score']
        )
        insights['readiness_score'] = self._calculate_readiness(
            overview_dict['quality_score'], insights
        )

        self.insights = insights
        return insights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sig_col(self, df: pd.DataFrame) -> pd.Series:
        """
        Return the best available significance column.
        Prefers FDR-corrected when present.
        """
        if 'Sig. after FDR' in df.columns:
            return df['Sig. after FDR']
        return df['Significant']

    def _data_quality_insights(self, insights: dict, overview: dict) -> None:
        quality = overview['quality_score']

        if quality >= 80:
            insights['highlights'].append(
                f"Excellent data quality score: {quality:.1f}/100"
            )
        elif quality >= 50:
            insights['warnings'].append(
                f"Moderate data quality score: {quality:.1f}/100 — consider deeper cleaning"
            )
        else:
            insights['warnings'].append(
                f"Low data quality score: {quality:.1f}/100 — significant cleaning required"
            )

        missing_pct = overview['missing_percentage']
        if missing_pct > 10:
            insights['warnings'].append(
                f"High missing data rate: {missing_pct:.1f}% — imputation or row removal needed"
            )
        elif missing_pct > 0:
            insights['summary'].append(
                f"Dataset contains {missing_pct:.1f}% missing values — imputation recommended"
            )
        else:
            insights['highlights'].append("No missing values — dataset is complete")

        dup_pct = overview.get('duplicate_percentage', 0)
        if dup_pct > 5:
            insights['warnings'].append(
                f"High duplicate rate: {dup_pct:.1f}% duplicate rows detected"
            )

    def _normality_insights(self, insights: dict, stats_results: dict) -> None:
        norm_df = stats_results.get('normality', pd.DataFrame())
        if norm_df.empty:
            return

        normal_count = int((norm_df['Normal'] == True).sum())
        total = len(norm_df)
        non_normal_count = total - normal_count

        if normal_count >= non_normal_count:
            insights['highlights'].append(
                f"{normal_count}/{total} columns are normally distributed"
            )
        else:
            insights['summary'].append(
                f"{non_normal_count}/{total} columns are non-normal — "
                f"non-parametric tests recommended"
            )

        # Skewness flags — only available in v2
        if 'Skewness' in norm_df.columns:
            highly_skewed = norm_df[norm_df['Skewness'].abs() > 1]
            if not highly_skewed.empty:
                cols = ', '.join(highly_skewed['Column'].tolist())
                insights['summary'].append(
                    f"High skewness (|skew| > 1) in: {cols} — log or Box-Cox transform may help"
                )

    def _correlation_insights(self, insights: dict, stats_results: dict) -> None:
        corr_df = stats_results.get('pearson_correlation', pd.DataFrame())
        if corr_df.empty:
            return

        # v2 uses 'Correlation (r)', v1 used 'Correlation' — handle both
        corr_col = 'Correlation (r)' if 'Correlation (r)' in corr_df.columns else 'Correlation'

        strong = corr_df[corr_df[corr_col].abs() > 0.7]

        if not strong.empty:
            top = strong.nlargest(1, corr_col).iloc[0]
            ci_note = ''
            if 'CI 95% Low' in corr_df.columns:
                ci_note = f" [95% CI: {top['CI 95% Low']:.2f}, {top['CI 95% High']:.2f}]"

            insights['highlights'].append(
                f"Strongest correlation: {top['Variable 1']} ↔ {top['Variable 2']} "
                f"(r = {top[corr_col]:.2f}){ci_note}"
            )

            if len(strong) > 3:
                insights['warnings'].append(
                    f"{len(strong)} strongly correlated pairs (|r| > 0.7) — "
                    f"potential multicollinearity"
                )

        # FDR-corrected significant pairs
        sig = self._sig_col(corr_df)
        sig_count = int(sig.sum())
        total = len(corr_df)
        if sig_count == 0:
            insights['summary'].append(
                "No significant correlations after FDR correction"
            )
        elif sig_count < total:
            insights['summary'].append(
                f"{sig_count}/{total} correlations remain significant after FDR correction"
            )

    def _vif_insights(self, insights: dict, vif_df: pd.DataFrame) -> None:
        if vif_df.empty:
            return

        high_vif = vif_df[vif_df['VIF'] > 10]
        moderate_vif = vif_df[(vif_df['VIF'] >= 5) & (vif_df['VIF'] <= 10)]

        if not high_vif.empty:
            insights['warnings'].append(
                f"Severe multicollinearity in {len(high_vif)} feature(s) (VIF > 10): "
                f"{', '.join(high_vif['Feature'].tolist())}"
            )
        elif not moderate_vif.empty:
            insights['summary'].append(
                f"Moderate multicollinearity in {len(moderate_vif)} feature(s) (VIF 5–10)"
            )
        else:
            insights['highlights'].append(
                "All VIF scores below 5 — no multicollinearity concerns"
            )

        # Condition number insight — only in v2
        if 'Condition Number (κ)' in vif_df.columns:
            kappa = vif_df['Condition Number (κ)'].iloc[0]
            k_risk = vif_df['κ Risk'].iloc[0]
            if k_risk == 'High':
                insights['warnings'].append(
                    f"Condition number κ = {kappa:.1f} (> 30) — matrix near-singular, "
                    f"regression coefficients may be unstable"
                )
            elif k_risk == 'Moderate':
                insights['summary'].append(
                    f"Condition number κ = {kappa:.1f} — moderate numerical instability"
                )

    def _anova_insights(self, insights: dict, stats_results: dict) -> None:
        anova_df = stats_results.get('anova', pd.DataFrame())
        if anova_df.empty:
            return

        sig = self._sig_col(anova_df)
        sig_df = anova_df[sig]

        if not sig_df.empty:
            # Report largest effect size if available
            if 'Eta-Squared (η²)' in sig_df.columns:
                top = sig_df.nlargest(1, 'Eta-Squared (η²)').iloc[0]
                insights['highlights'].append(
                    f"Strongest group effect: {top['Numeric Variable']} by {top['Grouping Variable']} "
                    f"(η² = {top['Eta-Squared (η²)']:.3f}, {top.get('Effect Size', '')})"
                )
            else:
                insights['highlights'].append(
                    f"{len(sig_df)} significant group differences found (ANOVA, p < 0.05)"
                )

    def _regression_insights(self, insights: dict, stats_results: dict) -> None:
        model_stats = stats_results.get('ols_model_stats', {})
        reg_df = stats_results.get('regression', pd.DataFrame())

        if not model_stats or reg_df.empty:
            return

        r2 = model_stats.get('R²', 0)
        adj_r2 = model_stats.get('Adj. R²', 0)
        target = model_stats.get('Target', 'target')
        f_p = model_stats.get('F P-Value', 1.0)

        if f_p < 0.05:
            insights['highlights'].append(
                f"OLS model for '{target}' is significant "
                f"(R² = {r2:.3f}, Adj. R² = {adj_r2:.3f})"
            )
        else:
            insights['summary'].append(
                f"OLS model for '{target}' is not significant "
                f"(R² = {r2:.3f}) — predictors may not explain variance well"
            )

        # Significant predictors (excluding intercept)
        pred_df = reg_df[reg_df['Variable'] != 'const']
        sig_col = 'Sig. after FDR' if 'Sig. after FDR' in pred_df.columns else 'Significant'
        sig_preds = pred_df[pred_df[sig_col] == True]['Variable'].tolist()

        if sig_preds:
            insights['highlights'].append(
                f"Significant predictors of '{target}': {', '.join(sig_preds)}"
            )

        kappa = model_stats.get('Condition Number', 0)
        if kappa > 1000:
            insights['warnings'].append(
                f"OLS condition number = {kappa:.0f} — severe numerical instability, "
                f"consider scaling or removing correlated predictors"
            )

    def _outlier_insights(self, insights: dict, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                outlier_counts[col] = n_out

        if not outlier_counts:
            return

        total = sum(outlier_counts.values())
        top_col = max(outlier_counts, key=outlier_counts.get)

        if total > len(df) * 0.1:
            insights['warnings'].append(
                f"High outlier count: {total} values across {len(outlier_counts)} columns — "
                f"most affected: {top_col} ({outlier_counts[top_col]})"
            )
        else:
            insights['summary'].append(
                f"{total} outliers across {len(outlier_counts)} columns "
                f"(most in '{top_col}')"
            )

    def _effect_size_summary(self, insights: dict, stats_results: dict) -> None:
        """Summarise how many tests found medium-or-larger effects after FDR correction."""
        large_effects = []

        corr_df = stats_results.get('pearson_correlation', pd.DataFrame())
        if not corr_df.empty and 'Effect Size' in corr_df.columns:
            sig = self._sig_col(corr_df)
            n = int(corr_df[sig & corr_df['Effect Size'].isin(['Medium', 'Large', 'Very Large'])].shape[0])
            if n:
                large_effects.append(f"{n} correlation pair(s)")

        anova_df = stats_results.get('anova', pd.DataFrame())
        if not anova_df.empty and 'Effect Size' in anova_df.columns:
            sig = self._sig_col(anova_df)
            n = int(anova_df[sig & anova_df['Effect Size'].isin(['Medium', 'Large'])].shape[0])
            if n:
                large_effects.append(f"{n} ANOVA result(s)")

        if large_effects:
            insights['highlights'].append(
                f"Practically meaningful effects (medium+) found in: {', '.join(large_effects)}"
            )

    # ------------------------------------------------------------------
    # Recommendation + readiness
    # ------------------------------------------------------------------

    def _generate_recommendations(self, insights: dict, quality_score: float) -> str:
        parts = []

        if quality_score < 70:
            parts.append(
                "Improve data quality through cleaning and imputation before advanced analysis."
            )

        warning_texts = ' '.join(insights['warnings']).lower()

        if 'multicollinearity' in warning_texts or 'vif' in warning_texts:
            parts.append(
                "Address multicollinearity via feature selection, PCA, or removing redundant columns."
            )

        if 'condition number' in warning_texts or 'unstable' in warning_texts:
            parts.append(
                "Scale or standardise predictors to reduce numerical instability in regression."
            )

        if 'skewness' in ' '.join(insights['summary']).lower():
            parts.append(
                "Apply log or Box-Cox transforms to skewed columns before parametric modelling."
            )

        if len(insights['warnings']) > 3:
            parts.append(
                "Multiple issues detected — consider a systematic preprocessing pipeline."
            )
        elif insights['warnings']:
            parts.append(
                "Address the identified warnings to improve analysis reliability."
            )

        if not parts:
            parts.append(
                "Dataset looks ready for advanced analysis. Proceed with modelling or hypothesis testing."
            )

        return ' '.join(parts)

    def _calculate_readiness(self, quality_score: float, insights: dict) -> float:
        readiness = quality_score * 0.5
        readiness -= len(insights['warnings']) * 5
        readiness += len(insights['highlights']) * 3
        return float(max(0, min(readiness, 100)))