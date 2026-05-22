import pandas as pd
import numpy as np

class InsightsGenerator:
    """
    Professional-grade insights engine.
    Generates actionable, executive-level insights like a senior data scientist.
    """
    
    def __init__(self):
        self.insights = {}
        
    def generate(self, df, stats_results, vif_df, overview_dict):
        insights = {
            'summary': [],
            'warnings': [],
            'highlights': [],
            'recommendation': ''
        }
        
        # ── DATA QUALITY ASSESSMENT ─────────────────────────
        quality_score = overview_dict['quality_score']
        missing_pct = overview_dict['missing_percentage']
        dup_pct = overview_dict.get('duplicate_percentage', 0)
        
        if quality_score >= 90:
            insights['highlights'].append(
                f"✅ Dataset integrity is excellent ({quality_score:.0f}/100). "
                f"Minimal cleaning required before modeling."
            )
        elif quality_score >= 70:
            insights['summary'].append(
                f"Dataset quality is acceptable ({quality_score:.0f}/100). "
                f"Address the flagged issues before production deployment."
            )
        else:
            insights['warnings'].append(
                f"⚠️ Data quality score is low ({quality_score:.0f}/100). "
                f"Significant preprocessing required before reliable analysis."
            )
        
        if missing_pct > 5:
            insights['warnings'].append(
                f"Missing data detected in {missing_pct:.1f}% of cells. "
                f"Investigate whether missingness is MCAR, MAR, or MNAR before imputation."
            )
        elif missing_pct > 0:
            insights['summary'].append(
                f"Minor missing values ({missing_pct:.1f}%). "
                f"KNN imputation applied — verify that imputed values maintain column distributions."
            )
        
        # ── FEATURE DISTRIBUTION ANALYSIS ───────────────────
        if not stats_results.get('normality', pd.DataFrame()).empty:
            normality = stats_results['normality']
            total_cols = len(normality)
            normal_cols = normality[normality['Normal'] == True]
            non_normal = normality[normality['Normal'] == False]
            
            if len(non_normal) > total_cols * 0.7:
                insights['summary'].append(
                    f"📊 {len(non_normal)}/{total_cols} features deviate from normality. "
                    f"Consider log-transform, Box-Cox, or Yeo-Johnson for features with |skew| > 1. "
                    f"Use non-parametric tests (Spearman, Mann-Whitney) for these variables."
                )
            
            # Highlight highly skewed features
            highly_skewed = non_normal[non_normal['Skewness'].abs() > 1.5] if 'Skewness' in non_normal.columns else pd.DataFrame()
            if not highly_skewed.empty:
                skewed_names = ', '.join(highly_skewed['Column'].head(5).tolist())
                insights['warnings'].append(
                    f"Highly skewed features detected (|skew| > 1.5): {skewed_names}. "
                    f"These may distort linear models and inflate Type I error rates."
                )
        
        # ── CORRELATION ANALYSIS ────────────────────────────
        pearson = stats_results.get('pearson_correlation', pd.DataFrame())
        if not pearson.empty:
            strong_corrs = pearson[pearson['Correlation (r)' if 'Correlation (r)' in pearson.columns else 'Correlation'].abs() > 0.7]
            
            if len(strong_corrs) > 0:
                corr_col = 'Correlation (r)' if 'Correlation (r)' in pearson.columns else 'Correlation'
                top_corr = strong_corrs.nlargest(1, corr_col).iloc[0]
                
                insights['highlights'].append(
                    f"🔗 Strongest linear relationship: **{top_corr['Variable 1']} ↔ {top_corr['Variable 2']}** "
                    f"(r = {top_corr[corr_col]:.3f}, "
                    f"95% CI: [{top_corr.get('CI 95% Low', 'N/A')}, {top_corr.get('CI 95% High', 'N/A')}]). "
                    f"{'This is expected in housing/economic data.' if 'bedroom' in str(top_corr['Variable 1']).lower() or 'house' in str(top_corr['Variable 1']).lower() else ''}"
                )
                
                if len(strong_corrs) > 3:
                    insights['warnings'].append(
                        f"Multicollinearity risk: {len(strong_corrs)} feature pairs have |r| > 0.7. "
                        f"This inflates standard errors in regression and makes coefficient interpretation unreliable."
                    )
        
        # ── MULTICOLLINEARITY (VIF) ──────────────────────────
        if not vif_df.empty:
            high_vif = vif_df[vif_df['VIF'] > 10]
            moderate_vif = vif_df[(vif_df['VIF'] >= 5) & (vif_df['VIF'] <= 10)]
            
            if len(high_vif) > 0:
                high_names = ', '.join(high_vif['Feature'].head(5).tolist())
                insights['warnings'].append(
                    f"Severe multicollinearity: {len(high_vif)} features with VIF > 10 ({high_names}). "
                    f"Recommended actions: (1) Remove redundant features, (2) Apply PCA, "
                    f"(3) Use Ridge/Lasso regression with regularization."
                )
            
            # Condition number
            if 'Condition Number (κ)' in vif_df.columns:
                kappa = vif_df['Condition Number (κ)'].iloc[0]
                if kappa > 30:
                    insights['warnings'].append(
                        f"Numerical instability detected: Condition number κ = {kappa:.1f} (> 30). "
                        f"Standardize predictors before regression. The design matrix is near-singular."
                    )
                elif kappa > 10:
                    insights['summary'].append(
                        f"Moderate collinearity: κ = {kappa:.1f}. Standardization recommended before OLS."
                    )
        
        # ── ANOVA & GROUP EFFECTS ───────────────────────────
        anova = stats_results.get('anova', pd.DataFrame())
        if not anova.empty:
            sig_anova = anova[anova['Significant'] == True]
            large_effects = sig_anova[sig_anova['Effect Size'].isin(['Large', 'Very Large'])] if 'Effect Size' in sig_anova.columns else pd.DataFrame()
            
            if not large_effects.empty:
                top_effect = large_effects.nlargest(1, 'Eta-Squared (η²)' if 'Eta-Squared (η²)' in large_effects.columns else large_effects.columns[0]).iloc[0]
                insights['highlights'].append(
                    f"📏 Largest group effect: **{top_effect['Numeric Variable']}** varies significantly "
                    f"by **{top_effect['Grouping Variable']}** "
                    f"(η² = {top_effect.get('Eta-Squared (η²)', 'N/A')}, {top_effect.get('Effect Size', '')}). "
                    f"This variable is a strong candidate for stratified analysis."
                )
            
            if len(sig_anova) > 0:
                insights['summary'].append(
                    f"{len(sig_anova)}/{len(anova)} group comparisons show significant differences (FDR-corrected). "
                    f"Consider these grouping variables for feature engineering."
                )
        
        # ── OLS REGRESSION ───────────────────────────────────
        ols_stats = stats_results.get('ols_model_stats', {})
        if ols_stats:
            r2 = ols_stats.get('R²', 0)
            f_pval = ols_stats.get('F P-Value', 1)
            kappa_ols = ols_stats.get('Condition Number', 0)
            
            if r2 > 0.7:
                insights['highlights'].append(
                    f"🎯 OLS model explains {r2*100:.1f}% of variance in **{ols_stats.get('Target', 'target')}** "
                    f"(R² = {r2:.3f}, Adj. R² = {ols_stats.get('Adj. R²', 0):.3f}). "
                    f"Model is {'statistically significant' if f_pval < 0.05 else 'not significant'}."
                )
            elif r2 > 0.4:
                insights['summary'].append(
                    f"OLS model shows moderate fit (R² = {r2:.3f}). "
                    f"Consider adding interaction terms or polynomial features."
                )
            else:
                insights['summary'].append(
                    f"Low R² ({r2:.3f}) — predictors explain limited variance. "
                    f"Consider feature engineering, non-linear models (Random Forest, GBM), or collecting additional data."
                )
            
            if kappa_ols > 1000:
                insights['warnings'].append(
                    f"⚠️ OLS condition number = {kappa_ols:.0f} — severe numerical instability. "
                    f"Standardize all predictors before regression. Results may be unreliable."
                )
        
        # ── OUTLIER ANALYSIS ────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_flags = 0
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > len(df) * 0.05:
                outlier_flags += 1
        
        if outlier_flags > 2:
            insights['summary'].append(
                f"🔍 {outlier_flags} features have > 5% outliers. "
                f"Winsorization applied. For production: investigate if outliers are genuine extreme values or data entry errors."
            )
        
        # ── FDR ANALYSIS ─────────────────────────────────────
        if not pearson.empty and 'Sig. after FDR' in pearson.columns:
            sig_after_fdr = pearson[pearson['Sig. after FDR'] == True]
            sig_before = pearson[pearson['Significant'] == True]
            if len(sig_before) > len(sig_after_fdr):
                insights['summary'].append(
                    f"FDR correction reduced significant correlations from {len(sig_before)} → {len(sig_after_fdr)}. "
                    f"This protects against false positives in multiple testing."
                )
        
        # ── FINAL RECOMMENDATION ─────────────────────────────
        recommendations = []
        
        if len(insights['warnings']) >= 3:
            recommendations.append(
                "**Priority Actions:** Address multicollinearity and numerical instability before model deployment. "
                "Use regularization (Ridge/Lasso) or dimensionality reduction (PCA)."
            )
        
        if 'multicollinearity' in str(insights['warnings']).lower() or 'vif' in str(insights['warnings']).lower():
            recommendations.append(
                "Drop one feature from each highly correlated pair (|r| > 0.85). "
                "Retain the feature with higher business relevance or lower VIF."
            )
        
        if missing_pct > 3:
            recommendations.append(
                "Investigate missing data mechanism. If NMAR, consider multiple imputation (MICE) rather than simple median/mode."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Dataset is analysis-ready. Proceed with model development. "
                "Monitor for data drift in production."
            )
        
        insights['recommendation'] = ' '.join(recommendations)
        
        # ── READINESS SCORE ──────────────────────────────────
        readiness = self._calculate_readiness(quality_score, insights, vif_df)
        insights['readiness_score'] = readiness
        
        self.insights = insights
        return insights
    
    def _calculate_readiness(self, quality_score, insights, vif_df):
        """Calculate production readiness score (0-100)."""
        score = quality_score * 0.4
        
        # Penalize warnings
        score -= len(insights['warnings']) * 8
        
        # Bonus for highlights
        score += len(insights['highlights']) * 5
        
        # VIF penalty
        if not vif_df.empty:
            high_vif_count = len(vif_df[vif_df['VIF'] > 10])
            score -= high_vif_count * 5
        
        return max(0, min(score, 100))
