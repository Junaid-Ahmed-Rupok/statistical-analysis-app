import pandas as pd
import numpy as np

class InsightsGenerator:
    """
    Enterprise-grade insights engine.
    Generates comprehensive, executive-level analysis reports.
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
        
        quality_score = overview_dict['quality_score']
        missing_pct = overview_dict['missing_percentage']
        dup_pct = overview_dict.get('duplicate_percentage', 0)
        n_rows = overview_dict['rows']
        n_cols = overview_dict['columns']
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 1: EXECUTIVE DATA QUALITY ASSESSMENT
        # ═══════════════════════════════════════════════════════════
        
        if quality_score >= 95:
            insights['highlights'].append(
                f"✅ **Exceptional Data Quality ({quality_score:.0f}/100):** "
                f"The dataset ({n_rows:,} rows × {n_cols} columns) is near-production-ready. "
                f"Minimal preprocessing was required. This level of quality is rare — "
                f"proceed directly to exploratory analysis and feature engineering."
            )
        elif quality_score >= 85:
            insights['highlights'].append(
                f"✅ **Good Data Quality ({quality_score:.0f}/100):** "
                f"The dataset is structurally sound with minor issues that have been addressed "
                f"through automated preprocessing. Suitable for model development with standard validation."
            )
        elif quality_score >= 70:
            insights['summary'].append(
                f"📋 **Adequate Data Quality ({quality_score:.0f}/100):** "
                f"The dataset requires attention to specific issues before production use. "
                f"Review the warnings below and perform manual validation on flagged columns."
            )
        else:
            insights['warnings'].append(
                f"⚠️ **Poor Data Quality ({quality_score:.0f}/100):** "
                f"Significant data quality issues detected. Automated cleaning has been applied, "
                f"but manual review is strongly recommended. Results below may be unreliable "
                f"without additional data cleaning."
            )
        
        # Missing data analysis
        if missing_pct > 10:
            insights['warnings'].append(
                f"🔴 **Critical Missing Data ({missing_pct:.1f}%):** "
                f"Over 10% of data is missing. Before modeling: (a) determine if missingness is "
                f"MCAR (Missing Completely At Random), MAR (Missing At Random), or MNAR "
                f"(Missing Not At Random). If MNAR, missingness itself is informative and should "
                f"be encoded as a feature. Standard imputation will introduce bias if data is MNAR."
            )
        elif missing_pct > 3:
            insights['summary'].append(
                f"🟡 **Moderate Missing Data ({missing_pct:.1f}%):** "
                f"KNN imputation (k=5) has been applied to numeric features; mode imputation for "
                f"categorical. Validate imputed values against original distributions. For production: "
                f"consider Multiple Imputation by Chained Equations (MICE) which preserves uncertainty."
            )
        elif missing_pct > 0:
            insights['summary'].append(
                f"🟢 **Minor Missing Data ({missing_pct:.1f}%):** "
                f"Negligible missing values were imputed. Impact on analysis is minimal."
            )
        else:
            insights['highlights'].append(
                f"✅ **Complete Dataset:** No missing values detected. All {n_rows:,} rows are fully populated."
            )
        
        # Duplicate analysis
        if dup_pct > 5:
            insights['warnings'].append(
                f"🔴 **High Duplicate Rate ({dup_pct:.1f}%):** "
                f"Duplicates can artificially inflate statistical significance and cause overfitting. "
                f"Verify that duplicates are not legitimate repeated measurements before removal."
            )
        elif dup_pct > 1:
            insights['summary'].append(
                f"🟡 **Duplicate Records ({dup_pct:.1f}%):** Removed during preprocessing. "
                f"Investigate source system for duplicate generation."
            )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 2: FEATURE DISTRIBUTION & NORMALITY
        # ═══════════════════════════════════════════════════════════
        
        normality = stats_results.get('normality', pd.DataFrame())
        if not normality.empty:
            total_cols = len(normality)
            normal_cols = normality[normality['Normal'] == True]
            non_normal = normality[normality['Normal'] == False]
            
            if len(non_normal) == total_cols and total_cols > 3:
                insights['summary'].append(
                    f"📊 **Universal Non-Normality:** All {total_cols} numeric features deviate from "
                    f"normal distribution. This is common in real-world data (income, prices, counts). "
                    f"Implications: (a) Use non-parametric tests (Spearman correlation, Mann-Whitney U, "
                    f"Kruskal-Wallis) rather than their parametric equivalents. (b) For linear models, "
                    f"apply Yeo-Johnson or Box-Cox transformations to reduce skew. "
                    f"(c) Tree-based models (Random Forest, XGBoost) are unaffected by non-normality."
                )
            elif len(non_normal) > total_cols * 0.5:
                insights['summary'].append(
                    f"📊 **Majority Non-Normal:** {len(non_normal)}/{total_cols} features are non-normal. "
                    f"Consider transformation before linear modeling. Non-parametric methods recommended."
                )
            
            # Skewness details
            if 'Skewness' in normality.columns:
                right_skewed = normality[normality['Skewness'] > 1]
                left_skewed = normality[normality['Skewness'] < -1]
                
                if len(right_skewed) > 0:
                    skewed_names = ', '.join(right_skewed.nlargest(3, 'Skewness')['Column'].tolist())
                    insights['summary'].append(
                        f"📈 **Right-Skewed Features:** {skewed_names} have long right tails "
                        f"(skew > 1). Common in count/currency data. Log transformation (log1p) or "
                        f"Yeo-Johnson recommended before linear regression."
                    )
                
                if len(left_skewed) > 0:
                    skewed_names = ', '.join(left_skewed.nsmallest(3, 'Skewness')['Column'].tolist())
                    insights['summary'].append(
                        f"📉 **Left-Skewed Features:** {skewed_names} have long left tails "
                        f"(skew < -1). Consider reflection transformation (max - x) then log."
                    )
            
            # Kurtosis
            if 'Kurtosis' in normality.columns:
                high_kurt = normality[normality['Kurtosis'] > 3]
                if len(high_kurt) > 0:
                    kurt_names = ', '.join(high_kurt.nlargest(3, 'Kurtosis')['Column'].tolist())
                    insights['summary'].append(
                        f"🔔 **Heavy-Tailed Features:** {kurt_names} show excess kurtosis (> 3), "
                        f"indicating frequent extreme values. Robust scalers (RobustScaler) are "
                        f"preferred over StandardScaler for these variables."
                    )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 3: CORRELATION & MULTICOLLINEARITY
        # ═══════════════════════════════════════════════════════════
        
        pearson = stats_results.get('pearson_correlation', pd.DataFrame())
        if not pearson.empty:
            corr_col = 'Correlation (r)' if 'Correlation (r)' in pearson.columns else 'Correlation'
            strong_corrs = pearson[pearson[corr_col].abs() > 0.7].sort_values(corr_col, ascending=False)
            
            if len(strong_corrs) > 0:
                top3 = strong_corrs.head(3)
                for _, row in top3.iterrows():
                    direction = "positive" if row[corr_col] > 0 else "negative"
                    ci = f" [95% CI: {row.get('CI 95% Low', 'N/A')}, {row.get('CI 95% High', 'N/A')}]" if 'CI 95% Low' in row else ""
                    insights['highlights'].append(
                        f"🔗 **Strong {direction} correlation:** {row['Variable 1']} ↔ {row['Variable 2']} "
                        f"(r = {row[corr_col]:.3f}{ci}). "
                        f"This relationship {'is expected and structurally valid' if abs(row[corr_col]) > 0.9 else 'warrants further investigation for potential confounding'}."
                    )
                
                if len(strong_corrs) > 5:
                    insights['warnings'].append(
                        f"🔴 **Severe Multicollinearity:** {len(strong_corrs)} feature pairs with |r| > 0.7. "
                        f"This level of intercorrelation will: (a) Inflate standard errors in OLS regression "
                        f"by 2-5×, making coefficients unreliable. (b) Cause sign reversals in coefficients. "
                        f"(c) Make feature importance from linear models misleading. "
                        f"**Solution Path:** (1) Apply Variance Inflation Factor (VIF) analysis — remove features "
                        f"with VIF > 10 iteratively. (2) Use L1 regularization (Lasso) which performs automatic "
                        f"feature selection. (3) Apply PCA to create orthogonal components. "
                        f"(4) Use tree-based models which are robust to collinearity."
                    )
                elif len(strong_corrs) > 2:
                    insights['summary'].append(
                        f"🟡 **Moderate Collinearity:** {len(strong_corrs)} strongly correlated pairs detected. "
                        f"Check VIF scores below. If VIF < 5 for all features, no action needed."
                    )
        
        # VIF Analysis
        if not vif_df.empty:
            high_vif = vif_df[vif_df['VIF'] > 10]
            moderate_vif = vif_df[(vif_df['VIF'] >= 5) & (vif_df['VIF'] <= 10)]
            
            if len(high_vif) > 0:
                high_features = high_vif['Feature'].tolist()
                insights['warnings'].append(
                    f"🔴 **Critical VIF Detected:** {len(high_vif)} features exceed VIF threshold of 10: "
                    f"**{', '.join(high_features[:5])}**"
                    f"{' and ' + str(len(high_features) - 5) + ' more' if len(high_features) > 5 else ''}. "
                    f"These features are nearly perfectly predictable from other features. "
                    f"**Remediation Priority:** (1) Drop the feature with highest VIF first, recalculate, "
                    f"repeat until all VIF < 5. (2) Combine correlated features via PCA or averaging. "
                    f"(3) Use Ridge regression (L2 penalty) which handles multicollinearity gracefully."
                )
            
            if len(moderate_vif) > 0:
                mod_features = moderate_vif['Feature'].tolist()
                insights['summary'].append(
                    f"🟡 **Moderate VIF (5-10):** {', '.join(mod_features[:3])}"
                    f"{' and others' if len(mod_features) > 3 else ''} show moderate collinearity. "
                    f"Acceptable for prediction but may affect coefficient interpretation."
                )
            
            # Condition number
            if 'Condition Number (κ)' in vif_df.columns:
                kappa = vif_df['Condition Number (κ)'].iloc[0]
                if kappa > 100:
                    insights['warnings'].append(
                        f"🔴 **Severe Numerical Instability (κ = {kappa:.1f}):** "
                        f"The design matrix is near-singular. OLS regression results are numerically "
                        f"unstable and may differ across machines. Standardization is mandatory. "
                        f"Consider Ridge regression with cross-validated alpha."
                    )
                elif kappa > 30:
                    insights['warnings'].append(
                        f"🟠 **High Condition Number (κ = {kappa:.1f}):** "
                        f"Indicates strong multicollinearity. Standardize features and recheck."
                    )
                elif kappa > 10:
                    insights['summary'].append(
                        f"🟡 **Moderate Condition Number (κ = {kappa:.1f}):** "
                        f"Some collinearity present. Standardization recommended."
                    )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 4: STATISTICAL TEST RESULTS
        # ═══════════════════════════════════════════════════════════
        
        # FDR Analysis
        if not pearson.empty and 'Sig. after FDR' in pearson.columns:
            sig_before = len(pearson[pearson['Significant'] == True])
            sig_after = len(pearson[pearson['Sig. after FDR'] == True])
            false_positives = sig_before - sig_after
            
            if false_positives > 0:
                insights['highlights'].append(
                    f"🎯 **FDR Correction Applied:** Benjamini-Hochberg procedure identified "
                    f"**{false_positives} potential false positives** out of {sig_before} nominally "
                    f"significant correlations. Without FDR correction, approximately "
                    f"{false_positives} spurious correlations would have been reported as significant. "
                    f"All reported p-values are FDR-adjusted for multiple testing."
                )
        
        # ANOVA
        anova = stats_results.get('anova', pd.DataFrame())
        if not anova.empty:
            sig_anova = anova[anova['Significant'] == True]
            
            if not sig_anova.empty and 'Effect Size' in sig_anova.columns:
                large_effects = sig_anova[sig_anova['Effect Size'].isin(['Large', 'Very Large'])]
                medium_effects = sig_anova[sig_anova['Effect Size'] == 'Medium']
                
                if not large_effects.empty:
                    for _, row in large_effects.head(3).iterrows():
                        eta2 = row.get('Eta-Squared (η²)', 'N/A')
                        insights['highlights'].append(
                            f"📏 **Large Group Effect:** **{row['Numeric Variable']}** differs substantially "
                            f"across **{row['Grouping Variable']}** groups "
                            f"(η² = {eta2}, {row['Effect Size']} effect). "
                            f"This categorical variable explains {float(eta2)*100:.1f}% of variance — "
                            f"include it as a key predictor or stratification variable in models."
                        )
                
                if not medium_effects.empty:
                    insights['summary'].append(
                        f"📏 **Moderate Group Effects:** {len(medium_effects)} relationships show medium "
                        f"effect sizes. Consider these grouping variables for feature engineering "
                        f"(target encoding, frequency encoding)."
                    )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 5: OLS REGRESSION DIAGNOSTICS
        # ═══════════════════════════════════════════════════════════
        
        ols_stats = stats_results.get('ols_model_stats', {})
        if ols_stats:
            r2 = ols_stats.get('R²', 0)
            adj_r2 = ols_stats.get('Adj. R²', 0)
            f_pval = ols_stats.get('F P-Value', 1)
            n_obs = ols_stats.get('N', 0)
            kappa_ols = ols_stats.get('Condition Number', 0)
            target = ols_stats.get('Target', 'target')
            
            if r2 > 0.8:
                insights['highlights'].append(
                    f"🎯 **Strong Model Fit:** OLS regression explains **{r2*100:.1f}% of variance** "
                    f"in **{target}** (R² = {r2:.3f}, Adj. R² = {adj_r2:.3f}). "
                    f"The model is highly predictive. However, with {n_obs:,} observations, "
                    f"high R² may indicate overfitting if many predictors are used. "
                    f"Validate with cross-validation and check for data leakage."
                )
            elif r2 > 0.5:
                insights['summary'].append(
                    f"📊 **Moderate Model Fit:** R² = {r2:.3f} (Adj. R² = {adj_r2:.3f}). "
                    f"The model captures meaningful signal but leaves substantial variance unexplained. "
                    f"**Improvement paths:** (a) Engineer interaction terms between top correlated features. "
                    f"(b) Include polynomial terms for non-linear relationships. "
                    f"(c) Try tree-based ensembles (XGBoost, LightGBM) which capture non-linear patterns."
                )
            else:
                insights['summary'].append(
                    f"📊 **Limited Linear Relationship:** R² = {r2:.3f} suggests linear assumptions "
                    f"may be insufficient for **{target}**. Consider: (a) Non-linear models "
                    f"(Random Forest, Gradient Boosting). (b) Feature engineering — create ratios, "
                    f"log transforms, and interaction terms. (c) Additional external data sources "
                    f"to capture unexplained variance."
                )
            
            if f_pval < 0.001:
                insights['highlights'].append(
                    f"✅ **Model is statistically significant** (F-test p < 0.001). "
                    f"At least one predictor has a genuine relationship with {target}."
                )
            elif f_pval < 0.05:
                insights['summary'].append(
                    f"✅ Model F-test significant at α = 0.05 (p = {f_pval:.4f})."
                )
            else:
                insights['warnings'].append(
                    f"⚠️ **Model Not Significant:** F-test p = {f_pval:.4f} > 0.05. "
                    f"The model does not outperform a constant mean prediction. "
                    f"Reconsider feature selection or model approach."
                )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 6: OUTLIER ANALYSIS
        # ═══════════════════════════════════════════════════════════
        
        outlier_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_pct = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).mean() * 100
            if outlier_pct > 5:
                outlier_cols.append((col, outlier_pct))
        
        if len(outlier_cols) > 3:
            top_outliers = sorted(outlier_cols, key=lambda x: x[1], reverse=True)[:3]
            top_names = [f"{name} ({pct:.1f}%)" for name, pct in top_outliers]
            insights['summary'].append(
                f"🔍 **High Outlier Features:** {', '.join(top_names)} contain substantial outliers. "
                f"Winsorization (capping at 1.5×IQR) has been applied. For production: "
                f"(a) Investigate if outliers represent genuine extreme values or errors. "
                f"(b) If genuine, consider robust models (Huber regression, quantile regression). "
                f"(c) If errors, implement data validation at collection point."
            )
        elif len(outlier_cols) > 0:
            insights['summary'].append(
                f"🔍 **Outliers Detected:** {len(outlier_cols)} features have > 5% outliers. "
                f"Capped via Winsorization. Verify these are not measurement errors."
            )
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 7: FINAL RECOMMENDATION
        # ═══════════════════════════════════════════════════════════
        
        recommendations = []
        
        # Prioritize by severity
        has_severe_vif = not vif_df.empty and len(vif_df[vif_df['VIF'] > 10]) > 2
        has_severe_kappa = not vif_df.empty and vif_df['Condition Number (κ)'].iloc[0] > 100 if 'Condition Number (κ)' in vif_df.columns else False
        has_missing = missing_pct > 3
        has_low_r2 = ols_stats.get('R²', 1) < 0.4 if ols_stats else False
        
        if has_severe_vif or has_severe_kappa:
            recommendations.append(
                "**🚨 PRIORITY 1 — Address Multicollinearity:** This is the most critical issue. "
                "Remove features iteratively by VIF, retrain, and validate. "
                "Target: all VIF < 5, condition number < 30."
            )
        
        if has_missing:
            recommendations.append(
                "**⚠️ PRIORITY 2 — Missing Data Strategy:** Implement MICE (Multiple Imputation "
                "by Chained Equations) for production pipelines. Simple imputation is acceptable "
                "for exploration but introduces bias in production models."
            )
        
        if has_low_r2:
            recommendations.append(
                "**📊 PRIORITY 3 — Improve Model Fit:** Current linear model explains limited "
                "variance. Explore non-linear algorithms (XGBoost, LightGBM, Random Forest), "
                "create interaction features, and consider external data enrichment."
            )
        
        if not recommendations:
            recommendations.append(
                "**✅ Production Ready:** The dataset meets quality thresholds for model development. "
                "Recommended workflow: (1) Train-test split with stratification on key groups. "
                "(2) Cross-validation with 5+ folds. (3) Compare linear vs. tree-based models. "
                "(4) Monitor feature distributions in production for drift detection."
            )
        
        insights['recommendation'] = ' '.join(recommendations)
        
        # ═══════════════════════════════════════════════════════════
        # SECTION 8: READINESS SCORE
        # ═══════════════════════════════════════════════════════════
        
        readiness = self._calculate_readiness(quality_score, insights, vif_df, missing_pct, ols_stats)
        insights['readiness_score'] = readiness
        
        self.insights = insights
        return insights
    
    def _calculate_readiness(self, quality_score, insights, vif_df, missing_pct, ols_stats):
        """Weighted readiness score for production deployment."""
        score = 0
        
        # Data quality (40%)
        score += (quality_score / 100) * 40
        
        # Completeness (15%)
        score += max(0, (1 - missing_pct / 100)) * 15
        
        # Collinearity penalty (20%)
        if not vif_df.empty:
            high_vif = len(vif_df[vif_df['VIF'] > 10])
            score -= min(high_vif * 3, 20)
        
        # Model fit bonus (15%)
        if ols_stats:
            r2 = ols_stats.get('R²', 0)
            score += min(r2 * 15, 15)
        
        # Insight quality (10%)
        score += len(insights['highlights']) * 1.5
        score -= len(insights['warnings']) * 5
        
        return max(0, min(score, 100))
