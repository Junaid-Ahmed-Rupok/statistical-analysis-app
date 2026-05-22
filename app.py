import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom modules
from modules.preprocessor import DataPreprocessor
from modules.statistics import StatisticalAnalyzer
from modules.visualizer import Visualizer
from modules.insights import InsightsGenerator
from modules.pdf_generator import PDFReportGenerator

# Page configuration
st.set_page_config(
    page_title="StatsPro — AI-Powered Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_file = Path(__file__).parent / "styles" / "main.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
_defaults = {
    'uploaded_df': None, 'cleaned_df': None, 'overview': None,
    'stats_results': None, 'insights': None, 'figures': [],
    'pdf_bytes': None, 'cleaning_log': [], 'outlier_counts': {},
    'loaded_file_name': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------
preprocessor  = DataPreprocessor()
analyzer      = StatisticalAnalyzer()
visualizer    = Visualizer()
insights_gen  = InsightsGenerator()
pdf_gen       = PDFReportGenerator()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _green(val):
    return 'background-color: #D5F5E3; color: #1E8449' if val is True else ''

def _color_missing(val):
    if val > 20:  return 'background-color: #FADBD8; color: #C0392B; font-weight: bold'
    if val > 5:   return 'background-color: #FFF3CD; color: #F0A500; font-weight: bold'
    return 'background-color: #D5F5E3; color: #1E8449; font-weight: bold'

def _color_vif(val):
    try:
        v = float(val)
    except Exception:
        return ''
    if v < 5:   return 'background-color: #D5F5E3; color: #1E8449'
    if v < 10:  return 'background-color: #FFF3CD; color: #F0A500'
    return 'background-color: #FADBD8; color: #C0392B'

def _color_effect(val):
    mapping = {
        'Negligible': 'color: #6C757D',
        'Small':      'color: #F0A500',
        'Medium':     'color: #1B4F72; font-weight:600',
        'Large':      'color: #1E8449; font-weight:700',
        'Very Large': 'color: #C0392B; font-weight:700',
    }
    return mapping.get(str(val), '')

def _color_fdr(val):
    return 'background-color: #D5F5E3; color: #1E8449' if val is True else \
           'background-color: #FADBD8; color: #C0392B' if val is False else ''

def _sig_col(df):
    return 'Sig. after FDR' if 'Sig. after FDR' in df.columns else 'Significant'

def _corr_col(df):
    return 'Correlation (r)' if 'Correlation (r)' in df.columns else 'Correlation'

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <h1 style="color:#F0A500;font-size:32px;font-family:'Poppins',sans-serif;margin:0;">StatsPro</h1>
        <p style="color:white;font-size:14px;opacity:0.9;margin-top:5px;">Professional Statistical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="color:white;padding:10px 0;">
        <h3 style="color:#F0A500;margin-bottom:15px;">How It Works</h3>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;margin:8px 0;"><strong>1. Upload your CSV</strong></div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;margin:8px 0;"><strong>2. Auto preprocessing</strong></div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;margin:8px 0;"><strong>3. Full statistical analysis</strong></div>
        <div style="background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;margin:8px 0;"><strong>4. Download PDF + Excel</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("Analysis Settings", expanded=False):
        correlation_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
        handle_outliers    = st.checkbox("Handle Outliers", value=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:10px 0;">
        <div style="color:white;font-size:12px;margin:5px 0;">Private & Secure</div>
        <div style="color:white;font-size:12px;margin:5px 0;">Instant Results</div>
        <div style="color:white;font-size:12px;margin:5px 0;">Free to Use</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Hero Banner
# ------------------------------------------------------------------
st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1B4F72 50%,#0D2F56 100%);
    padding:60px 40px;border-radius:16px;margin-bottom:30px;">
    <h1 style="color:white;font-family:'Poppins',sans-serif;font-size:42px;font-weight:700;margin:0 0 15px 0;">
        Turn Raw Data Into Powerful Insights
    </h1>
    <p style="color:#F0A500;font-size:18px;font-weight:300;margin:0 0 30px 0;max-width:800px;">
        Upload any CSV. Get instant statistical analysis, beautiful charts,
        and a professional PDF report — in seconds.
    </p>
    <div style="display:flex;gap:15px;flex-wrap:wrap;">
        <span style="background:rgba(240,165,0,0.2);color:#F0A500;padding:8px 20px;border-radius:20px;border:1px solid #F0A500;font-weight:600;font-size:14px;">10 Statistical Tests</span>
        <span style="background:rgba(240,165,0,0.2);color:#F0A500;padding:8px 20px;border-radius:20px;border:1px solid #F0A500;font-weight:600;font-size:14px;">15+ Visualizations</span>
        <span style="background:rgba(240,165,0,0.2);color:#F0A500;padding:8px 20px;border-radius:20px;border:1px solid #F0A500;font-weight:600;font-size:14px;">Effect Sizes + FDR</span>
        <span style="background:rgba(240,165,0,0.2);color:#F0A500;padding:8px 20px;border-radius:20px;border:1px solid #F0A500;font-weight:600;font-size:14px;">OLS Regression</span>
        <span style="background:rgba(240,165,0,0.2);color:#F0A500;padding:8px 20px;border-radius:20px;border:1px solid #F0A500;font-weight:600;font-size:14px;">PDF + Excel Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tabs = st.tabs([
    "Upload & Overview",
    "Preprocessing",
    "Statistical Tests",
    "Visualizations",
    "AI Insights",
    "PDF Report"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Overview
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Drag & drop your CSV file here or click to browse",
        type=["csv"],
        help="Supports CSV files up to 200MB"
    )

    st.markdown("**No data? Try our sample dataset:**")
    if st.button("Load Sample Dataset (Restaurant Tips)", type="secondary"):
        import seaborn as sns
        sample_df = sns.load_dataset('tips')
        sample_df['tip_percentage'] = (sample_df['tip'] / sample_df['total_bill']) * 100
        sample_df['is_weekend'] = sample_df['day'].isin(['Sat', 'Sun']).astype(int)
        sample_df['party_size_category'] = pd.cut(
            sample_df['size'],
            bins=[0, 2, 4, 10],
            labels=['Small', 'Medium', 'Large']
        )
        st.session_state.uploaded_df = sample_df
        st.session_state.loaded_file_name = "sample_tips.csv"
        for k in ['cleaned_df', 'overview', 'stats_results', 'insights', 'figures', 'pdf_bytes',
                  'cleaning_log', 'outlier_counts']:
            if k in st.session_state:
                st.session_state[k] = _defaults.get(k, None)
        overview = preprocessor.get_overview(sample_df)
        st.session_state.overview = overview
        st.rerun()

    st.markdown("---")

    if uploaded_file is not None or st.session_state.uploaded_df is not None:
        if uploaded_file is not None and uploaded_file.name != st.session_state.loaded_file_name:
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.session_state.loaded_file_name = uploaded_file.name

        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                overview = preprocessor.get_overview(df)
                st.session_state.overview = overview
            else:
                df = st.session_state.uploaded_df
                overview = st.session_state.overview

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows",      f"{overview['rows']:,}")
            with col2:
                st.metric("Total Columns",   f"{overview['columns']}")
            with col3:
                st.metric("Complete Rows %", f"{100 - overview['missing_percentage']:.1f}%")
            with col4:
                st.metric("Missing Values",  f"{overview['total_missing']:,}")

            st.markdown("---")
            st.markdown("### Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

            st.markdown("### Missing Values Analysis")
            if not overview['missing_table'].empty:
                st.dataframe(
                    overview['missing_table'].style.map(
                        _color_missing, subset=['Missing Percentage']
                    ),
                    use_container_width=True
                )
            else:
                st.success("No missing values detected.")

            st.markdown("---")
            st.markdown("### Data Quality Score")
            quality = overview['quality_score']
            color   = '#1E8449' if quality >= 80 else '#F0A500' if quality >= 50 else '#C0392B'
            label   = ('Excellent — ready for analysis' if quality >= 80 else
                       'Good — minor cleaning recommended' if quality >= 50 else
                       'Needs attention — significant issues detected')
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:30px;text-align:center;border:2px solid {color};">
                <div style="font-size:48px;font-weight:800;color:{color};font-family:'Poppins',sans-serif;">{quality:.0f}/100</div>
                <div style="font-size:16px;color:#6C757D;margin-top:10px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Upload a CSV file or click 'Load Sample Dataset' to get started!")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Data Preprocessing")

    if st.session_state.uploaded_df is not None:
        df   = st.session_state.uploaded_df
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Before Preprocessing")
            st.metric("Missing Values",  f"{df.isnull().sum().sum():,}")
            st.metric("Duplicate Rows",  f"{df.duplicated().sum():,}")
            st.metric("Total Rows",      f"{len(df):,}")

        if st.button("Run Preprocessing", key="preprocess_btn"):
            with st.spinner("Preprocessing data..."):
                cleaned_df, cleaning_log = preprocessor.clean(df)
                if handle_outliers:
                    cleaned_df, outlier_counts = preprocessor.handle_outliers(cleaned_df)
                else:
                    outlier_counts = {}
                st.session_state.cleaned_df    = cleaned_df
                st.session_state.cleaning_log  = cleaning_log
                st.session_state.outlier_counts = outlier_counts
                st.session_state.stats_results = None
                st.session_state.insights      = None
                st.session_state.pdf_bytes     = None

        with col2:
            st.markdown("#### After Preprocessing")
            if st.session_state.cleaned_df is not None:
                cleaned_df = st.session_state.cleaned_df
                st.metric("Missing Values", "0")
                st.metric("Duplicate Rows", "0")
                st.metric("Total Rows",     f"{len(cleaned_df):,}")

        if st.session_state.cleaning_log:
            st.markdown("---")
            st.markdown("### Preprocessing Activity Log")
            for log in st.session_state.cleaning_log:
                st.markdown(f"""
                <div style="background:white;padding:15px;border-radius:8px;
                    border-left:4px solid #1E8449;margin:10px 0;">
                    <strong style="color:#1E8449;">✓</strong> {log['detail']}
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.outlier_counts:
            st.markdown("---")
            st.markdown("### Outlier Detection Results")
            outlier_data = (
                pd.DataFrame({
                    'Column':   list(st.session_state.outlier_counts.keys()),
                    'Outliers': list(st.session_state.outlier_counts.values())
                }).sort_values('Outliers', ascending=False)
            )
            if not outlier_data.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(outlier_data['Column'], outlier_data['Outliers'], color='#34D399')
                ax.set_xlabel('Number of Outliers')
                st.pyplot(fig)
                plt.close(fig)

        if st.session_state.cleaned_df is not None:
            st.markdown("---")
            st.download_button(
                label="Download Cleaned CSV",
                data=st.session_state.cleaned_df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload data in the 'Upload & Overview' tab first.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Statistical Tests
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Comprehensive Statistical Testing")
    st.markdown("*Scale-aware tests · Effect sizes · FDR correction · OLS regression*")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        if st.button("Run Statistical Tests", key="run_stats_btn"):
            with st.spinner("Running statistical tests..."):
                stats_results = analyzer.run_all(df)
                st.session_state.stats_results = stats_results
                st.session_state.insights  = None
                st.session_state.pdf_bytes = None

        if st.session_state.stats_results is not None:
            sr = st.session_state.stats_results

            with st.expander("Normality Tests", expanded=True):
                norm = sr.get('normality', pd.DataFrame())
                if not norm.empty:
                    styled = norm.style.map(_green, subset=['Normal'])
                    if 'Sig. after FDR' in norm.columns:
                        styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                    st.dataframe(styled, use_container_width=True)
                    st.caption("Test chosen automatically by sample size: Shapiro-Wilk (n≤50) · D'Agostino-Pearson (n≤5000) · Anderson-Darling (n>5000)")

            with st.expander("Correlation Analysis", expanded=True):
                corr_col = _corr_col(sr.get('pearson_correlation', pd.DataFrame()))

                col1, col2 = st.columns(2)
                for label, key, col in [
                    ("Pearson Correlation",  'pearson_correlation',  col1),
                    ("Spearman Correlation", 'spearman_correlation', col2),
                ]:
                    with col:
                        st.markdown(f"**{label}**")
                        cdf = sr.get(key, pd.DataFrame())
                        if not cdf.empty and corr_col in cdf.columns:
                            top = cdf.nlargest(10, corr_col)
                            styled = top.style
                            if 'Effect Size' in top.columns:
                                styled = styled.map(_color_effect, subset=['Effect Size'])
                            if 'Sig. after FDR' in top.columns:
                                styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                            st.dataframe(styled, use_container_width=True)

            chi = sr.get('chi_square', pd.DataFrame())
            if not chi.empty:
                with st.expander("Chi-Square Tests + Cramér's V", expanded=False):
                    sig_c = _sig_col(chi)
                    styled = chi.style.map(_green, subset=[sig_c])
                    if 'Effect Size' in chi.columns:
                        styled = styled.map(_color_effect, subset=['Effect Size'])
                    st.dataframe(styled, use_container_width=True)

            anova = sr.get('anova', pd.DataFrame())
            if not anova.empty:
                with st.expander("ANOVA Tests + Eta-Squared (η²)", expanded=False):
                    sig_c  = _sig_col(anova)
                    styled = anova.style.map(_green, subset=[sig_c])
                    if 'Effect Size' in anova.columns:
                        styled = styled.map(_color_effect, subset=['Effect Size'])
                    st.dataframe(styled, use_container_width=True)

            mw = sr.get('mann_whitney', pd.DataFrame())
            if not mw.empty:
                with st.expander("Mann-Whitney U + Rank-Biserial r", expanded=False):
                    sig_c  = _sig_col(mw)
                    styled = mw.style.map(_green, subset=[sig_c])
                    if 'Effect Size' in mw.columns:
                        styled = styled.map(_color_effect, subset=['Effect Size'])
                    st.dataframe(styled, use_container_width=True)

            lev = sr.get('levene', pd.DataFrame())
            if not lev.empty:
                with st.expander("Levene's Test (Variance Equality)", expanded=False):
                    st.dataframe(
                        lev.style.map(_green, subset=['Equal Variance']),
                        use_container_width=True
                    )

            vif = sr.get('vif', pd.DataFrame())
            if not vif.empty:
                with st.expander("Multicollinearity — VIF + Condition Number (κ)", expanded=False):
                    styled = vif.style.map(_color_vif, subset=['VIF'])
                    if 'κ Risk' in vif.columns:
                        styled = styled.map(
                            lambda v: ('color:#C0392B;font-weight:700' if v == 'High' else
                                       'color:#F0A500' if v == 'Moderate' else 'color:#1E8449'),
                            subset=['κ Risk']
                        )
                    st.dataframe(styled, use_container_width=True)
                    if 'Condition Number (κ)' in vif.columns:
                        kappa = vif['Condition Number (κ)'].iloc[0]
                        k_risk = vif['κ Risk'].iloc[0]
                        k_color = '#C0392B' if k_risk == 'High' else '#F0A500' if k_risk == 'Moderate' else '#1E8449'
                        st.markdown(f"""
                        <div style="background:white;padding:16px;border-radius:8px;
                            border-left:4px solid {k_color};margin-top:12px;">
                            <strong>Condition Number κ = {kappa:.2f}</strong> — {k_risk} risk
                        </div>
                        """, unsafe_allow_html=True)

            reg_df    = sr.get('regression', pd.DataFrame())
            ols_stats = sr.get('ols_model_stats', {})
            if not reg_df.empty and ols_stats:
                with st.expander(f"OLS Regression — predicting '{ols_stats.get('Target', '')}'", expanded=True):
                    m = ols_stats
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("R²",           f"{m.get('R²', 0):.4f}")
                    mc2.metric("Adj. R²",      f"{m.get('Adj. R²', 0):.4f}")
                    mc3.metric("F-Statistic",  f"{m.get('F-Statistic', 0):.3f}")
                    mc4.metric("AIC",          f"{m.get('AIC', 0):.1f}")
                    mc5.metric("BIC",          f"{m.get('BIC', 0):.1f}")

                    sig_c  = _sig_col(reg_df)
                    styled = reg_df.style.map(_green, subset=[sig_c])
                    if 'Sig. after FDR' in reg_df.columns:
                        styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                    st.dataframe(styled, use_container_width=True)
    else:
        st.info("Please preprocess your data in the 'Preprocessing' tab first.")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Visualizations
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Interactive Visualizations")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        with st.expander("Distribution Analysis", expanded=True):
            st.markdown("**Histograms with KDE Overlay**")
            fig = visualizer.histograms(df)
            if fig:
                st.pyplot(fig); plt.close(fig)

            st.markdown("**Box Plots**")
            fig = visualizer.boxplots(df)
            if fig:
                st.pyplot(fig); plt.close(fig)

        with st.expander("Correlation Visualizations", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pearson Heatmap**")
                fig = visualizer.correlation_heatmap(df, method='pearson')
                if fig:
                    st.pyplot(fig); plt.close(fig)
            with col2:
                st.markdown("**Spearman Heatmap**")
                fig = visualizer.correlation_heatmap(df, method='spearman')
                if fig:
                    st.pyplot(fig); plt.close(fig)

            st.markdown("**Pairplot**")
            fig = visualizer.pairplot(df)
            if fig:
                st.pyplot(fig); plt.close(fig)

        with st.expander("Normality Assessment", expanded=False):
            st.markdown("**Q-Q Plots**")
            fig = visualizer.qq_plots(df)
            if fig:
                st.pyplot(fig); plt.close(fig)

        sr = st.session_state.stats_results
        if sr and not sr.get('vif', pd.DataFrame()).empty:
            with st.expander("Multicollinearity Analysis", expanded=False):
                fig = visualizer.vif_chart(sr['vif'])
                if fig:
                    st.pyplot(fig); plt.close(fig)
    else:
        st.info("Please preprocess your data first.")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — AI Insights
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### What Your Data Is Telling You")

    if st.session_state.cleaned_df is not None and st.session_state.stats_results is not None:
        df            = st.session_state.cleaned_df
        stats_results = st.session_state.stats_results
        overview      = st.session_state.overview

        if st.button("Generate AI Insights", key="insights_btn"):
            with st.spinner("Generating insights..."):
                vif_df   = stats_results.get('vif', pd.DataFrame())
                insights = insights_gen.generate(df, stats_results, vif_df, overview)
                st.session_state.insights = insights

        if st.session_state.insights is not None:
            insights = st.session_state.insights

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;border:2px solid #1E8449;height:100%;">
                    <h4 style="color:#1E8449;">Data Quality</h4>
                    <p><strong>Quality Score:</strong> {overview['quality_score']:.0f}/100</p>
                    <p><strong>Completeness:</strong> {100 - overview['missing_percentage']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                highlights = '<br>'.join([f"• {h}" for h in insights.get('highlights', [])[:3]])
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;border:2px solid #F0A500;height:100%;">
                    <h4 style="color:#F0A500;">Key Findings</h4>
                    <p>{highlights or 'No significant findings'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                warnings = '<br>'.join([f"• {w}" for w in insights.get('warnings', [])[:3]])
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;border:2px solid #C0392B;height:100%;">
                    <h4 style="color:#C0392B;">Warnings</h4>
                    <p>{warnings or 'No warnings detected'}</p>
                </div>
                """, unsafe_allow_html=True)

            if insights.get('summary'):
                st.markdown("---")
                st.markdown("#### Summary Notes")
                for s in insights['summary']:
                    st.markdown(f"- {s}")

            st.markdown("---")
            readiness = insights.get('readiness_score', 0)
            r_color   = '#1E8449' if readiness >= 70 else '#F0A500' if readiness >= 40 else '#C0392B'
            st.markdown(f"""
            <div style="text-align:center;padding:30px;">
                <h3>Data Readiness Score</h3>
                <div style="width:150px;height:150px;border-radius:50%;
                    background:conic-gradient({r_color} {readiness}%,#E0E6ED {readiness}%);
                    display:inline-flex;align-items:center;justify-content:center;margin:20px;">
                    <div style="width:120px;height:120px;border-radius:50%;background:white;
                        display:flex;align-items:center;justify-content:center;
                        font-size:36px;font-weight:800;color:{r_color};font-family:'Poppins',sans-serif;">
                        {readiness:.0f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if insights.get('recommendation'):
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0A2342,#1B4F72);color:white;
                    padding:20px;border-radius:12px;margin-top:20px;">
                    <h4 style="color:#F0A500;">Recommendation</h4>
                    <p>{insights['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please complete data preprocessing and statistical tests first.")

# ══════════════════════════════════════════════════════════════════
# TAB 6 — PDF Report
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Generate Professional PDF Report")

    if st.session_state.cleaned_df is not None and st.session_state.stats_results is not None:
        st.markdown("""
        <div style="background:white;padding:24px;border-radius:12px;
            border-left:4px solid #F0A500;margin-bottom:20px;">
            <h4 style="color:#0A2342;">Your Report Will Include:</h4>
            <p>Executive Summary with Key Metrics</p>
            <p>Complete Data Overview</p>
            <p>Statistical Tests — with effect sizes & FDR correction</p>
            <p>OLS Regression Summary</p>
            <p>All Visualizations</p>
            <p>AI-Powered Insights & Recommendations</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate My PDF Report", key="pdf_btn"):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            try:
                status_text.markdown("**Compiling statistics...**");  progress_bar.progress(20)
                time.sleep(0.3)
                status_text.markdown("**Rendering visualizations...**"); progress_bar.progress(45)

                figures = []
                for fn in [
                    lambda: visualizer.histograms(st.session_state.cleaned_df),
                    lambda: visualizer.correlation_heatmap(st.session_state.cleaned_df),
                    lambda: visualizer.boxplots(st.session_state.cleaned_df),
                ]:
                    try:
                        fig = fn()
                        if fig:
                            figures.append(fig)
                    except Exception:
                        pass

                time.sleep(0.3)
                status_text.markdown("**Writing insights...**");  progress_bar.progress(70)
                time.sleep(0.3)
                status_text.markdown("**Building PDF...**");      progress_bar.progress(85)

                pdf_buffer = pdf_gen.generate(
                    df=st.session_state.uploaded_df,
                    overview=st.session_state.overview,
                    stats_results=st.session_state.stats_results,
                    insights=st.session_state.insights or {},
                    figures=figures,
                    cleaned_df=st.session_state.cleaned_df
                )
                st.session_state.pdf_bytes = pdf_buffer.getvalue()
                progress_bar.progress(100)
                status_text.markdown("**Report generated successfully!**")

                for fig in figures:
                    plt.close(fig)

            except Exception as e:
                st.error(f"Error generating PDF: {e}")

        if st.session_state.pdf_bytes is not None:
            st.markdown("---")
            st.success("Your report is ready!")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.pdf_bytes,
                    file_name=f"statspro_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
            with col_dl2:
                if st.session_state.cleaned_df is not None:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        st.session_state.cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                        if st.session_state.stats_results:
                            for name, result_df in st.session_state.stats_results.items():
                                if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                                    sheet_name = name[:31]
                                    result_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    excel_buffer.seek(0)

                    st.download_button(
                        label="Download Excel Report",
                        data=excel_buffer,
                        file_name=f"statspro_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"
                    )
    else:
        st.info("Please complete all previous steps to generate the PDF report.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:20px;color:#6C757D;font-size:14px;">
    <p>Built with Streamlit | StatsPro v2.0 | 2025</p>
</div>
""", unsafe_allow_html=True)
