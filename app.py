import io
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Import custom modules
from modules.preprocessor import DataPreprocessor
from modules.statistics import StatisticalAnalyzer
from modules.visualizer import Visualizer
from modules.insights import InsightsGenerator
from modules.pdf_generator import PDFReportGenerator
from modules.ml_engine import MLEngine

# ── Page configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="StatsPro — AI-Powered Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ─────────────────────────────────────────────────────
def load_css():
    css_file = Path(__file__).parent / "styles" / "main.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# ── Session state initialisation ────────────────────────────────
_defaults = {
    'uploaded_df':    None,
    'cleaned_df':     None,
    'overview':       None,
    'stats_results':  None,
    'insights':       None,
    'figures':        [],
    'pdf_bytes':      None,
    'cleaning_log':   [],
    'outlier_counts': {},
    'loaded_file_name': None,
    'ml_results':     None,
    'ml_target':      None,
    'ml_best_model':  None,
    'preprocessor':   None,
    'analyzer':       None,
    'visualizer':     None,
    'insights_gen':   None,
    'pdf_gen':        None,
    'ml_engine':      None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.preprocessor is None:
    st.session_state.preprocessor  = DataPreprocessor()
if st.session_state.analyzer is None:
    st.session_state.analyzer      = StatisticalAnalyzer()
if st.session_state.visualizer is None:
    st.session_state.visualizer    = Visualizer()
if st.session_state.insights_gen is None:
    st.session_state.insights_gen  = InsightsGenerator()
if st.session_state.pdf_gen is None:
    st.session_state.pdf_gen       = PDFReportGenerator()
if st.session_state.ml_engine is None:
    st.session_state.ml_engine     = MLEngine()

preprocessor = st.session_state.preprocessor
analyzer     = st.session_state.analyzer
visualizer   = st.session_state.visualizer
insights_gen = st.session_state.insights_gen
pdf_gen      = st.session_state.pdf_gen
ml_engine    = st.session_state.ml_engine

# ── Styler helpers ───────────────────────────────────────────────
def _green(val):
    return 'background-color: #D5F5E3; color: #1E8449' if val is True else ''

def _color_missing(val):
    if val > 20: return 'background-color: #FADBD8; color: #C0392B; font-weight: bold'
    if val > 5:  return 'background-color: #FFF3CD; color: #F0A500; font-weight: bold'
    return 'background-color: #D5F5E3; color: #1E8449; font-weight: bold'

def _color_vif(val):
    try:
        v = float(val)
    except Exception:
        return ''
    if v < 5:  return 'background-color: #D5F5E3; color: #1E8449'
    if v < 10: return 'background-color: #FFF3CD; color: #F0A500'
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

def _safe_col_stats(series):
    """Return (min, max, mean) for a numeric series, guarding against all-NaN and empty series."""
    if series is None or len(series) == 0:
        return 0.0, 1.0, 0.0
    clean = series.dropna()
    if clean.empty:
        return 0.0, 1.0, 0.0
    try:
        return float(clean.min()), float(clean.max()), float(clean.mean())
    except Exception:
        return 0.0, 1.0, 0.0

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <h1 style="color:#34D399;font-size:28px;font-family:'JetBrains Mono',monospace;margin:0;">
            📊 StatsPro
        </h1>
        <p style="color:#aaaaaa;font-size:12px;margin-top:6px;font-family:'JetBrains Mono',monospace;">
            Professional Statistical Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="padding:8px 0;">
        <p style="color:#34D399;font-size:11px;font-family:'JetBrains Mono',monospace;
            text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">
            How It Works
        </p>
        <div style="background:rgba(255,255,255,0.04);padding:12px 14px;border-radius:6px;
            margin:6px 0;border-left:2px solid #2a3a2e;font-family:'JetBrains Mono',monospace;
            font-size:12px;color:#aaaaaa;">
            <strong style="color:#f0f0f0;">1.</strong> Upload your CSV
        </div>
        <div style="background:rgba(255,255,255,0.04);padding:12px 14px;border-radius:6px;
            margin:6px 0;border-left:2px solid #2a3a2e;font-family:'JetBrains Mono',monospace;
            font-size:12px;color:#aaaaaa;">
            <strong style="color:#f0f0f0;">2.</strong> Auto preprocessing
        </div>
        <div style="background:rgba(255,255,255,0.04);padding:12px 14px;border-radius:6px;
            margin:6px 0;border-left:2px solid #2a3a2e;font-family:'JetBrains Mono',monospace;
            font-size:12px;color:#aaaaaa;">
            <strong style="color:#f0f0f0;">3.</strong> Explore &amp; analyze
        </div>
        <div style="background:rgba(255,255,255,0.04);padding:12px 14px;border-radius:6px;
            margin:6px 0;border-left:2px solid #2a3a2e;font-family:'JetBrains Mono',monospace;
            font-size:12px;color:#aaaaaa;">
            <strong style="color:#f0f0f0;">4.</strong> Download PDF + Excel
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("⚙️ Analysis Settings", expanded=False):
        correlation_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
        handle_outliers    = st.checkbox("Handle Outliers", value=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:8px 0;">
        <div style="color:#777;font-size:11px;margin:5px 0;font-family:'JetBrains Mono',monospace;">
            🔒 Private &amp; Secure
        </div>
        <div style="color:#777;font-size:11px;margin:5px 0;font-family:'JetBrains Mono',monospace;">
            ⚡ Instant Results
        </div>
        <div style="color:#777;font-size:11px;margin:5px 0;font-family:'JetBrains Mono',monospace;">
            🆓 Free to Use
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1B4F72 50%,#0D2F56 100%);
    padding:60px 40px;border-radius:16px;margin-bottom:30px;">
    <h1 style="color:white;font-family:'JetBrains Mono',monospace;font-size:38px;
        font-weight:700;margin:0 0 15px 0;letter-spacing:-1px;">
        Turn Raw Data Into Powerful Insights
    </h1>
    <p style="color:#34D399;font-size:17px;font-weight:300;margin:0 0 30px 0;
        max-width:800px;font-family:'Space Grotesk',sans-serif;">
        Upload any CSV. Get instant EDA, statistical analysis, ML modeling,
        and a professional PDF report — in seconds.
    </p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <span style="background:rgba(52,211,153,0.12);color:#34D399;padding:7px 18px;
            border-radius:4px;border:1px solid #1a4030;font-weight:600;font-size:12px;
            font-family:'JetBrains Mono',monospace;">Full EDA</span>
        <span style="background:rgba(52,211,153,0.12);color:#34D399;padding:7px 18px;
            border-radius:4px;border:1px solid #1a4030;font-weight:600;font-size:12px;
            font-family:'JetBrains Mono',monospace;">10 Statistical Tests</span>
        <span style="background:rgba(52,211,153,0.12);color:#34D399;padding:7px 18px;
            border-radius:4px;border:1px solid #1a4030;font-weight:600;font-size:12px;
            font-family:'JetBrains Mono',monospace;">15+ Visualizations</span>
        <span style="background:rgba(52,211,153,0.12);color:#34D399;padding:7px 18px;
            border-radius:4px;border:1px solid #1a4030;font-weight:600;font-size:12px;
            font-family:'JetBrains Mono',monospace;">6 ML Models</span>
        <span style="background:rgba(52,211,153,0.12);color:#34D399;padding:7px 18px;
            border-radius:4px;border:1px solid #1a4030;font-weight:600;font-size:12px;
            font-family:'JetBrains Mono',monospace;">PDF + Excel Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📂 Upload & Overview",
    "⚙️ Preprocessing",
    "📊 EDA",
    "🧪 Statistical Tests",
    "📈 Visualizations",
    "💡 AI Insights",
    "📄 PDF Report",
    "🤖 ML Studio",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Overview
# ════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📂 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Drag & drop your CSV file here or click to browse",
        type=["csv"],
        help="Supports CSV files up to 200MB",
        label_visibility="visible"
    )

    st.markdown("**No data? Try our sample dataset:**")
    if st.button("📊 Load Sample Dataset (Restaurant Tips)", type="secondary"):
        sample_df = sns.load_dataset('tips')
        sample_df['tip_percentage'] = (sample_df['tip'] / sample_df['total_bill']) * 100
        sample_df['is_weekend'] = sample_df['day'].isin(['Sat', 'Sun']).astype(int)
        sample_df['party_size_category'] = pd.cut(
            sample_df['size'], bins=[0, 2, 4, 10], labels=['Small', 'Medium', 'Large']
        )
        st.session_state.uploaded_df = sample_df
        st.session_state.loaded_file_name = "sample_tips.csv"
        for k in ['cleaned_df', 'overview', 'stats_results', 'insights',
                  'figures', 'pdf_bytes', 'cleaning_log', 'outlier_counts',
                  'ml_results', 'ml_target', 'ml_best_model']:
            st.session_state[k] = _defaults.get(k, None)
        st.session_state.overview = preprocessor.get_overview(sample_df)
        st.session_state.ml_engine = MLEngine()
        ml_engine = st.session_state.ml_engine
        st.rerun()

    st.markdown("---")

    if uploaded_file is not None or st.session_state.uploaded_df is not None:
        if uploaded_file is not None and uploaded_file.name != st.session_state.loaded_file_name:
            for k, v in _defaults.items():
                if k not in ('preprocessor', 'analyzer', 'visualizer',
                             'insights_gen', 'pdf_gen', 'ml_engine'):
                    st.session_state[k] = v
            st.session_state.loaded_file_name = uploaded_file.name
            st.session_state.ml_engine = MLEngine()
            ml_engine = st.session_state.ml_engine

        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.overview = preprocessor.get_overview(df)

            df      = st.session_state.uploaded_df
            overview = st.session_state.overview

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📋 Total Rows",      f"{overview['rows']:,}")
            col2.metric("📌 Total Columns",   f"{overview['columns']}")
            col3.metric("✅ Complete Rows %", f"{100 - overview['missing_percentage']:.1f}%")
            col4.metric("⚠️ Missing Values",  f"{overview['total_missing']:,}")

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
                st.success("✅ No missing values detected.")

            st.markdown("---")
            st.markdown("### Data Quality Score")
            quality = overview['quality_score']
            color   = '#1E8449' if quality >= 80 else '#F0A500' if quality >= 50 else '#C0392B'
            label   = ('Excellent — ready for analysis' if quality >= 80 else
                       'Good — minor cleaning recommended' if quality >= 50 else
                       'Needs attention — significant issues detected')
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:30px;text-align:center;
                border:2px solid {color};">
                <div style="font-size:48px;font-weight:800;color:{color};
                    font-family:'JetBrains Mono',monospace;">{quality:.0f}/100</div>
                <div style="font-size:16px;color:#6C757D;margin-top:10px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("👆 Upload a CSV file or click 'Load Sample Dataset' to get started!")

# ════════════════════════════════════════════════════════════════
# TAB 2 — Preprocessing
# ════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### ⚙️ Data Preprocessing")

    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Before Preprocessing")
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
            st.metric("Total Rows",     f"{len(df):,}")

        if st.button("🚀 Run Preprocessing", key="preprocess_btn"):
            with st.spinner("Preprocessing data..."):
                cleaned_df, cleaning_log = preprocessor.clean(df)
                if handle_outliers:
                    cleaned_df, outlier_counts = preprocessor.handle_outliers(cleaned_df)
                else:
                    outlier_counts = {}
                st.session_state.cleaned_df     = cleaned_df
                st.session_state.cleaning_log   = cleaning_log
                st.session_state.outlier_counts = outlier_counts
                st.session_state.stats_results  = None
                st.session_state.insights       = None
                st.session_state.pdf_bytes      = None
                st.session_state.ml_results     = None
                st.session_state.ml_engine      = MLEngine()
                ml_engine = st.session_state.ml_engine

        with col2:
            st.markdown("#### After Preprocessing")
            if st.session_state.cleaned_df is not None:
                st.metric("Missing Values", "0")
                st.metric("Duplicate Rows", "0")
                st.metric("Total Rows",     f"{len(st.session_state.cleaned_df):,}")

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
                label="📥 Download Cleaned CSV",
                data=st.session_state.cleaned_df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload data in the 'Upload & Overview' tab first.")

# ════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown("*Column summaries · Distributions · Ranked correlations · Missing patterns*")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        st.markdown("#### 📋 Column Summary")
        summary_data = []
        for col in df.columns:
            col_data = df[col]
            missing     = col_data.isna().sum()
            missing_pct = (missing / len(df)) * 100
            unique      = col_data.nunique()
            dtype       = str(col_data.dtype)

            if pd.api.types.is_numeric_dtype(col_data):
                summary_data.append({
                    'Column':    col,
                    'Type':      dtype,
                    'Unique':    unique,
                    'Missing':   missing,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Mean':      round(col_data.mean(), 2),
                    'Median':    round(col_data.median(), 2),
                    'Std':       round(col_data.std(), 2),
                    'Min':       round(col_data.min(), 2),
                    'Max':       round(col_data.max(), 2),
                    'Skewness':  round(col_data.skew(), 2),
                })
            else:
                top_val  = str(col_data.mode().iloc[0]) if not col_data.mode().empty else 'N/A'
                top_freq = int(col_data.value_counts().iloc[0]) if len(col_data.value_counts()) > 0 else 0
                summary_data.append({
                    'Column':    col,
                    'Type':      dtype,
                    'Unique':    unique,
                    'Missing':   missing,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Top Value': top_val,
                    'Top Freq':  top_freq,
                })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        st.markdown("---")

        st.markdown("#### 📈 Numeric Column Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cols_per_row = 3
            for i in range(0, len(numeric_cols), cols_per_row):
                batch    = numeric_cols[i:i + cols_per_row]
                fig_cols = st.columns(len(batch))
                for j, col in enumerate(batch):
                    with fig_cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.hist(df[col].dropna(), bins=30, color='#34D399',
                                edgecolor='#111', alpha=0.8)
                        ax.set_title(col, fontsize=10, fontweight='bold', color='#444')
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("No numeric columns found.")

        st.markdown("---")

        st.markdown("#### 📊 Categorical Column Distributions")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cols_per_row = 3
            for i in range(0, len(cat_cols), cols_per_row):
                batch    = cat_cols[i:i + cols_per_row]
                fig_cols = st.columns(len(batch))
                for j, col in enumerate(batch):
                    with fig_cols[j]:
                        top_n = df[col].value_counts().nlargest(10)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.barh(top_n.index[::-1], top_n.values[::-1],
                                color='#34D399', edgecolor='#111')
                        ax.set_title(col, fontsize=10, fontweight='bold', color='#444')
                        ax.set_xlabel('Count')
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("No categorical columns found.")

        st.markdown("---")

        st.markdown("#### 🔗 Top Correlations (Ranked)")
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            corr_pairs  = (
                corr_matrix
                .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            corr_pairs.columns = ['Feature A', 'Feature B', 'Pearson r']
            corr_pairs['Abs r'] = corr_pairs['Pearson r'].abs()
            corr_pairs = corr_pairs.sort_values('Abs r', ascending=False).drop(columns='Abs r')
            corr_pairs['Pearson r'] = corr_pairs['Pearson r'].round(4)

            def _color_corr(val):
                try:
                    v = abs(float(val))
                except Exception:
                    return ''
                if v >= 0.7:  return 'background-color: #D5F5E3; color: #1E8449; font-weight:700'
                if v >= 0.4:  return 'background-color: #FFF3CD; color: #F0A500; font-weight:600'
                return 'color: #6C757D'

            st.dataframe(
                corr_pairs.style.map(_color_corr, subset=['Pearson r']),
                use_container_width=True
            )
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

        st.markdown("---")

        st.markdown("#### 🗺️ Missing Value Pattern")
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(12, max(4, len(df.columns) * 0.3)))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=True,
                        cmap='RdYlGn_r', ax=ax)
            ax.set_title('Missing Value Matrix (Yellow = Missing)',
                         fontweight='bold', fontsize=12)
            ax.set_xlabel('Column')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.success("✅ No missing values in the dataset.")

    else:
        st.info("Please preprocess your data in the 'Preprocessing' tab first.")

# ════════════════════════════════════════════════════════════════
# TAB 4 — Statistical Tests
# ════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🧪 Comprehensive Statistical Testing")
    st.markdown("*Scale-aware tests · Effect sizes · FDR correction · OLS regression*")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        if st.button("🔬 Run Statistical Tests", key="run_stats_btn"):
            with st.spinner("Running statistical tests..."):
                stats_results = analyzer.run_all(df)
                st.session_state.stats_results = stats_results
                st.session_state.insights  = None
                st.session_state.pdf_bytes = None

        if st.session_state.stats_results is not None:
            sr = st.session_state.stats_results

            with st.expander("📊 Normality Tests — Shapiro-Wilk · D'Agostino-Pearson · Anderson-Darling", expanded=True):
                norm = sr.get('normality', pd.DataFrame())
                if not norm.empty:
                    styled = norm.style.map(_green, subset=['Normal'])
                    if 'Sig. after FDR' in norm.columns:
                        styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                    st.dataframe(styled, use_container_width=True)

            with st.expander("🔗 Correlation Analysis — Pearson & Spearman with Effect Sizes & FDR", expanded=True):
                corr_col_name = _corr_col(sr.get('pearson_correlation', pd.DataFrame()))
                col1, col2 = st.columns(2)
                for lbl, key, col in [
                    ("Pearson",  'pearson_correlation',  col1),
                    ("Spearman", 'spearman_correlation', col2),
                ]:
                    with col:
                        st.markdown(f"**{lbl}**")
                        cdf = sr.get(key, pd.DataFrame())
                        if not cdf.empty and corr_col_name in cdf.columns:
                            top    = cdf.nlargest(10, corr_col_name)
                            styled = top.style
                            if 'Effect Size' in top.columns:
                                styled = styled.map(_color_effect, subset=['Effect Size'])
                            if 'Sig. after FDR' in top.columns:
                                styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                            st.dataframe(styled, use_container_width=True)

            for name, key, extra_cols in [
                ("📐 Chi-Square Independence Test + Cramér's V Effect Size", 'chi_square',   ['Effect Size']),
                ("📏 One-Way ANOVA + Eta-Squared (η²) Effect Size",          'anova',        ['Effect Size']),
                ("📉 Mann-Whitney U Test + Rank-Biserial Correlation",        'mann_whitney', ['Effect Size']),
                ("⚖️ Levene's Test — Equality of Variances",                 'levene',       []),
            ]:
                df_test = sr.get(key, pd.DataFrame())
                if not df_test.empty:
                    with st.expander(name, expanded=False):
                        sig_c  = _sig_col(df_test)
                        styled = df_test.style.map(_green, subset=[sig_c])
                        for ec in extra_cols:
                            if ec in df_test.columns:
                                styled = styled.map(_color_effect, subset=[ec])
                        st.dataframe(styled, use_container_width=True)

            vif = sr.get('vif', pd.DataFrame())
            if not vif.empty:
                with st.expander("🏗️ Multicollinearity Analysis — VIF Scores + Condition Number (κ)", expanded=False):
                    styled = vif.style.map(_color_vif, subset=['VIF'])
                    if 'κ Risk' in vif.columns:
                        styled = styled.map(
                            lambda v: ('color:#C0392B;font-weight:700' if v == 'High' else
                                       'color:#F0A500'                  if v == 'Moderate' else
                                       'color:#1E8449'),
                            subset=['κ Risk'])
                    st.dataframe(styled, use_container_width=True)
                    if 'Condition Number (κ)' in vif.columns:
                        kappa  = vif['Condition Number (κ)'].iloc[0]
                        k_risk = vif['κ Risk'].iloc[0]
                        k_color = ('#C0392B' if k_risk == 'High' else
                                   '#F0A500'  if k_risk == 'Moderate' else '#1E8449')
                        st.markdown(f"""
                        <div style="background:white;padding:16px;border-radius:8px;
                            border-left:4px solid {k_color};margin-top:12px;">
                            <strong>Condition Number κ = {kappa:.2f}</strong> — {k_risk} risk
                        </div>
                        """, unsafe_allow_html=True)

            reg_df    = sr.get('regression', pd.DataFrame())
            ols_stats = sr.get('ols_model_stats', {})
            if not reg_df.empty and ols_stats:
                with st.expander(f"📈 OLS Regression — Predicting: {ols_stats.get('Target', '')} | R²={ols_stats.get('R²', 0):.3f}", expanded=True):
                    m = ols_stats
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("R²",      f"{m.get('R²', 0):.4f}")
                    mc2.metric("Adj. R²", f"{m.get('Adj. R²', 0):.4f}")
                    mc3.metric("F",       f"{m.get('F-Statistic', 0):.3f}")
                    mc4.metric("AIC",     f"{m.get('AIC', 0):.1f}")
                    mc5.metric("BIC",     f"{m.get('BIC', 0):.1f}")
                    sig_c  = _sig_col(reg_df)
                    styled = reg_df.style.map(_green, subset=[sig_c])
                    if 'Sig. after FDR' in reg_df.columns:
                        styled = styled.map(_color_fdr, subset=['Sig. after FDR'])
                    st.dataframe(styled, use_container_width=True)
    else:
        st.info("Please preprocess your data in the 'Preprocessing' tab first.")

# ════════════════════════════════════════════════════════════════
# TAB 5 — Visualizations
# ════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 📈 Interactive Visualizations")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        with st.expander("📊 Distribution Analysis", expanded=True):
            st.markdown("**Histograms with KDE Overlay**")
            fig = visualizer.histograms(df)
            if fig: st.pyplot(fig); plt.close(fig)
            st.markdown("**Box Plots**")
            fig = visualizer.boxplots(df)
            if fig: st.pyplot(fig); plt.close(fig)

        with st.expander("🔗 Correlation Visualizations", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pearson Heatmap**")
                fig = visualizer.correlation_heatmap(df, method='pearson')
                if fig: st.pyplot(fig); plt.close(fig)
            with col2:
                st.markdown("**Spearman Heatmap**")
                fig = visualizer.correlation_heatmap(df, method='spearman')
                if fig: st.pyplot(fig); plt.close(fig)
            st.markdown("**Pairplot**")
            fig = visualizer.pairplot(df)
            if fig: st.pyplot(fig); plt.close(fig)

        with st.expander("📐 Normality Assessment", expanded=False):
            fig = visualizer.qq_plots(df)
            if fig: st.pyplot(fig); plt.close(fig)

        sr = st.session_state.stats_results
        if sr and not sr.get('vif', pd.DataFrame()).empty:
            with st.expander("🏗️ Multicollinearity", expanded=False):
                fig = visualizer.vif_chart(sr['vif'])
                if fig: st.pyplot(fig); plt.close(fig)
    else:
        st.info("Please preprocess your data first.")

# ════════════════════════════════════════════════════════════════
# TAB 6 — AI Insights
# ════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 💡 What Your Data Is Telling You")

    if st.session_state.cleaned_df is not None and st.session_state.stats_results is not None:
        df           = st.session_state.cleaned_df
        stats_results = st.session_state.stats_results
        overview     = st.session_state.overview

        if st.button("🤖 Generate AI Insights", key="insights_btn"):
            with st.spinner("Generating insights..."):
                vif_df   = stats_results.get('vif', pd.DataFrame())
                insights = insights_gen.generate(df, stats_results, vif_df, overview)
                st.session_state.insights = insights

        if st.session_state.insights is not None:
            insights = st.session_state.insights

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;
                    border:2px solid #1E8449;height:100%;">
                    <h4 style="color:#1E8449;">✅ Data Quality</h4>
                    <p><strong>Score:</strong> {overview['quality_score']:.0f}/100</p>
                    <p><strong>Complete:</strong> {100 - overview['missing_percentage']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                highlights = '<br>'.join([f"• {h}" for h in insights.get('highlights', [])[:3]])
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;
                    border:2px solid #F0A500;height:100%;">
                    <h4 style="color:#F0A500;">🔍 Key Findings</h4>
                    <p>{highlights or 'None'}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                warn_items = '<br>'.join([f"• {w}" for w in insights.get('warnings', [])[:3]])
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:12px;
                    border:2px solid #C0392B;height:100%;">
                    <h4 style="color:#C0392B;">⚠️ Warnings</h4>
                    <p>{warn_items or 'None'}</p>
                </div>
                """, unsafe_allow_html=True)

            if insights.get('summary'):
                st.markdown("---")
                st.markdown("#### 📋 Summary")
                for s in insights['summary']:
                    st.markdown(f"- {s}")

            readiness = insights.get('readiness_score', 0)
            r_color   = ('#1E8449' if readiness >= 70 else
                         '#F0A500'  if readiness >= 40 else '#C0392B')
            st.markdown(f"""
            <div style="text-align:center;padding:30px;">
                <h3>Readiness Score</h3>
                <div style="width:150px;height:150px;border-radius:50%;
                    background:conic-gradient({r_color} {readiness}%, #E0E6ED {readiness}%);
                    display:inline-flex;align-items:center;justify-content:center;margin:20px;">
                    <div style="width:120px;height:120px;border-radius:50%;background:white;
                        display:flex;align-items:center;justify-content:center;
                        font-size:36px;font-weight:800;color:{r_color};
                        font-family:'JetBrains Mono',monospace;">{readiness:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if insights.get('recommendation'):
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0A2342,#1B4F72);color:white;
                    padding:20px;border-radius:12px;margin-top:20px;">
                    <h4 style="color:#34D399;">💡 Recommendation</h4>
                    <p>{insights['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Complete preprocessing and statistical tests first.")

# ════════════════════════════════════════════════════════════════
# TAB 7 — PDF Report
# ════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 📄 Generate Professional PDF Report")

    if st.session_state.cleaned_df is not None and st.session_state.stats_results is not None:
        st.markdown("""
        <div style="background:white;padding:24px;border-radius:12px;
            border-left:4px solid #F0A500;margin-bottom:20px;">
            <h4 style="color:#0A2342;">Your Report Will Include:</h4>
            <p>✓ Executive Summary with Key Metrics</p>
            <p>✓ Complete Data Overview</p>
            <p>✓ All Statistical Tests</p>
            <p>✓ All Visualizations</p>
            <p>✓ AI-Powered Insights &amp; Recommendations</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Generate My PDF Report", key="pdf_btn"):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            figures      = []
            try:
                status_text.markdown("📊 **Compiling statistics...**")
                progress_bar.progress(20)
                time.sleep(0.3)

                status_text.markdown("🎨 **Rendering visualizations...**")
                progress_bar.progress(45)
                figure_fns = [
                    lambda: visualizer.histograms(st.session_state.cleaned_df),
                    lambda: visualizer.correlation_heatmap(st.session_state.cleaned_df),
                    lambda: visualizer.boxplots(st.session_state.cleaned_df),
                ]
                for fn in figure_fns:
                    try:
                        fig = fn()
                        if fig:
                            figures.append(fig)
                    except Exception as fig_err:
                        st.warning(f"A visualization could not be rendered: {fig_err}")
                time.sleep(0.3)

                status_text.markdown("📝 **Writing insights...**")
                progress_bar.progress(70)
                time.sleep(0.3)

                status_text.markdown("📄 **Building PDF...**")
                progress_bar.progress(85)
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
                status_text.markdown("✅ **Report generated successfully!**")

            except Exception as e:
                st.error(f"Error generating PDF: {e}")
            finally:
                for fig in figures:
                    plt.close(fig)

        if st.session_state.pdf_bytes is not None:
            st.markdown("---")
            st.success("🎉 Your report is ready!")
            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=st.session_state.pdf_bytes,
                    file_name=f"statspro_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )

            with col_dl2:
                if st.session_state.cleaned_df is not None:
                    excel_buffer = io.BytesIO()
                    try:
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            st.session_state.cleaned_df.to_excel(
                                writer, index=False, sheet_name='Cleaned Data'
                            )
                            if st.session_state.stats_results:
                                for sheet_name, result_df in st.session_state.stats_results.items():
                                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                                        safe_name = sheet_name[:31]
                                        result_df.to_excel(writer, index=False, sheet_name=safe_name)
                        excel_buffer.seek(0)
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_buffer,
                            file_name=f"statspro_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                    except Exception as xl_err:
                        st.error(f"Excel export failed: {xl_err}")
    else:
        st.info("Please complete all previous steps to generate the PDF report.")

# ════════════════════════════════════════════════════════════════
# TAB 8 — ML Studio
# ════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### 🤖 Machine Learning Studio")
    st.markdown("*Auto-detects task · Trains 6 models · 5-fold CV · Live predictions*")

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            target_col = st.selectbox("🎯 Select target column to predict:", df.columns.tolist())

        if target_col:
            task  = ml_engine.detect_task(df, target_col)
            emoji = "📈" if task == "regression" else "🏷️"
            with col_t2:
                st.markdown(f"""
                <div style="background:#141414;border:1px solid #242424;border-radius:8px;
                    padding:18px 20px;margin-top:10px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:24px;">{emoji}</span>
                    <span style="color:#34D399;font-family:'JetBrains Mono',monospace;
                        font-size:16px;font-weight:600;margin-left:10px;">
                        Task: {task.title()}
                    </span>
                    <span style="color:#777;font-family:'JetBrains Mono',monospace;
                        font-size:12px;margin-left:12px;">
                        {df[target_col].nunique()} unique values
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚀 Train All Models", key="ml_train_btn", type="primary"):
            with st.spinner("Training 6 models with 5-fold CV..."):
                try:
                    results = ml_engine.run(df, target_col)
                    st.session_state.ml_results    = results
                    st.session_state.ml_target     = target_col
                    st.session_state.ml_best_model = ml_engine.best_model_name
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if st.session_state.ml_results is not None:
            results = st.session_state.ml_results

            st.markdown(f"""
            <div style="background:#0a2a1a;border:1px solid #34D399;border-radius:10px;
                padding:20px;margin:16px 0;">
                <span style="color:#34D399;font-family:'JetBrains Mono',monospace;
                    font-size:13px;">🏆 Best Model:</span>
                <span style="color:#fff;font-family:'JetBrains Mono',monospace;
                    font-size:18px;font-weight:700;margin-left:8px;">
                    {ml_engine.best_model_name}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Model Comparison")
            def highlight_best(row):
                if row['Model'] == ml_engine.best_model_name:
                    return ['background-color:#0a2a1a;color:#34D399;font-weight:bold'] * len(row)
                return [''] * len(row)

            styled_results = (
                results.style.apply(highlight_best, axis=1)
                if 'Error' not in results.columns
                else results
            )
            st.dataframe(styled_results, use_container_width=True)

            if ml_engine.feature_importance is not None:
                st.markdown("#### 📈 Feature Importance")
                fig, ax = plt.subplots(figsize=(8, 4))
                fi = ml_engine.feature_importance.head(15)
                ax.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='#34D399')
                ax.set_xlabel('Importance')
                ax.set_title(f'Feature Importance — {ml_engine.best_model_name}',
                             fontweight='bold')
                st.pyplot(fig)
                plt.close(fig)

            # ── Live Prediction ───────────────────────────────
            st.markdown("---")
            st.markdown("#### 🔮 Live Prediction")
            st.markdown("Enter values below to get a prediction from the best model:")

            input_data = {}
            pred_cols  = st.columns(min(4, len(ml_engine.feature_cols)))
            for i, col in enumerate(ml_engine.feature_cols):
                with pred_cols[i % len(pred_cols)]:
                    if col in df.columns:
                        if df[col].dtype in ['object', 'category']:
                            options = df[col].dropna().unique().tolist()
                            input_data[col] = st.selectbox(col, options, key=f"pred_{col}")
                        else:
                            col_min, col_max, col_mean = _safe_col_stats(df[col])
                            input_data[col] = st.number_input(
                                col,
                                value=col_mean,
                                min_value=col_min,
                                max_value=col_max,
                                key=f"pred_{col}"
                            )
                    else:
                        input_data[col] = st.number_input(col, value=0.0, key=f"pred_{col}")

            if st.button("🔮 Predict", key="predict_btn", type="primary"):
                try:
                    input_df   = pd.DataFrame([input_data])
                    prediction = ml_engine.predict(input_df)[0]
                    st.markdown(f"""
                    <div style="background:#0a2a1a;border:1px solid #34D399;border-radius:10px;
                        padding:20px;margin:16px 0;text-align:center;">
                        <span style="color:#34D399;font-family:'JetBrains Mono',monospace;
                            font-size:14px;">Prediction:</span>
                        <span style="color:#fff;font-family:'JetBrains Mono',monospace;
                            font-size:24px;font-weight:700;margin-left:10px;">
                            {prediction}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("Please preprocess your data first in the 'Preprocessing' tab.")

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:20px;color:#6C757D;font-size:13px;
    font-family:'JetBrains Mono',monospace;">
    Built with ❤️ using Streamlit &nbsp;|&nbsp; StatsPro v2.3 &nbsp;|&nbsp; © 2025
</div>
""", unsafe_allow_html=True)
