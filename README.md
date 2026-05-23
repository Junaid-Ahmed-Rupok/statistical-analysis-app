# 📊 StatsPro — AI-Powered Statistical Analysis Platform

<div align="center">

![StatsPro](https://img.shields.io/badge/StatsPro-v2.3-emerald?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?style=for-the-badge&logo=streamlit)
![ML](https://img.shields.io/badge/ML-6%20Models-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Turn Raw Data Into Powerful Insights — Instantly.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment)

</div>

---

## 🌟 Overview

StatsPro is a production-grade Streamlit web application that transforms any CSV file into comprehensive exploratory data analysis, statistical testing, machine learning modeling, and professional PDF reports — all in seconds.

Built with a premium dark theme, monospace typography, and emerald green accents. Designed to make you look like a senior data scientist without writing a single line of code.

### 🎯 Live Demo

👉 **[Try StatsPro Now](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app)**

---

## ✨ Features

### 📂 Smart Data Upload & Overview
- Drag-and-drop CSV upload (up to 200MB)
- Built-in sample dataset for instant testing
- Automatic data quality scoring (0-100)
- Real-time missing value analysis with color-coded table
- Column type detection and memory usage

### ⚙️ Intelligent Preprocessing
- KNN imputation for numeric columns (k=5)
- Mode imputation for categorical columns
- IQR-based outlier detection with Winsorization (capping)
- Exact duplicate removal
- Constant column detection and removal
- Download cleaned dataset as CSV
- Full preprocessing activity log

### 📊 Exploratory Data Analysis (EDA)
- Column-by-column summary (type, unique values, missing %, statistics)
- Numeric distribution histograms
- Categorical bar charts (top 10 values)
- Ranked correlation table with color-coded strength
- Missing value heatmap
- Descriptive statistics (mean, median, std, min, max, skewness)

### 🧪 Comprehensive Statistical Testing
| Test | Description | Extras |
|------|-------------|--------|
| Shapiro-Wilk / D'Agostino-Pearson / Anderson-Darling | Scale-aware normality testing | Skewness & Kurtosis |
| Pearson & Spearman | Correlation analysis | 95% Confidence Intervals |
| Chi-Square | Categorical independence | Cramér's V effect size |
| One-Way ANOVA | Group comparisons | Eta-Squared (η²) effect size |
| Mann-Whitney U | Non-parametric group test | Rank-Biserial correlation |
| Levene's Test | Variance equality | — |
| VIF Analysis | Multicollinearity detection | Condition Number (κ) |
| OLS Regression | Auto-target linear modeling | R², AIC, BIC, full summary |

All tests include **Benjamini-Hochberg FDR correction** to protect against false positives.

### 📈 Professional Visualizations (15+)
- Histograms with KDE overlay
- Box plots with outlier markers
- Pearson & Spearman correlation heatmaps
- Pairwise scatter plots (pairplot)
- Q-Q plots for normality assessment
- VIF bar charts with risk color-coding

### 💡 AI-Powered Insights
- Executive data quality tier assessment
- Missing data mechanism analysis (MCAR/MAR/MNAR)
- Skewness & kurtosis interpretation with transformation recommendations
- Correlation strength analysis with confidence intervals
- Multicollinearity remediation paths
- FDR correction impact quantification
- Prioritized recommendations (P1/P2/P3)
- Production readiness score (0-100)

### 🤖 Machine Learning Studio
- **Auto-detection** of regression vs classification tasks
- **6 models trained simultaneously**:
  - Linear/Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - CatBoost
  - AdaBoost
- **5-fold cross-validation** with held-out test metrics
- Model comparison table (best model highlighted)
- Feature importance visualization
- **Live prediction form** — enter values, get instant predictions
- Automatic handling of mixed-type columns (numeric + string values)

### 📄 PDF Report Generation
- Professional cover page with gold accents
- Executive summary with KPI cards
- Complete data overview and column summaries
- All statistical test results with color-coded p-values
- High-resolution embedded visualizations
- AI-powered insights and recommendations
- Confidentiality watermarks and page numbers

### 📥 Export Options
- **PDF Report** — professional, branded, shareable
- **Excel Report** — all data + test results in separate sheets
- **Cleaned CSV** — download preprocessed data

---

## 🎨 Design System

- **Theme:** Pure black background with emerald green (#34D399) accents
- **Typography:** JetBrains Mono (headings, code) + Space Grotesk (body)
- **Animations:** Fade-in on elements, cursor blink on metrics, hover effects
- **Responsive:** Fully functional on desktop and tablet

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app.git
cd statistical-analysis-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

Open http://localhost:8501 in your browser.

---

## 📦 Project Structure
statistical-analysis-app/
├── app.py                    # Main Streamlit application (8 tabs)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── README.md                 # This file
├── styles/
│   └── main.css              # Complete dark theme CSS
└── modules/
    ├── __init__.py
    ├── preprocessor.py       # Data cleaning & preprocessing
    ├── statistics.py         # Statistical analysis engine (10 tests)
    ├── visualizer.py         # Chart generation (15+ types)
    ├── insights.py           # AI-powered insights generator
    ├── pdf_generator.py      # Professional PDF report builder
    └── ml_engine.py          # AutoML pipeline (6 models)



---

## 🎯 Usage Guide

1. **Upload** → Tab 1 — Upload CSV or load sample dataset
2. **Preprocess** → Tab 2 — Clean data, handle outliers
3. **EDA** → Tab 3 — Column summaries, distributions, correlations
4. **Statistics** → Tab 4 — Run 10 statistical tests
5. **Visualizations** → Tab 5 — Charts, heatmaps, Q-Q plots
6. **Insights** → Tab 6 — AI-generated narrative report
7. **Export** → Tab 7 — Download PDF + Excel reports
8. **ML Studio** → Tab 8 — Train 6 models, make predictions

---

## 🌐 Deployment

Already deployed on Streamlit Cloud. Auto-updates on every push to `main`.

To deploy your own:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → Select repo → Deploy

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Streamlit |
| Data | Pandas, NumPy |
| Statistics | SciPy, Statsmodels |
| ML | Scikit-learn, XGBoost, CatBoost |
| Charts | Matplotlib, Seaborn |
| PDF | ReportLab |
| Excel | OpenPyXL |
| Design | Custom CSS, JetBrains Mono, Space Grotesk |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built with ❤️ by Junaid Ahmed Rupok**

*StatsPro — Professional Statistical Analysis Made Simple*

</div>

