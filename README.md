<div align="center">

# StatsPro

**AI-Powered Statistical Analysis Platform**

[![Version](https://img.shields.io/badge/version-2.3-34D399?style=flat-square)](https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.57-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)](LICENSE)

Turn raw CSV data into professional statistical reports — no code required.

[**Live Demo →**](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app)

</div>

---

## What is StatsPro?

StatsPro is a production-ready Streamlit web app that takes any CSV file and delivers comprehensive exploratory data analysis, statistical hypothesis testing, automated machine learning, and professional PDF reports — all through a polished, code-free interface.

It's built for analysts, researchers, and data scientists who need rigorous results fast, with a premium dark theme (pure black + emerald green) and monospace typography designed for extended analytical sessions.

---

## Features at a Glance

### Data Upload & Quality Scoring
- Drag-and-drop CSV upload up to 200 MB
- Built-in sample dataset for instant testing
- Automatic data quality score (0–100) with missing value heatmap
- Column type detection and memory usage summary

### Intelligent Preprocessing
- KNN imputation (k=5) for numeric columns; mode imputation for categorical
- IQR-based outlier detection with Winsorization
- Exact duplicate and constant-column removal
- Full preprocessing activity log
- Download cleaned dataset as CSV

### Exploratory Data Analysis
- Per-column summary: type, unique count, missing %, descriptive stats
- Histograms, bar charts (top 10), ranked correlation table
- Skewness, kurtosis, and missing value heatmap

### Statistical Testing

Ten tests, all with **Benjamini-Hochberg FDR correction** to control false positives:

| Test | Use Case | Effect Size |
|---|---|---|
| Shapiro-Wilk / D'Agostino-Pearson / Anderson-Darling | Normality | Skewness & Kurtosis |
| Pearson & Spearman Correlation | Linear & monotonic relationships | 95% Confidence Intervals |
| Chi-Square | Categorical independence | Cramér's V |
| One-Way ANOVA | Group mean comparisons | Eta-Squared (η²) |
| Mann-Whitney U | Non-parametric group test | Rank-Biserial r |
| Levene's Test | Variance equality | — |
| VIF Analysis | Multicollinearity detection | Condition Number (κ) |
| OLS Regression | Auto-target linear modeling | R², AIC, BIC |

### Visualizations (15+ chart types)
- Histograms with KDE overlay
- Box plots with outlier markers
- Pearson & Spearman correlation heatmaps
- Pairwise scatter plots
- Q-Q plots for normality assessment
- VIF bar charts with risk color-coding

### AI-Powered Insights
- Executive data quality tier assessment
- Missing data mechanism classification (MCAR / MAR / MNAR)
- Transformation recommendations based on skewness and kurtosis
- Multicollinearity remediation paths
- Prioritized action items (P1 / P2 / P3)
- Production readiness score (0–100)

### Machine Learning Studio
- Auto-detects regression vs. classification tasks
- Trains 6 models simultaneously: Linear/Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost
- 5-fold cross-validation with held-out test metrics
- Model comparison table with the best model highlighted
- Feature importance visualization
- Live prediction form — enter values, get instant results
- Handles mixed-type columns automatically

### Export Options
- **PDF Report** — professional cover page, KPI cards, embedded charts, AI insights, page numbers, and confidentiality watermarks
- **Excel Report** — all data and test results in separate sheets
- **Cleaned CSV** — preprocessed data ready for downstream use

---

## Quickstart

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app.git
cd statistical-analysis-app
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

The app is organized into 8 tabs, meant to be used in order:

| Tab | Purpose |
|---|---|
| 1. Upload | Load a CSV or use the built-in sample dataset |
| 2. Preprocess | Clean data and handle outliers |
| 3. EDA | Explore distributions and correlations |
| 4. Statistics | Run all 10 statistical tests |
| 5. Visualizations | Generate charts, heatmaps, and Q-Q plots |
| 6. Insights | Read the AI-generated narrative report |
| 7. Export | Download PDF and Excel reports |
| 8. ML Studio | Train 6 models and make live predictions |

---

## Project Structure

```
statistical-analysis-app/
├── app.py                  # Main application (8 tabs)
├── requirements.txt        # Dependencies
├── styles/
│   └── main.css            # Dark theme CSS
└── modules/
    ├── preprocessor.py     # Data cleaning pipeline
    ├── statistics.py       # Statistical testing engine
    ├── visualizer.py       # Chart generation (15+ types)
    ├── insights.py         # AI insights generator
    ├── pdf_generator.py    # PDF report builder
    └── ml_engine.py        # AutoML pipeline
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Streamlit |
| Data | Pandas, NumPy |
| Statistics | SciPy, Statsmodels |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Visualization | Matplotlib, Seaborn |
| PDF | ReportLab |
| Excel | OpenPyXL |

---

## Deployment

The app is deployed on Streamlit Cloud and auto-updates on every push to `main`.

To deploy your own fork:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account, select the repo, and deploy

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Junaid Ahmed Rupok](https://github.com/Junaid-Ahmed-Rupok)

</div>
