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

## 👨‍💻 About the Developer

<div align="center">
<img src="https://avatars.githubusercontent.com/Junaid-Ahmed-Rupok" width="100" style="border-radius:50%"/>

### Sarder Junaid Ahmed
**Data Scientist & Machine Learning Engineer**

*Transforming complex data into strategic decisions through rigorous statistical modeling and production-ready machine learning systems.*

[![GitHub](https://img.shields.io/badge/GitHub-Junaid--Ahmed--Rupok-181717?logo=github)](https://github.com/Junaid-Ahmed-Rupok)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarder%20Junaid%20Ahmed-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarder-junaid-ahmed-059b68240/)
[![Portfolio](https://img.shields.io/badge/Portfolio-junaid--ahmed--rupok.github.io-1E88E5?logo=githubpages&logoColor=white)](https://junaid-ahmed-rupok.github.io/__portfolio__Yes/)
[![Email](https://img.shields.io/badge/Email-junaidahmedrupok%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:junaidahmedrupok@gmail.com)
</div>

**Specializations:** Statistical ML · Causal Inference · Trustworthy AI · Fairness-Aware ML · RAG Systems

**Selected Research:**
- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* Journal of Machine Learning and Applications, 2(1), 9–17. [DOI: 10.61577/jmla.2026.100002](https://doi.org/10.61577/jmla.2026.100002)
- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* IEEE SPICSCON 2026, BAUET, Bangladesh (Aug 13–14, 2026). **Accepted for Presentation** — IEEE Xplore.
- 📄 **Ahmed, S.J.** et al. (2026). *CF-EGAT: A Causal Fairness-Aware Equity Graph Attention Network for Country-Level Environmental Livability Classification.* SPECTRA 2026. 🏆 **1st Best Paper Award**
- 📄 **Ahmed, S.J.** (2025). *Multi-Dimensional Statistical Similarity for Governance Classification: Beyond Arbitrary Thresholds.* APMEE 2025. 🏆 **Best Research Paper Award**
- 📄 **Ahmed, S.J.** (2026). *DeepEnMap: Ordinal-Aware Multi-Modal Deep Learning for Energy Poverty Risk Mapping.* IEMIS 2026, University of British Columbia, Vancouver, Canada (Aug 10–12, 2026). **Accepted for Presentation** — Springer LNNS Series (Scopus, EI-Compendex, DBLP, ISI Proceedings).
- 📄 **Ahmed, S.J.** (2026). *Density-Decoupled, Mask-Ablated Segmentation-Guided Diffusion for Controllable Mammography Synthesis: A Preliminary Study.* IEMIS 2026, University of British Columbia, Vancouver, Canada (Aug 10–12, 2026). **Accepted for Presentation** — Springer LNNS Series (Scopus, EI-Compendex, DBLP, ISI Proceedings).
- 📄 **Ahmed, S.J.**, Islam Nahian, M.T., & Kwoshik, M.H.R. (2026). *Environmental Livability Assessment via Adaptive Bootstrap-Retrained SHAP and Statistically-Constrained Pareto Counterfactuals: A Cross-National Analysis.* IEEE SPICSCON 2026, BAUET, Bangladesh (Aug 13–14, 2026). **Accepted for Presentation** — IEEE Xplore.
- 📄 **Ahmed, S.J.** (2026). *DemocracyGuard: Testing a Divergence-Index Reconciliation of Subjective and Objective Democracy Indicators for Forecasting Adverse Regime Transitions.* **Under Review**, Transactions on Machine Learning Research (TMLR) — Q1, Top-Tier Journal.
- 📄 **Ahmed, S.J.** (2026). *FAI: Feature-Wise Adaptive Imputation via Downstream-Aware Method Selection.* **Under Review**, ICISET 2026 (IEEE Xplore).

**Other Deployed Projects:**
- 🔬 [ReproHub](https://reproapp-8jb7vbhnqyltxq23bsr8xn.streamlit.app/) — Automated research reproducibility platform with composite scoring across 11 statistical tests
- 📊 [StatsPro](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app/) — AI-powered statistical analysis platform with automated CSV-to-report workflows

**Honors:**
🏆 1st Best Paper — SPECTRA 2026 &nbsp;·&nbsp;
🏆 Best Research Paper — APMEE 2025 &nbsp;·&nbsp;
🎖️ Esteemed Alumni Award — YLRL RUET 2024 &nbsp;·&nbsp;
⭐ Perfect GPA 5.00/5.00 — SSC & HSC &nbsp;·&nbsp;
🎓 National Merit Scholarship — 2009 & 2013

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built by [Junaid Ahmed Rupok](https://github.com/Junaid-Ahmed-Rupok)

Built with [Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [Groq](https://groq.com) · [FAISS](https://github.com/facebookresearch/faiss) · [sentence-transformers](https://www.sbert.net)

</div>
```
