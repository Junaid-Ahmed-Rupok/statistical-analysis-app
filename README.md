<div align="center">

# StatsPro

**AI-Powered Statistical Analysis Platform**

[![Version](https://img.shields.io/badge/version-2.3-34D399?style=flat-square)](https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.67-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)](LICENSE)

Turn raw CSV data into professional statistical reports — no code required.

[**Live Demo →**](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app)

</div>

---

## 📸 Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/images/03-assumption-doctor.png" alt="Assumption Doctor"/><br/><sub><b>Assumption Doctor</b> — checks assumptions, switches to a safer test, explains why</sub></td>
    <td width="50%"><img src="docs/images/05-ml-trust-report.png" alt="ML Trust Report"/><br/><sub><b>ML Trust Report</b> — calibrated prediction intervals and fairness by group</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/04-writeup.png" alt="Thesis-ready write-up"/><br/><sub><b>Thesis-ready write-up</b> — APA Methods and Results, plus a script that reproduces every number</sub></td>
    <td width="50%"><img src="docs/images/06-pdf-integrity-seal.png" alt="Integrity Seal"/><br/><sub><b>Integrity Seal</b> — SHA-256 fingerprint and QR code in every PDF report</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/01-upload-overview.png" alt="Upload and overview"/><br/><sub><b>Upload &amp; Overview</b> — data quality score at a glance</sub></td>
    <td width="50%"><img src="docs/images/02-statistical-tests.png" alt="Statistical tests"/><br/><sub><b>Statistical Tests</b> — scale-aware tests with effect sizes and FDR correction</sub></td>
  </tr>
</table>

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
- Python **3.11 or newer** (check with `python --version`) — or just Docker

### Run it (pick one)

**Any machine with Docker** (no Python needed):

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app.git
cd statistical-analysis-app
docker build -t statspro .
docker run -p 8501:8501 statspro
```

**macOS / Linux:** `./run.sh`  **Windows:** double-click `run.bat`

Both create a private `.venv`, install dependencies and start the app.

**Manual:**

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/statistical-analysis-app.git
cd statistical-analysis-app
python -m venv .venv
# macOS/Linux: source .venv/bin/activate    Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

The app has 11 tabs, meant to be used roughly in order:

| Tab | Purpose |
|---|---|
| 1. Upload | Load a CSV or use the built-in sample dataset |
| 2. Preprocess | Clean data and handle outliers |
| 3. EDA | Explore distributions and correlations |
| 4. Statistics | Run all 10 statistical tests |
| 5. Visualizations | Generate charts, heatmaps, and Q-Q plots |
| 6. Insights | Read the AI-generated narrative report |
| 7. PDF Report | Download the PDF report |
| 8. ML Studio | Train 6 models, check uncertainty and fairness, make live predictions |
| 9. Assumption Doctor | See which test fits your data and why |
| 10. Write-up | APA-style Methods/Results text and a reproduce script |
| 11. About | About the developer and how to get in touch |

---

## Project Structure

```
statistical-analysis-app/
├── app.py                  # Main application (11 tabs)
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Test dependencies
├── Dockerfile, run.sh, run.bat   # Run anywhere
├── MONETIZATION.md         # Optional Pro licence setup
├── index.html              # GitHub Pages landing page
├── keepalive.py            # Keeps the Streamlit app awake (GitHub Actions)
├── data/tips.csv           # Bundled sample dataset (works offline)
├── styles/
│   └── main.css            # Dark theme CSS
├── modules/
│   ├── preprocessor.py     # Data cleaning pipeline
│   ├── statistics.py       # Statistical testing engine
│   ├── advisor.py          # Assumption Doctor (picks the right test)
│   ├── writeup.py          # APA write-up + reproduce.py generator
│   ├── seal.py             # Integrity Seal + Honesty Ledger
│   ├── visualizer.py       # Chart generation (15+ types)
│   ├── insights.py         # AI insights generator
│   ├── pdf_generator.py    # PDF report builder
│   ├── ml_engine.py        # AutoML pipeline + Trust Report
│   └── licensing.py        # Optional Pro licence check
└── tests/                  # pytest suite (run: pip install -r requirements-dev.txt && pytest)
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

## ✨ What makes StatsPro different

- **🩺 Assumption Doctor** — checks normality, variance and outliers for every comparison, switches to Welch's ANOVA / Mann-Whitney / Kruskal-Wallis / Spearman when the classic test would mislead, and explains each choice in plain English (with FDR correction across all tests).
- **✍️ Thesis-ready write-up** — APA-style Methods and Results paragraphs, plus a `reproduce.py` script that recomputes every reported p-value from your data (tested to match).
- **🛡️ Trust Report in ML Studio** — prediction intervals / sets with *measured* coverage (split-conformal), per-group performance and fairness gaps (even for columns left out of the model), and warnings when a result looks too good to be true.

- **🔐 Integrity Seal & Honesty Ledger (in every PDF)** — the last page fingerprints your exact data (SHA-256, checkable with `sha256sum cleaned_data.csv` or the in-app verifier) and states how many hypotheses were tested, how many would look significant by chance alone, and how many survive multiple-testing correction.

## 💎 Pro & services

Everything analytical is free. **PDF report export, reproducible-script export, batch prediction and model download** are Pro features, unlocked by a licence key (off by default — see [MONETIZATION.md](MONETIZATION.md)). Need hands-on help? Use **Work with me** on the About tab.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built by [Junaid Ahmed Rupok](https://github.com/Junaid-Ahmed-Rupok)

Built with [Streamlit](https://streamlit.io), [SciPy](https://scipy.org), [statsmodels](https://www.statsmodels.org) and [scikit-learn](https://scikit-learn.org)

</div>
