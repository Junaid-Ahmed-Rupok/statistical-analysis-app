# 📊 StatsPro — AI-Powered Statistical Analysis Platform

<div align="center">

![StatsPro Banner](https://img.shields.io/badge/StatsPro-v1.0-gold?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Turn Raw Data Into Powerful Insights — Instantly.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment)

</div>

---

## 🌟 Overview

StatsPro is a premium, production-grade Streamlit web application that transforms CSV data into comprehensive statistical analysis, beautiful visualizations, and professional PDF reports. Built with a focus on user experience, design excellence, and statistical rigor.

### Why StatsPro?

- **🎨 Premium Design** — Inspired by Mixpanel, Tableau, and Notion AI
- **⚡ Instant Analysis** — 9 statistical tests, 15+ visualizations in seconds
- **🤖 AI Insights** — Automated interpretation and recommendations
- **📄 Professional Reports** — Downloadable PDF reports for sharing
- **🔒 Private & Secure** — All processing happens locally, no data leaves your browser

---

## ✨ Features

### 📂 Smart Data Upload
- Drag-and-drop CSV upload (up to 200MB)
- Automatic data quality assessment
- Real-time missing value analysis
- Quality scoring system (0-100)

### ⚙️ Intelligent Preprocessing
- Automatic duplicate removal
- Smart missing value imputation (median/mode)
- IQR-based outlier detection with Winsorization
- Label encoding & standardization
- Download cleaned datasets

### 🧪 Comprehensive Statistical Testing
| Test | Description |
|------|-------------|
| Shapiro-Wilk / KS | Normality testing |
| Pearson & Spearman | Correlation analysis |
| Chi-Square | Categorical associations |
| Levene's Test | Variance equality |
| One-Way ANOVA | Group comparisons |
| Mann-Whitney U | Non-parametric testing |
| VIF Analysis | Multicollinearity detection |

### 📈 Professional Visualizations
- Distribution analysis (Histograms + KDE)
- Box plots with outlier detection
- Correlation heatmaps (Pearson & Spearman)
- Pairwise relationship plots
- Q-Q plots for normality assessment
- VIF bar charts with risk indicators
- Missing value matrices

### 💡 AI-Powered Insights
- Automated pattern detection
- Data quality warnings
- Statistical significance highlights
- Actionable recommendations
- Readiness scoring for advanced analysis

### 📄 PDF Report Generation
- Executive summary with key metrics
- Complete statistical test results
- High-resolution visualizations
- Professional navy/gold styling
- Confidentiality watermarks
- Page numbers and branding

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/statspro.git
cd statspro