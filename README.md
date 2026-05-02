# 🏙️ The Urban Pulse — CS 301 Spring 2026

**Predicting NYC Housing Affordability from Socioeconomic Data**

**Team:** Michael Hanson (Lead Analyst) · Steven Martinez (Co-Analyst)

---

## Project Overview

This project leverages two open NYC datasets to answer:
> *What socioeconomic and geographic factors most strongly predict residential property sale prices across NYC neighborhoods — and can these factors reliably classify a neighborhood as affordable vs. unaffordable for median-income households?*

---

## Repository Structure

```
urban-pulse/
├── Urban_Pulse_Milestone2.ipynb   # Main Colab notebook (run top-to-bottom)
├── README.md                      # This file
```

---

## How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. File → Upload notebook → select `Urban_Pulse_Milestone2.ipynb`
3. Runtime → Run all
4. No local data files needed — the notebook downloads directly from NYC Open Data APIs

---

## Data Sources

| Dataset | Source | URL |
|---------|--------|-----|
| NYC Citywide Annualized Property Sales | NYC Dept. of Finance / NYC Open Data | https://data.cityofnewyork.us/resource/w2pb-icbu.csv |
| Demographic Statistics By ZIP Code | NYC Dept. of City Planning / NYC Open Data | https://data.cityofnewyork.us/resource/kku6-nxdu.csv |

---

## Methodology

1. **EDA** — KDE/histogram distributions, Pearson correlation heatmap, bivariate scatter/box plots
2. **Hypothesis Testing** — Pearson correlation + one-tailed independent t-test (α = 0.05)
3. **Models** — Logistic Regression + Random Forest classifier (target: AUC > 0.75)
4. **Evaluation** — AUC-ROC, 5-fold CV, Confusion Matrix, Feature Importance

---

## Key Results

- Rejected H₀: Pearson r = 0.78, p < 0.001
- Random Forest AUC = 0.978 (exceeds 0.75 target)
- Top predictor: `median_household_income`
- Actionable insight: Pre-emptive zoning in boundary ZIPs before affordability tipping point

---

## Dependencies

All pre-installed in Google Colab:
`pandas · numpy · matplotlib · seaborn · scipy · scikit-learn · requests`
