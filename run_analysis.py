"""
The Urban Pulse — Milestone 2
Reproducible analysis script for Docker container.
Generates all visualizations and saves them to /output folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("The Urban Pulse — Milestone 2 Analysis")
print("=" * 50)

# ── 1. Generate Synthetic Data ──────────────────────
print("\n[1/5] Generating dataset...")
np.random.seed(42)
N = 800

BOROUGH_PROFILES = {
    "Manhattan": {"income": 95000, "poverty": 0.14, "public_asst": 0.12, "bach": 0.60, "density": 70000},
    "Brooklyn":  {"income": 62000, "poverty": 0.22, "public_asst": 0.18, "bach": 0.38, "density": 36000},
    "Queens":    {"income": 67000, "poverty": 0.14, "public_asst": 0.11, "bach": 0.33, "density": 21000},
    "Bronx":     {"income": 38000, "poverty": 0.30, "public_asst": 0.28, "bach": 0.17, "density": 32000},
    "Staten Isl":{"income": 78000, "poverty": 0.12, "public_asst": 0.09, "bach": 0.33, "density": 8000},
}

rows = []
for _ in range(N):
    borough = np.random.choice(list(BOROUGH_PROFILES.keys()), p=[0.20, 0.30, 0.25, 0.18, 0.07])
    p = BOROUGH_PROFILES[borough]
    income   = max(20000, np.random.normal(p["income"], p["income"]*0.15))
    poverty  = max(0, min(0.6, np.random.normal(p["poverty"], 0.05)))
    pub_asst = max(0, min(0.6, np.random.normal(p["public_asst"], 0.04)))
    bach     = max(0, min(1.0, np.random.normal(p["bach"], 0.08)))
    density  = max(1000, np.random.normal(p["density"], p["density"]*0.2))
    sqft     = max(200, np.random.normal(900, 350))
    year     = int(np.random.normal(1965, 30))
    base     = 300 + (sqft * 0.15) + ((2024 - year) * -0.8) + np.random.normal(0, 80)
    if borough == "Manhattan": base *= 1.8
    elif borough == "Queens":  base *= 1.1
    price_sqft = max(50, base)
    rows.append({
        "borough": borough, "median_household_income": income,
        "percent_below_poverty": poverty, "percent_public_assistance": pub_asst,
        "percent_bach_or_higher": bach, "population_density_sqmi": density,
        "gross_square_feet": sqft, "year_built": year, "price_per_sqft": price_sqft,
    })

df = pd.DataFrame(rows)
median_price = df["price_per_sqft"].median()
df["affordable"] = (df["price_per_sqft"] <= median_price).astype(int)
print(f"   Dataset: {len(df)} records | NYC median price/sqft: ${median_price:.0f}")

FEATURES = ["median_household_income","percent_below_poverty","percent_public_assistance",
            "percent_bach_or_higher","population_density_sqmi","gross_square_feet","year_built"]

# ── 2. EDA ───────────────────────────────────────────
print("\n[2/5] Generating EDA plots...")

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("The Urban Pulse — Feature Distributions (KDE + Histogram)", fontsize=14, fontweight='bold')
cols = ["price_per_sqft","median_household_income","percent_below_poverty",
        "population_density_sqmi","gross_square_feet","year_built"]
for ax, col in zip(axes.flat, cols):
    skew = df[col].skew()
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue", alpha=0.5)
    ax.set_title(col.replace("_"," ").title())
    ax.text(0.97, 0.95, f"skew = {skew:.2f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 01_feature_distributions.png")

# Correlation heatmap
heat_cols = ["price_per_sqft","median_household_income","percent_below_poverty",
             "population_density_sqmi","percent_public_assistance","percent_bach_or_higher",
             "gross_square_feet","year_built"]
corr = df[heat_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix — Socioeconomic & Property Features", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 02_correlation_heatmap.png")

# Bivariate analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("The Urban Pulse — Bivariate Analysis", fontsize=13, fontweight='bold')
order = df.groupby("borough")["price_per_sqft"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="borough", y="price_per_sqft", order=order, ax=axes[0], palette="Set2")
axes[0].set_title("Price/Sqft by Borough"); axes[0].set_xlabel(""); axes[0].set_ylabel("Price / Sq Ft ($)")
r, p = stats.pearsonr(df["median_household_income"], df["price_per_sqft"])
axes[1].scatter(df["median_household_income"], df["price_per_sqft"], alpha=0.3, color="teal", s=15)
m, b = np.polyfit(df["median_household_income"], df["price_per_sqft"], 1)
x_line = np.linspace(df["median_household_income"].min(), df["median_household_income"].max(), 100)
axes[1].plot(x_line, m*x_line+b, color="red", lw=2, label=f"r = {r:.3f}  (p < 0.001)")
axes[1].set_title("Income vs Price/Sqft"); axes[1].set_xlabel("Median Household Income ($)"); axes[1].set_ylabel("Price / Sq Ft ($)")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_bivariate_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 03_bivariate_analysis.png")

# ── 3. Hypothesis Test ───────────────────────────────
print("\n[3/5] Running hypothesis test...")
r_val, p_val = stats.pearsonr(df["median_household_income"], df["price_per_sqft"])
low  = df[df["median_household_income"] <  df["median_household_income"].median()]["price_per_sqft"]
high = df[df["median_household_income"] >= df["median_household_income"].median()]["price_per_sqft"]
t_stat, t_p = stats.ttest_ind(high, low, alternative='greater')
print(f"   Pearson r={r_val:.3f}, p={p_val:.3e}")
print(f"   T-test t={t_stat:.2f}, p={t_p:.3e} -> {'REJECT H0' if t_p < 0.05 else 'FAIL TO REJECT H0'} (alpha=0.05)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hypothesis Test — Income vs. Residential Price/Sqft", fontsize=13, fontweight='bold')
zip_df = df.groupby(df["median_household_income"].round(-3)).agg(avg_price=("price_per_sqft","mean")).reset_index()
axes[0].scatter(zip_df["median_household_income"], zip_df["avg_price"], alpha=0.6, color="teal", s=30)
m2,b2 = np.polyfit(zip_df["median_household_income"], zip_df["avg_price"], 1)
xl = np.linspace(zip_df["median_household_income"].min(), zip_df["median_household_income"].max(), 100)
axes[0].plot(xl, m2*xl+b2, color="red", lw=2)
axes[0].set_title(f"Pearson Correlation: r = {r_val:.3f}"); axes[0].set_xlabel("ZIP-Level Median Income ($)"); axes[0].set_ylabel("ZIP-Level Avg Price / Sq Ft ($)")
axes[1].set_title(f"t = {t_stat:.2f}  |  p = {t_p:.1e}\n-> {'REJECT H0 (p < 0.05)' if t_p < 0.05 else 'FAIL TO REJECT H0'}")
data_box = [low.values, high.values]
axes[1].boxplot(data_box, labels=["Low-Income ZIPs\n(< median income)", "High-Income ZIPs\n(>= median income)"])
axes[1].set_ylabel("Avg Price / Sq Ft ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_hypothesis_test.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 04_hypothesis_test.png")

# ── 4. Model Training ────────────────────────────────
print("\n[4/5] Training models...")
X = df[FEATURES]
y = df["affordable"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc').mean()
print(f"   Logistic Regression AUC: {lr_auc:.3f}")
print(f"   Random Forest AUC (test): {rf_auc:.3f} | CV: {rf_cv:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Evaluation — Logistic Regression vs. Random Forest", fontsize=13, fontweight='bold')
for model, name, color in [(lr,"Logistic Reg",  "green"), (rf,"Random Forest","red")]:
    proba = model.predict_proba(X_test_s if model==lr else X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0,0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
axes[0,0].plot([0,1],[0,1],'k--', label="Random baseline"); axes[0,0].legend(); axes[0,0].set_title("ROC Curves")
axes[0,0].set_xlabel("False Positive Rate"); axes[0,0].set_ylabel("True Positive Rate")
cm = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=["Affordable","Unaffordable"], yticklabels=["Affordable","Unaffordable"])
axes[0,1].set_title("Confusion Matrix — Random Forest"); axes[0,1].set_xlabel("Predicted label"); axes[0,1].set_ylabel("True label")
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors_fi = ["red" if fi.index[-1] == f else "teal" for f in fi.index]
fi.plot(kind='barh', ax=axes[1,0], color=colors_fi)
for i, v in enumerate(fi.values): axes[1,0].text(v+0.002, i, f"{v:.3f}", va='center', fontsize=8)
axes[1,0].set_title("Random Forest — Feature Importance"); axes[1,0].set_xlabel("Gini Importance")
x_pos = [0, 1]; auc_vals = [lr_auc, rf_auc]; cv_vals = [None, rf_cv]
axes[1,1].bar([0-0.15, 1-0.15], [lr_auc, rf_auc], width=0.3, label="Test AUC", color="teal")
axes[1,1].bar([1+0.15], [rf_cv], width=0.3, label="CV AUC (5-fold)", color="goldenrod")
axes[1,1].axhline(0.75, color='red', linestyle='--', label="Target AUC = 0.75")
for i, v in enumerate([lr_auc, rf_auc]): axes[1,1].text(i-0.15, v+0.005, f"{v:.3f}", ha='center', fontsize=9)
axes[1,1].text(1+0.15, rf_cv+0.005, f"{rf_cv:.3f}", ha='center', fontsize=9)
axes[1,1].set_xticks([0,1]); axes[1,1].set_xticklabels(["Logistic\nRegression","Random Forest"])
axes[1,1].set_ylim(0.4, 1.0); axes[1,1].set_title("Model AUC — Test vs. Cross-Validation"); axes[1,1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_model_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 05_model_evaluation.png")

# ── 5. Done ──────────────────────────────────────────
print("\n[5/5] Complete!")
print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
print("Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
