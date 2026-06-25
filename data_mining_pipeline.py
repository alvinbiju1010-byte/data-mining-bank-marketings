"""
Data Mining —  Project
Student: Alvin Biju (ID: GH1029339)
Dataset: Bank Marketing Dataset (UCI Machine Learning Repository)

Complete Data Mining Pipeline:
  1. Data Loading & Understanding
  2. Exploratory Data Analysis (EDA)
  3. Data Preprocessing & Feature Engineering
  4. Model Training (Multiple Algorithms)
  5. Model Evaluation & Comparison
  6. Results & Business Recommendations

Business Problem: Predict whether a bank client will subscribe to a term deposit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Style Configuration 
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
})

OUTPUT_DIR = "./charts/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("Bank Marketing Data Mining Pipeline — Alvin Biju (GH1029339)")
print("=" * 65)


# STAGE 1: DATA LOADING & INITIAL UNDERSTANDING

print("\n[1/5] Loading Bank Marketing Dataset...")

# Dataset: Bank Marketing from UCI ML Repository
# Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
# Direct download URL for bank-additional-full.csv
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"

try:
    df = pd.read_csv(DATA_URL, sep=";")
    print(f"✓ Downloaded from UCI repository")
except Exception:
    print("⚠ Could not download. Using local copy if available...")
    df = pd.read_csv("./data/bank-additional-full.csv", sep=";")

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nColumn info:")
print(df.info())
print(f"\nTarget variable 'y' distribution:")
print(df["y"].value_counts())
print(f"  Subscription rate: {df['y'].value_counts(normalize=True)['yes']*100:.1f}%")


df.describe().to_csv(f"{OUTPUT_DIR}descriptive_stats.csv")


# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)

print("\n[2/5] Exploratory Data Analysis...")

# --- Chart 1: Target Variable Distribution ---
fig, ax = plt.subplots(figsize=(8, 6))
counts = df["y"].value_counts()
colors = ["#EF5350", "#4CAF50"]
bars = ax.bar(["No (Did Not Subscribe)", "Yes (Subscribed)"], counts.values, color=colors)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,} ({val/len(df)*100:.1f}%)", ha="center", fontweight="bold", fontsize=12)
ax.set_title("Target Variable: Term Deposit Subscription", fontweight="bold")
ax.set_ylabel("Number of Clients")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}01_target_distribution.png")
plt.close()
print("✓ Chart 1: Target distribution")

# --- Chart 2: Age Distribution by Subscription ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=df, x="age", hue="y", bins=40, kde=True, ax=ax,
             palette={"no": "#EF5350", "yes": "#4CAF50"}, alpha=0.5)
ax.set_title("Age Distribution by Subscription Status", fontweight="bold")
ax.set_xlabel("Age")
ax.legend(["Did Not Subscribe", "Subscribed"])
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}02_age_distribution.png")
plt.close()
print("✓ Chart 2: Age distribution")

# --- Chart 3: Job Type vs Subscription Rate ---
fig, ax = plt.subplots(figsize=(12, 6))
job_sub = df.groupby("job")["y"].apply(lambda x: (x == "yes").mean() * 100).sort_values()
colors = ["#EF5350" if v < 15 else "#4CAF50" for v in job_sub.values]
bars = ax.barh(job_sub.index, job_sub.values, color=colors)
for bar, val in zip(bars, job_sub.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_title("Subscription Rate by Job Category", fontweight="bold")
ax.set_xlabel("Subscription Rate (%)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}03_job_subscription_rate.png")
plt.close()
print("✓ Chart 3: Job vs subscription")

# --- Chart 4: Education Level vs Subscription ---
fig, ax = plt.subplots(figsize=(10, 6))
edu_sub = df.groupby("education")["y"].apply(lambda x: (x == "yes").mean() * 100).sort_values()
colors = ["#EF5350" if v < 15 else "#4CAF50" for v in edu_sub.values]
bars = ax.bar(edu_sub.index, edu_sub.values, color=colors)
for bar, val in zip(bars, edu_sub.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Subscription Rate by Education Level", fontweight="bold")
ax.set_ylabel("Subscription Rate (%)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}04_education_subscription.png")
plt.close()
print("✓ Chart 4: Education vs subscription")

# --- Chart 5: Correlation Matrix (Numeric Features) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "y" not in numeric_cols:
    numeric_cols_for_corr = numeric_cols
else:
    numeric_cols_for_corr = numeric_cols

fig, ax = plt.subplots(figsize=(14, 12))
corr = df[numeric_cols_for_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Numeric Features", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}05_correlation_matrix.png")
plt.close()
print("✓ Chart 5: Correlation matrix")

# --- Chart 6: Previous Contact vs Subscription ---
fig, ax = plt.subplots(figsize=(10, 6))
contact_data = df.groupby(["previous", "y"]).size().unstack(fill_value=0)
# Convert to % of each previous category
contact_pct = contact_data.div(contact_data.sum(axis=1), axis=0) * 100
contact_pct["yes"].plot(kind="bar", ax=ax, color="#4CAF50")
ax.set_title("Subscription Rate by Number of Previous Contacts", fontweight="bold")
ax.set_xlabel("Number of Previous Contacts")
ax.set_ylabel("Subscription Rate (%)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}06_previous_contacts.png")
plt.close()
print("✓ Chart 6: Previous contacts")

# --- Chart 7: Month (Seasonality) ---
fig, ax = plt.subplots(figsize=(12, 6))
month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
month_sub = df.groupby("month", observed=False)["y"].apply(lambda x: (x == "yes").mean() * 100)
sns.lineplot(x=month_sub.index, y=month_sub.values, ax=ax, marker="o",
             markersize=10, linewidth=2.5, color="#2196F3")
ax.set_title("Subscription Rate by Month (Seasonal Trend)", fontweight="bold")
ax.set_ylabel("Subscription Rate (%)")
ax.set_xlabel("Month")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}07_seasonal_trend.png")
plt.close()
print("✓ Chart 7: Seasonal trend")


# STAGE 3: DATA PREPROCESSING & FEATURE ENGINEERING

print("\n[3/5] Data Preprocessing & Feature Engineering...")

# Create a copy for preprocessing
df_clean = df.copy()

# Handle missing/unknown values
# Replace "unknown" strings with NaN for proper imputation
for col in df_clean.select_dtypes(include=["object"]).columns:
    if col != "y":
        df_clean[col] = df_clean[col].replace("unknown", np.nan)
        df_clean[col] = df_clean[col].replace("nonexistent", np.nan)

# Report missing values
missing = df_clean.isnull().sum()
missing_pct = (missing / len(df_clean) * 100).round(2)
missing_report = pd.DataFrame({"Missing": missing, "Pct": missing_pct})
missing_report = missing_report[missing_report["Missing"] > 0]
if len(missing_report) > 0:
    print(f"\nFeatures with missing values:")
    print(missing_report.to_string())

# Encode target variable
df_clean["y_encoded"] = (df_clean["y"] == "yes").astype(int)

# Identify column types
categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("y")  # Remove target

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if "y_encoded" in numeric_cols:
    numeric_cols.remove("y_encoded")

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# Prepare features and target
X = df_clean[numeric_cols + categorical_cols]
y = df_clean["y_encoded"]

# Train/test split (stratified — important for imbalanced data!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"Training subscription rate: {y_train.mean()*100:.1f}%")
print(f"Test subscription rate:     {y_test.mean()*100:.1f}%")

# Fit preprocessor on training data
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
all_feature_names = list(numeric_cols) + list(cat_feature_names)
print(f"\nTotal features after preprocessing: {len(all_feature_names)}")


# STAGE 4: MODEL TRAINING — Multiple Algorithms

print("\n[4/5] Training Multiple Models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15,
                                              random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,
                                                      max_depth=5, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_processed, y_train,
                                 cv=cv, scoring="roc_auc")
    print(f"    CV ROC-AUC (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full training set
    model.fit(X_train_processed, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "y_proba": y_proba,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"    ROC-AUC:   {roc_auc:.4f}")


# STAGE 5: MODEL EVALUATION & COMPARISON

print("\n[5/5] Evaluating & Comparing Models...")

# --- Chart 8: Metrics Comparison Bar Chart ---
metrics_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1-Score": res["f1_score"],
        "ROC-AUC": res["roc_auc"] if res["roc_auc"] else 0,
    }
    for name, res in results.items()
])

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(metrics_df))
width = 0.15
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, metrics_df[metric], width, label=metric, color=color)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=7, fontweight="bold", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metrics_df["Model"], fontsize=10)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Evaluation Metrics", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}08_model_comparison.png")
plt.close()
print("✓ Chart 8: Model comparison")

# --- Chart 9: ROC Curves — All Models ---
fig, ax = plt.subplots(figsize=(10, 8))
for name, res in results.items():
    if res["y_proba"] is not None:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}09_roc_curves.png")
plt.close()
print("✓ Chart 9: ROC curves")

# --- Chart 10: Confusion Matrix — Best Model ---
best_model_name = max(results, key=lambda n: results[n]["f1_score"])
best_result = results[best_model_name]
cm = confusion_matrix(y_test, best_result["model"].predict(X_test_processed))

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["No (0)", "Yes (1)"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"Confusion Matrix — {best_model_name}", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}10_confusion_matrix.png")
plt.close()
print(f"✓ Chart 10: Confusion Matrix ({best_model_name})")

# --- Chart 11: Feature Importance (Random Forest) ---
if "Random Forest" in results:
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices],
            color=sns.color_palette("viridis", len(indices)))
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([all_feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances — Random Forest", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}11_feature_importance.png")
    plt.close()
    print("✓ Chart 11: Feature importance")

# --- Chart 12: Cross-Validation Comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame([
    {"Model": name, "CV Mean": res["cv_mean"], "CV Std": res["cv_std"]}
    for name, res in results.items()
])
bars = ax.bar(cv_data["Model"], cv_data["CV Mean"],
              yerr=cv_data["CV Std"], capsize=8,
              color=sns.color_palette("Set2", len(cv_data)))
for bar, val in zip(bars, cv_data["CV Mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
ax.set_title("5-Fold Cross-Validation ROC-AUC Scores", fontweight="bold")
ax.set_ylabel("ROC-AUC Score")
ax.set_ylim(0.7, 1.0)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}12_cross_validation.png")
plt.close()
print("✓ Chart 12: Cross-validation comparison")

# ── FINAL SUMMARY 
print("\n" + "=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 75)
for name, res in results.items():
    roc_str = f"{res['roc_auc']:.4f}" if res['roc_auc'] else "N/A"
    print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>10.4f} {res['f1_score']:>10.4f} {roc_str:>10}")

print(f"\n★ Best model by F1-Score: {best_model_name}")
print(f"  Accuracy:  {best_result['accuracy']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  F1-Score:  {best_result['f1_score']:.4f}")
print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")

# Save detailed results
summary_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": f"{res['accuracy']:.4f}",
        "Precision": f"{res['precision']:.4f}",
        "Recall": f"{res['recall']:.4f}",
        "F1_Score": f"{res['f1_score']:.4f}",
        "ROC_AUC": f"{res['roc_auc']:.4f}" if res['roc_auc'] else "N/A",
        "CV_ROC_AUC_Mean": f"{res['cv_mean']:.4f}",
        "CV_ROC_AUC_Std": f"{res['cv_std']:.4f}",
    }
    for name, res in results.items()
])
summary_df.to_csv(f"{OUTPUT_DIR}model_comparison_summary.csv", index=False)

print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print(f"✓ Model comparison CSV: {OUTPUT_DIR}model_comparison_summary.csv")
print("\n" + "=" * 65)
print("DATA MINING PIPELINE COMPLETE!")
print("=" * 65)
