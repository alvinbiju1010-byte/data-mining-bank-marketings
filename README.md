# Bank Marketing — Term Deposit Prediction

**Data Mining Individual Project — Classification & Model Evaluation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)](https://scikit-learn.org)

**Student:** Alvin Biju  
**Student ID:** GH1029339  
**Course:** Data Mining — Individual Project  
**Dataset:** [Bank Marketing (UCI ML Repository)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## Project Overview

A complete data mining pipeline for predicting term deposit subscriptions from a Portuguese bank's telemarketing campaign data. The project applies the full data mining workflow: exploratory data analysis, preprocessing, model training with multiple algorithms, and rigorous evaluation using five performance metrics.

### Business Problem

*Can we predict which bank clients are most likely to subscribe to a term deposit, enabling more efficient targeting of marketing resources?*

### Key Features

- **Complete Data Mining Pipeline:** EDA → Preprocessing → Training → Evaluation → Recommendations
- **5 Classification Algorithms:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN
- **5 Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **12 Visualizations:** Demographics, correlations, seasonal trends, model comparison, ROC curves, feature importance
- **5-Fold Stratified Cross-Validation:** Robust performance estimation
- **Business Recommendations:** Actionable insights derived from model results

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Pipeline

```bash
python data_mining_pipeline.py
```

The script automatically downloads the dataset from UCI, runs the full pipeline, and generates all charts.

### 3. View Results

Charts and CSV results are saved to the `./charts/` directory:
- `01_target_distribution.png` — Class imbalance visualization
- `02_age_distribution.png` — Age vs subscription
- `03_job_subscription_rate.png` — Job category analysis
- `04_education_subscription.png` — Education level analysis
- `05_correlation_matrix.png` — Feature correlations
- `06_previous_contacts.png` — Contact history impact
- `07_seasonal_trend.png` — Monthly seasonality
- `08_model_comparison.png` — All metrics across all models
- `09_roc_curves.png` — ROC curves for all models
- `10_confusion_matrix.png` — Best model confusion matrix
- `11_feature_importance.png` — Top 15 predictive features
- `12_cross_validation.png` — CV performance comparison
- `model_comparison_summary.csv` — Full numerical results

---

## Pipeline Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   LOAD   │───▶│   EDA    │───▶│PREPROCESS│───▶│  TRAIN   │───▶│ EVALUATE │
│ 41K rows │    │ 7 charts │    │ Encode + │    │5 models  │    │5 metrics │
│ 20 feats │    │Insights  │    │ Scale    │    │CV + Test │    │12 charts │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.903 | 0.658 | 0.427 | 0.518 | 0.793 |
| Decision Tree | 0.886 | 0.563 | 0.524 | 0.543 | 0.722 |
| Random Forest | 0.907 | 0.654 | 0.488 | 0.559 | 0.795 |
| **Gradient Boosting** | **0.904** | **0.633** | **0.507** | **0.563** | **0.793** |
| K-Nearest Neighbors | 0.893 | 0.592 | 0.372 | 0.457 | 0.751 |

**Best Model: Gradient Boosting** (F1-Score = 0.563) — balances precision and recall for an imbalanced dataset.

---

## Key Findings

1. **Class imbalance:** Only 11.3% of clients subscribe — accuracy alone is misleading
2. **Top predictors:** Euribor rate, employment variation, age, previous campaign outcome, contact type
3. **Seasonality:** March–May have 2–3× higher subscription rates than Q4
4. **Financial health matters:** Clients without loans/defaults are ~2× more likely to subscribe
5. **Ensemble methods win:** Random Forest & Gradient Boosting significantly outperform single Decision Tree

---

## Project Structure

```
data-mining-bank-marketing/
├── data_mining_pipeline.py    # Complete pipeline (all 5 stages)
├── report.md                   # Project report (~2,800 words)
├── README.md                   # This file
├── charts/                     # Generated charts & CSVs
├── data/                       # Dataset (downloaded automatically or placed manually)
└── .gitignore
```

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). Bank Marketing Dataset. UCI ML Repository.
- Dataset URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Scikit-learn Documentation: https://scikit-learn.org/stable/
