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

A data mining pipeline , like the full set up, uses telemarketing campaign data from a Portuguese bank to predict whether people will subscribe to term deposits. In this research, the complete workflow is put to use, from preprocessing to model training , with multiple techniques, plus exploratory data analysis that tries to understand patterns. Then there is the careful evaluation part with five performance measures, so the results are not just “guessed” but actually checked.

### Business Problem

Is it possible to forecast which bank customers are most likely to sign up for a term deposit, allowing for more effective marketing resource targeting?

### Key Features

-So  the entire data mining pipeline goes kinda like this EDA → Preprocessing → Training → Evaluation → Suggestions, and yep it continues. For classification, Logistic regression, decision trees, random forests, gradient boosting, and KNN are the five classification algorithms. 

For evaluation there are 5 metrics, well more or less: F1-Score, ROC-AUC, Accuracy, Precision, and Recall. And there are twelve visualizations such as Demographics, correlations, seasonal patterns, model comparison, ROC curves, and feature importance. 

Then, there’s strong performance estimation using 5-Fold Stratified Cross-Validation , it feels solid. Finally, the model results provide actionable information, so it’s not just numbers.

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
├── README.md                   # This file
├── charts/                     # Generated charts & CSVs
└── .gitignore
```

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). Bank Marketing Dataset. UCI ML Repository.
- Dataset URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Scikit-learn Documentation: https://scikit-learn.org/stable/
