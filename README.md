# credit-risk-tree-models
Credit risk (good/bad) prediction using logistic regression, decision trees, random forest, and gradient boosting.
## Project Overview

This project explores **credit risk classification** on the classic German credit dataset
(Statlog German Credit, numeric version).  
The goal is to predict whether a customer is a **good** or **bad** credit risk.

- Rows: 1,000 customers
- Features: 24 numeric attributes (anonymised as `feat_1` ... `feat_24`)
- Target: `BadCredit` (0 = good credit, 1 = bad credit)

## Repository structure

- `data/`
  - `german.data`, `german.data-numeric`: raw UCI files
  - `german_credit_numeric_clean.csv`: cleaned dataset used for modelling
- `notebooks/`
  - `01_credit_eda_and_cleaning.ipynb`: load numeric data, assign column names, create `BadCredit` target, save clean CSV
  - `02_credit_baseline_models.ipynb`: train/test split, logistic regression, decision tree, random forest, and gradient boosting
- `src/`: (empty for now, reserved for future helper code)
- `reports/`: (reserved for future figures or markdown summaries)

## Models and results (test set)

All models use the same 80/20 train–test split (stratified).  
Target: `BadCredit` (0 = good, 1 = bad).  
Test set: 200 customers (140 good, 60 bad).

| Model                      | Accuracy | Precision (bad=1) | Recall (bad=1) | F1 (bad=1) |
|---------------------------|----------|--------------------|----------------|------------|
| Logistic Regression       | 0.770    | 0.667              | 0.467          | 0.549      |
| Decision Tree             | 0.720    | 0.531              | 0.567          | 0.548      |
| Random Forest (200 trees) | 0.755    | 0.580              | 0.667          | 0.620      |
| Gradient Boosting         | 0.780    | 0.660              | 0.550          | 0.600      |

**Key points:**

- Logistic regression is a simple **baseline**, better than predicting all customers as good, but it misses more than half of the bad credit cases.
- A single decision tree captures some non-linear patterns but is less stable and has lower overall accuracy.
- **Random Forest** improves recall for bad credit to about **67%**, with the best F1-score for the bad class, making it a strong choice when the priority is detecting risky customers.
- **Gradient Boosting** achieves the **highest accuracy (0.78)** and the best precision for bad credit, at the cost of slightly lower recall than the Random Forest.

## What I practised in this project

- Loading and cleaning a classic credit risk dataset.
- Creating a binary target (`BadCredit`) from the original label.
- Building and evaluating:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Interpreting accuracy, precision, recall, F1, and confusion matrices, especially for an **imbalanced** classification problem.
- Comparing linear vs tree-based vs ensemble models on the same dataset.
