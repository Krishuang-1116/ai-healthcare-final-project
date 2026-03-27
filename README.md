# AI in Healthcare Course Final Project  
## Mortality Prediction in ICU Pneumonia Patients

This project investigates the performance of different machine learning approaches for predicting in-hospital mortality among ICU patients diagnosed with pneumonia, using structured clinical data from the first 24 hours of ICU admission.

We compare classical statistical models, tree-based methods, interpretable additive models, deep learning, and foundation-style tabular models under a unified experimental pipeline.

---

## Problem Setup

- **Task**: Binary classification (mortality prediction)
- **Unit of analysis**: ICU stay
- **Features**: Aggregated first-day clinical variables (vitals, labs, blood gas, SOFA)
- **Target**: In-hospital mortality

---

## Models Evaluated

- Logistic Regression (baseline clinical model)
- XGBoost (tree-based ensemble)
- Explainable Boosting Machine (EBM / GA2M)
- Multilayer Perceptron (MLP)
- TabICL (foundation-style tabular model)

---

## Methodology

- **5-fold stratified cross-validation**
- **Hyperparameter tuning via GridSearchCV (within training folds)**
- **Evaluation metrics**:
  - AUROC
  - AUPRC
  - Brier Score (calibration)

- **Model-specific preprocessing**:
  - Logistic Regression / MLP → imputation + scaling
  - XGBoost → native missing value handling
  - EBM → missing values as separate bins
  - TabICL → raw features (no preprocessing)

---

## Repository Structure
```
├── src/
│ ├── config.py # Global configuration (random seeds, constants)
│ ├── cv_runner.py # Cross-validation pipeline (run_cv_model)
│ ├── data_utils.py # Data loading and preprocessing utilities
│ ├── metrics_utils.py # Evaluation metrics (AUROC, AUPRC, Brier)
│ ├── results_utils.py # Plotting (ROC, PR, calibration curves)
│ ├── models_classical.py # Logistic Regression, XGBoost
│ ├── models_deep.py # MLP and TabICL
│ └── .DS_Store # System file (can be ignored)
│
├── .gitignore
├── requirements.txt
```
---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run experiments

Use the main pipeline function:
```bash
from src.cv_runner import run_cv_model
from src.models_classical import fit_predict_logistic, fit_predict_xgboost
```
Each model follows the same interface:
```bash
metrics_df, preds_df = run_cv_model(
    X=X,
    y=y,
    ids=ids,
    model_name="XGBoost",
    fit_predict_fn=fit_predict_xgboost
)
```
### 3. Outputs

The pipeline generates:

* Fold-level metrics
* Predicted probabilities
* Aggregated summary metrics

These outputs can be used to generate:

* ROC curves
* Precision-recall curves
* Calibration plots
* Model comparison tables

--- 

## Key Findings
* TabICL achieves the best overall performance across AUROC and AUPRC
* XGBoost and EBM perform competitively, with strong calibration
* Logistic Regression remains a strong baseline, but underperforms on nonlinear patterns
* MLP does not outperform classical tabular models, consistent with prior literature

--- 

## Data Availability

Due to data governance constraints of the MIMIC-IV database, raw data cannot be shared in this repository.

The cohort is constructed using SQL queries (see report appendix), and all experiments are conducted on first-day ICU features to ensure reproducibility and prevent data leakage.

--- 

## References
* Johnson et al., MIMIC-IV database
* Caruana et al., 2015 (GA2M / EBM)
* TabICL paper (foundation models for tabular data)

--- 
## Authors
Kris Huang
Junyang Zhao


