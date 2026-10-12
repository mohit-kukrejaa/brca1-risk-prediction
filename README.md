# BRCA1 Mutation Risk Prediction

A machine learning pipeline for predicting BRCA1 variant pathogenicity using structural, positional, biochemical, and sequence-derived features.

The project compares Logistic Regression, Random Forest, and XGBoost models using grouped cross-validation and evaluates the final selected model on a held-out test set.

---

## Project Structure

```text
brca1-risk-prediction/
├── config/
│   ├── config.toml
│   └── model_config.toml
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       ├── brca1_features.csv
│       └── split_data.csv
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_eda.ipynb
│   └── 03_feature_engineering.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── data/
│   │   └── model/
│   ├── models/
│   │   ├── logistic_regression/
│   │   ├── random_forest/
│   │   └── xgboost/
│   ├── tables/
│   ├── test/
│   └── error_analysis/
│
├── report/
│   └── Summary_report.md
│
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── split_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── make_plots.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run all commands from the project root directory.

### 1. Prepare the Data

```bash
python src/data_prep.py
```
Loads the raw dataset, prepares labels, validates the data, and creates engineered features.

**Output:**

`data/processed/brca1_features.csv`

### 2. Create Train / Validation / Test Splits

```bash
python src/split_data.py
```

Creates approximately 70/15/15 train, validation, and test splits grouped by protein_position.

**Output:**

`data/processed/split_data.csv`

### 3. Train and Compare Models

```bash
python src/train.py
```

Trains and compares Logistic Regression, Random Forest, and XGBoost using grouped cross-validation and multiple feature configurations.

**Main outputs are saved under:**

- `outputs/models/`
- `outputs/tables/`

### 4. Evaluate the Final Model

```bash
python src/evaluate.py
```

Selects the final model, chooses the validation threshold, evaluates the held-out test set, and generates performance and explainability outputs.

**Final test outputs are saved under:**

`outputs/test/`

### 5. Run Error Analysis

```bash
python src/error_analysis.py
```

Analyzes false positives, false negatives, prediction confidence, domain-level errors, and representative variants.

**Outputs are saved under:**

`outputs/error_analysis/`

## Full Run Order

For a complete run from raw data to final evaluation:

```bash
python src/data_prep.py
python src/split_data.py
python src/train.py
python src/evaluate.py
python src/error_analysis.py
```

## Configuration

Project paths and feature definitions are stored in:

[config/config.toml](config/config.toml)

Model search spaces, cross-validation settings, and threshold-selection settings are stored in:

[config/model_config.toml](config/model_config.toml)

Random seed 42 is used where applicable for reproducibility.

## Final Model

The final selected model was Logistic Regression.

Held-out test performance:

| Metric | Value |
| --- | ---: |
| Average Precision | 0.693 |
| ROC-AUC | 0.788 |
| Precision | 0.725 |
| Recall / Sensitivity | 0.556 |
| Specificity | 0.870 |
| F1 Score | 0.629 |
| Balanced Accuracy | 0.713 |
| Brier Score | 0.182 |

The final classification threshold was selected on the validation set and fixed at 0.52 before test evaluation.

## Report

Analysis and summary report:

[report/Summary_report.md](report/Summary_report.md)

