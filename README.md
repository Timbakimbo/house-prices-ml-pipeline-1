# house-prices-ml-pipeline-1
End-to-end machine learning pipeline using scikit-learn

## Problem
Predict house sale prices based on numerical and categorical of a kaggle dataset

## Pipeline
- Numerical: Median imputation
- Categorical:
  - Low cardinality → OneHotEncoder
  - High cardinality → OrdinalEncoder (unknown = -1)

## Models
- Random Forest (baseline)
- Random Forest + preprocessing pipeline
- HistGradientBoostingRegressor
  - log-transformed target
  - 5-fold cross-validation

## Hyperparameter Tuning
Three-stage sweep:
1. Global sweep (lr, depth, leaf)
2. Local sweep (depth=4, leaf=15)
3. Fine sweep (lr, max_iter)

Best config:
- max_depth = 4  
- min_samples_leaf = 15  
- learning_rate ≈ 0.03  
- max_iter ≈ 1000

## How to run
```bash
pip install -r requirements.txt
python -m src.train

