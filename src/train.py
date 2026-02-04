import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split,KFold, cross_validate
from sklearn.metrics import root_mean_squared_error, mean_absolute_error 
from src.data_loading import load_raw_data
from src.pre_processing import (split_target_feature, split_num_cat, build_preprocessor)
from sklearn.pipeline import Pipeline


def train_baseline_rf_1(X, y, random_state=0):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)

    return model, rmse, mae

def train_rf_2(X, y, preprocessor, random_state=0):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)

    return pipe, rmse, mae

### XGBOOST MODELLE ###
def train_hgbr_cv_log_target(
    X,
    y,
    preprocessor,
    random_state=0,
    n_splits=5,
    learning_rate=0.1,
    max_iter=500,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0):

    base_model = HistGradientBoostingRegressor(
        random_state=random_state,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_model),])

    model = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        },
        n_jobs=-1)

    rmse_scores = -scores["test_rmse"]
    mae_scores = -scores["test_mae"]

    return (
        rmse_scores.mean(),
        rmse_scores.std(),
        mae_scores.mean(),
        mae_scores.std())

## Parametersweep ##
#phase1#
def phase1_hgbr_sweep(X, y, preprocessor):
    results = []

    for lr in [0.03, 0.05, 0.1]:
        for mi in [500, 1000, 2000]:
            for md in [None, 4, 6]:
                for msl in [20, 50]:

                    rmse_mean, rmse_std, mae_mean, mae_std = train_hgbr_cv_log_target(
                        X, y, preprocessor,
                        learning_rate=lr,
                        max_iter=mi,
                        max_depth=md,
                        min_samples_leaf=msl,
                        l2_regularization=0.0
                    )

                    results.append({
                        "learning_rate": lr,
                        "max_iter": mi,
                        "max_depth": md,
                        "min_samples_leaf": msl,
                        "rmse": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae": mae_mean
                    })

                    print(
                        f"lr={lr}, it={mi}, depth={md}, leaf={msl} "
                        f"→ RMSE={rmse_mean:.2f}"
                    )

    return sorted(results, key=lambda r: r["rmse"])
#phase2 sweep auf struktur#
#max depth3,4,5 und min samples leaf15,20,25#
def phase2_hgbr_sweep(X, y, preprocessor):
    results = []
    for md in [3,4]:
        for msl in [14,15,16, 17]:
            rmse_mean, rmse_std, mae_mean, mae_std = train_hgbr_cv_log_target(
                X, y, preprocessor,
                learning_rate=0.03,
                max_iter=1000,
                max_depth=md,
                min_samples_leaf=msl,
                l2_regularization=0.0
            )
            results.append((rmse_mean, rmse_std, mae_mean, mae_std, md, msl))
            print(f"depth={md}, leaf={msl} -> RMSE={rmse_mean:.2f} ± {rmse_std:.2f}")
    return sorted(results, key=lambda t: t[0])
#phase3 sweep auf lr und it#
def phase3_hgbr_sweep(X, y, preprocessor, best_depth=4, best_leaf=15):
    results = []
    for lr in [0.025, 0.03, 0.04, 0.05]:
        for mi in [800, 1000, 1500]:
            rmse_mean, rmse_std, mae_mean, mae_std = train_hgbr_cv_log_target(
                X, y, preprocessor,
                learning_rate=lr,
                max_iter=mi,
                max_depth=best_depth,
                min_samples_leaf=best_leaf,
                l2_regularization=0.0
            )
            results.append((rmse_mean, rmse_std, mae_mean, mae_std, lr, mi))
            print(f"lr={lr}, it={mi} -> RMSE={rmse_mean:.2f} ± {rmse_std:.2f}")
    return sorted(results, key=lambda t: t[0])


    
def main():
    # Load raw data
    train, test = load_raw_data('data/raw/train.csv', 'data/raw/test.csv')
    # Pre-process the data
    train = train.drop(columns=["Id"])
    test = test.drop(columns=["Id"])
    X, y = split_target_feature(train, target_col='SalePrice')
    num_cols, cat_cols = split_num_cat(X)
    X_num = X[num_cols]
    X_cat = X[cat_cols] 
    # Train a Baseline Model
    baseline_rf_model, baseline_rf_rmse, baseline_rf_mae = train_baseline_rf_1(X_num, y, random_state=0)

   # print(f'Baseline RF Validation RMSE: {baseline_rf_rmse}')
    print(f'Baseline RF Validation MAE: {baseline_rf_mae}')
    
    # Train second model with better preprocessing
    preprocessor = build_preprocessor(X,num_cols, cat_cols,card_threshold=10)
    pipe, rmse, mae = train_rf_2(X, y, preprocessor, random_state=0)
    print(f"Baseline RF RMSE: {rmse:.2f}")
    print(f"Baseline RF MAE : {mae:.2f}")
    # Train third model with cross-validation and log-transformed target and HGBR
    hgbr_rmse_mean, hgbr_rmse_std, hgbr_mae_mean, hgbr_mae_std = train_hgbr_cv_log_target(
        X, y, preprocessor, random_state=0, n_splits=5)
    print("HGBR + Preprocess + log1p(y) — 5-Fold CV")
    print(f"RMSE: {hgbr_rmse_mean:.2f} ± {hgbr_rmse_std:.2f}")
    print(f"MAE : {hgbr_mae_mean:.2f} ± {hgbr_mae_std:.2f}")
    # Parameter sweep phase 3
    #fixierte werteaus pahse 1 und 2
    best_depth = 4
    best_leaf = 15
    print("\nPHASE 2.2 — Local lr/iter sweep (depth=4, leaf=15)")
    phase22_results = phase3_hgbr_sweep(X, y, preprocessor, best_depth=best_depth, best_leaf=best_leaf)
    print("\nTop 5 configs (Phase 2.2):")
    for r in phase22_results[:5]:
        rmse_mean, rmse_std, mae_mean, mae_std, lr, mi = r
        print(
            f"RMSE={rmse_mean:.2f} ± {rmse_std:.2f} | "
            f"MAE={mae_mean:.2f} ± {mae_std:.2f} | "
            f"lr={lr}, max_iter={mi}"
        )

if __name__ == "__main__":
    main()