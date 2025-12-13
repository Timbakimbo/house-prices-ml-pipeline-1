import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error 
from src.pre_processing import split_target_feature, split_num_cat, build_preprocessor
from src.data_loading import load_raw_data
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

    print(f'Baseline RF Validation RMSE: {baseline_rf_rmse}')
    print(f'Baseline RF Validation MAE: {baseline_rf_mae}')
    
    # Train second model with better preprocessing
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipe, rmse, mae = train_rf_2(X, y, preprocessor, random_state=0)

    print(f"Baseline RF RMSE: {rmse:.2f}")
    print(f"Baseline RF MAE : {mae:.2f}")

if __name__ == "__main__":
    main()