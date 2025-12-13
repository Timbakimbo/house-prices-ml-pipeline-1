import pandas as pd
from src.data_loading import load_raw_data
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def split_target_feature(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
def split_num_cat(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )










