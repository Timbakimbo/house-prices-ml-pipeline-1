import pandas as pd
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

def split_by_cardinality(X,cat_cols, threshold=10):
    low_card_cols = []
    high_card_cols = []
    for col in cat_cols:
        if X[col].nunique(dropna=True) <= threshold:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)
    return low_card_cols, high_card_cols

def build_preprocessor(X,num_cols, cat_cols, card_threshold=10):
    #als erstes cat values nach cardinality splitten
    low_card_cols, high_card_cols = split_by_cardinality(X,cat_cols, threshold=card_threshold)
    #num pipeline für missing values
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    #cat pipeline für low cardinality & missing values
    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    #cat pipeline für high cardinality & missing values
    cat_high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])
    cat_pipe = ColumnTransformer(
        transformers=[
            ("low_card", low_cat_pipe, low_card_cols),
            ("high_card", cat_high_pipe, high_card_cols),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )










