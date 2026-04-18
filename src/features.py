import numpy as np
import pandas as pd

def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df

def basic_outlier_filter(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a few known extreme outliers often discussed in this dataset:
    very large living area with unusually low sale price.
    This is applied only to training data.
    """
    df = train_df.copy()
    if {"GrLivArea", "SalePrice"}.issubset(df.columns):
        df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]
    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create useful engineered features that often improve regression performance.
    """
    df = df.copy()

    def safe_col(col_name, default=0):
        return df[col_name] if col_name in df.columns else default

    # Total area related features
    df["TotalSF"] = (
        safe_col("TotalBsmtSF", 0)
        + safe_col("1stFlrSF", 0)
        + safe_col("2ndFlrSF", 0)
    )

    df["TotalPorchSF"] = (
        safe_col("OpenPorchSF", 0)
        + safe_col("EnclosedPorch", 0)
        + safe_col("3SsnPorch", 0)
        + safe_col("ScreenPorch", 0)
        + safe_col("WoodDeckSF", 0)
    )

    df["TotalBath"] = (
        safe_col("FullBath", 0)
        + 0.5 * safe_col("HalfBath", 0)
        + safe_col("BsmtFullBath", 0)
        + 0.5 * safe_col("BsmtHalfBath", 0)
    )

    df["HouseAge"] = safe_col("YrSold", 0) - safe_col("YearBuilt", 0)
    df["RemodAge"] = safe_col("YrSold", 0) - safe_col("YearRemodAdd", 0)
    df["GarageAge"] = safe_col("YrSold", 0) - safe_col("GarageYrBlt", 0)

    # Binary presence features
    df["HasBasement"] = (safe_col("TotalBsmtSF", 0) > 0).astype(int)
    df["HasGarage"] = (safe_col("GarageArea", 0) > 0).astype(int)
    df["HasFireplace"] = (safe_col("Fireplaces", 0) > 0).astype(int)
    df["HasPool"] = (safe_col("PoolArea", 0) > 0).astype(int)
    df["Has2ndFloor"] = (safe_col("2ndFlrSF", 0) > 0).astype(int)

    return df

def log_transform_target(y: pd.Series) -> pd.Series:
    return np.log1p(y)

def inverse_log_target(y_log):
    return np.expm1(y_log)

def get_feature_groups(df: pd.DataFrame):
    """
    Return numerical and categorical feature names.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return numerical_cols, categorical_cols
