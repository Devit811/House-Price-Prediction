import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    get_feature_groups,
    add_engineered_features,
    remove_identifier_columns,
    basic_outlier_filter,
    log_transform_target,
    inverse_log_target,
)
from .evaluate import regression_metrics, results_to_dataframe


def build_preprocessor(X: pd.DataFrame):
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_candidate_models():
    """
    Lightweight tuning by validation set.
    This keeps the project practical to run on a normal student laptop
    while still comparing several tuned regression models.
    """
    return {
        "Linear Regression": [LinearRegression()],
        "Ridge Regression": [Ridge(alpha=a) for a in [1.0, 10.0, 50.0]],
        "Lasso Regression": [Lasso(alpha=a, max_iter=10000) for a in [0.0005, 0.001, 0.01]],
        "ElasticNet": [
            ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
            for a in [0.0005, 0.001, 0.01]
            for l1 in [0.2, 0.5, 0.8]
        ],
        "Gradient Boosting": [
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=200,
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
            )
            for lr in [0.05, 0.1]
            for depth in [2, 3]
            for subsample in [0.8, 1.0]
        ],
    }


def prepare_training_data(train_df):
    train_df = basic_outlier_filter(train_df)
    train_df = remove_identifier_columns(train_df)
    train_df = add_engineered_features(train_df)

    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    y_log = log_transform_target(y)

    return X, y, y_log


def fit_and_compare_models(train_df):
    X, y, y_log = prepare_training_data(train_df)

    X_train, X_valid, y_train_log, _, y_train_raw, y_valid_raw = train_test_split(
        X, y_log, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    candidate_models = get_candidate_models()

    trained_models = {}
    best_params = {}
    results = {}
    predictions_store = {}

    for model_name, model_list in candidate_models.items():
        best_rmse = float("inf")
        best_model = None
        best_pred = None
        best_model_params = {}

        for model in model_list:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train_log)

            pred_log = pipeline.predict(X_valid)
            pred_raw = inverse_log_target(pred_log)
            metrics = regression_metrics(y_valid_raw, pred_raw)

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = pipeline
                best_pred = pred_raw
                best_model_params = {
                    key: value
                    for key, value in model.get_params().items()
                    if key in [
                        "alpha",
                        "l1_ratio",
                        "n_estimators",
                        "learning_rate",
                        "max_depth",
                        "subsample",
                    ]
                }
                results[model_name] = metrics

        trained_models[model_name] = best_model
        best_params[model_name] = best_model_params
        predictions_store[model_name] = {
            "y_true": y_valid_raw,
            "y_pred": best_pred,
        }

    results_df = results_to_dataframe(results)
    return trained_models, best_params, results_df, predictions_store


def get_transformed_feature_names(preprocessor, input_features):
    try:
        return preprocessor.get_feature_names_out(input_features)
    except Exception:
        try:
            return preprocessor.get_feature_names_out()
        except Exception:
            return None


def extract_linear_coefficients(trained_pipeline, feature_names):
    model = trained_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None

    coef = np.ravel(model.coef_)
    if feature_names is None or len(coef) != len(feature_names):
        return None

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef,
        "AbsoluteCoefficient": np.abs(coef)
    }).sort_values("AbsoluteCoefficient", ascending=False)
    return coef_df


def extract_tree_importance(trained_pipeline, feature_names):
    model = trained_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.ravel(model.feature_importances_)
    if feature_names is None or len(importances) != len(feature_names):
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    return importance_df
