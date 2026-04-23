"""Utilities for House Price Prediction project."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
FIG_DIR = OUTPUT_DIR / 'figures'
MODEL_DIR = OUTPUT_DIR / 'models'
METRIC_DIR = OUTPUT_DIR / 'metrics'
for folder in [OUTPUT_DIR, FIG_DIR, MODEL_DIR, METRIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = 'SalePrice'


def print_header(title: str):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def find_existing_file(candidates: list[str]) -> Path:
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            return path
    available = ', '.join(sorted(p.name for p in DATA_DIR.glob('*.csv'))) or 'No CSV files found'
    raise FileNotFoundError(
        'Could not find any expected file in data/. '        f'Tried: {candidates}. Available CSV files: {available}'
    )


def load_train_data() -> pd.DataFrame:
    train_path = find_existing_file(['train.csv', 'train(1).csv'])
    print(f'Using training file: {train_path}')
    df = pd.read_csv(train_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in training file, but it was not found.")
    print(f'Training data shape: {df.shape}')
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False):
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features


def evaluate_model(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred))
    }


def save_metrics(metrics: dict, filename: str):
    path = METRIC_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f'Saved metrics: {path.resolve()}')


def save_predictions(y_test, y_pred, filename: str):
    path = OUTPUT_DIR / filename
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residual': y_test - y_pred})
    pred_df.to_csv(path, index=False)
    print(f'Saved predictions: {path.resolve()}')


def plot_actual_vs_predicted(y_true, y_pred, output_path: Path, title: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved figure: {output_path.resolve()}')


def plot_residuals(y_true, y_pred, output_path: Path, title: str):
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted SalePrice')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved figure: {output_path.resolve()}')


def get_feature_names(preprocessor, numeric_features, categorical_features):
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names
