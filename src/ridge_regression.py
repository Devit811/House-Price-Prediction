"""Ridge Regression model with hyperparameter tuning for House Price Prediction."""


from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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


def load_data():
    train_path = DATA_DIR / 'train.csv'
    if not train_path.exists():
        raise FileNotFoundError(
            f"Could not find {train_path}. Please place Kaggle train.csv inside the data/ folder."
        )
    df = pd.read_csv(train_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in train.csv")
    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].copy()
    return X, y


def build_preprocessor(X, scale_numeric=False):
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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def save_metrics(metrics, filename):
    with open(METRIC_DIR / filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


def save_predictions(y_test, y_pred, filename):
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residual': y_test - y_pred})
    pred_df.to_csv(OUTPUT_DIR / filename, index=False)


def plot_actual_vs_predicted(y_true, y_pred, output_path, title):
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


def plot_residuals(y_true, y_pred, output_path, title):
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


def plot_comparison_bar(metrics_dict, metric_name, output_path, title):
    model_names = list(metrics_dict.keys())
    values = [metrics_dict[m][metric_name] for m in model_names]
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, values)
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_feature_names(preprocessor, numeric_features, categorical_features):
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def plot_top_coefficients(feature_names, coefficients, output_path, top_n=20):
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
    top_coef = coef_df.sort_values('AbsCoefficient', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_coef['Feature'][::-1], top_coef['Coefficient'][::-1])
    plt.xlabel('Coefficient Value')
    plt.title('Top Ridge Regression Coefficients')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train, scale_numeric=True)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    metrics['best_alpha'] = float(grid_search.best_params_['regressor__alpha'])

    print('Ridge Regression Metrics:')
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value}')

    save_metrics(metrics, 'ridge_regression_metrics.json')
    save_predictions(y_test, y_pred, 'ridge_regression_predictions.csv')
    joblib.dump(best_model, MODEL_DIR / 'ridge_regression_model.joblib')

    plot_actual_vs_predicted(
        y_test, y_pred,
        FIG_DIR / 'ridge_actual_vs_predicted.png',
        'Ridge Regression: Actual vs Predicted'
    )
    plot_residuals(
        y_test, y_pred,
        FIG_DIR / 'ridge_residuals.png',
        'Ridge Regression: Residual Plot'
    )

    fitted_preprocessor = best_model.named_steps['preprocessor']
    feature_names = get_feature_names(fitted_preprocessor, numeric_features, categorical_features)
    coefficients = best_model.named_steps['regressor'].coef_
    plot_top_coefficients(
        feature_names, coefficients,
        FIG_DIR / 'ridge_top_coefficients.png'
    )


if __name__ == '__main__':
    main()
