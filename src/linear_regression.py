"""Linear Regression model for House Price Prediction."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from common import (
    FIG_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE,
    build_preprocessor, evaluate_model, get_feature_names,
    load_train_data, plot_actual_vs_predicted, plot_residuals,
    print_header, save_metrics, save_predictions, split_features_target,
)


def plot_top_coefficients(feature_names, coefficients, output_path: Path, top_n: int = 20):
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
    top_coef = coef_df.sort_values('AbsCoefficient', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top_coef['Feature'][::-1], top_coef['Coefficient'][::-1])
    plt.xlabel('Coefficient Value')
    plt.title('Top Linear Regression Coefficients')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved figure: {output_path.resolve()}')


def main():
    print_header('LINEAR REGRESSION - HOUSE PRICE PREDICTION')
    print('Step 1/7: Loading data...')
    df = load_train_data()
    X, y = split_features_target(df)

    print('Step 2/7: Splitting train and validation data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print('Step 3/7: Building preprocessing pipeline...')
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train, scale_numeric=True)

    print('Step 4/7: Training Linear Regression model...')
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)

    print('Step 5/7: Generating predictions...')
    y_pred = model.predict(X_test)

    print('Step 6/7: Evaluating model...')
    metrics = evaluate_model(y_test, y_pred)
    print('Linear Regression Metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')

    print('Step 7/7: Saving outputs...')
    save_metrics(metrics, 'linear_regression_metrics.json')
    save_predictions(y_test, y_pred, 'linear_regression_predictions.csv')
    joblib.dump(model, MODEL_DIR / 'linear_regression_model.joblib')
    print(f'Saved model: {(MODEL_DIR / "linear_regression_model.joblib").resolve()}')
    plot_actual_vs_predicted(y_test, y_pred, FIG_DIR / 'linear_actual_vs_predicted.png', 'Linear Regression: Actual vs Predicted')
    plot_residuals(y_test, y_pred, FIG_DIR / 'linear_residuals.png', 'Linear Regression: Residual Plot')
    fitted_preprocessor = model.named_steps['preprocessor']
    feature_names = get_feature_names(fitted_preprocessor, numeric_features, categorical_features)
    coefficients = model.named_steps['regressor'].coef_
    plot_top_coefficients(feature_names, coefficients, FIG_DIR / 'linear_top_coefficients.png')
    print(f'\nCompleted successfully. Check outputs here: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
