"""Random Forest Regression model with hyperparameter tuning for House Price Prediction."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from common import (
    FIG_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE,
    build_preprocessor, evaluate_model, get_feature_names,
    load_train_data, plot_actual_vs_predicted, plot_residuals,
    print_header, save_metrics, save_predictions, split_features_target,
)


def plot_feature_importance(feature_names, importances, output_path: Path, top_n: int = 20):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_importance = importance_df.sort_values('Importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top_importance['Feature'][::-1], top_importance['Importance'][::-1])
    plt.xlabel('Importance Score')
    plt.title('Top Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved figure: {output_path.resolve()}')


def main():
    print_header('RANDOM FOREST REGRESSION - HOUSE PRICE PREDICTION')
    print('Step 1/7: Loading data...')
    df = load_train_data()
    X, y = split_features_target(df)

    print('Step 2/7: Splitting train and validation data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print('Step 3/7: Building preprocessing pipeline...')
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train, scale_numeric=False)

    print('Step 4/7: Running GridSearchCV for Random Forest...')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    param_grid = {
        'regressor__n_estimators': [100, 150],
        'regressor__max_depth': [None, 10],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print('Step 5/7: Generating predictions...')
    y_pred = best_model.predict(X_test)

    print('Step 6/7: Evaluating model...')
    metrics = evaluate_model(y_test, y_pred)
    metrics.update({
        'best_n_estimators': int(grid_search.best_params_['regressor__n_estimators']),
        'best_max_depth': None if grid_search.best_params_['regressor__max_depth'] is None else int(grid_search.best_params_['regressor__max_depth']),
        'best_min_samples_split': int(grid_search.best_params_['regressor__min_samples_split']),
        'best_min_samples_leaf': int(grid_search.best_params_['regressor__min_samples_leaf']),
    })
    print('Random Forest Regression Metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')

    print('Step 7/7: Saving outputs...')
    save_metrics(metrics, 'random_forest_metrics.json')
    save_predictions(y_test, y_pred, 'random_forest_predictions.csv')
    joblib.dump(best_model, MODEL_DIR / 'random_forest_model.joblib')
    print(f'Saved model: {(MODEL_DIR / "random_forest_model.joblib").resolve()}')
    plot_actual_vs_predicted(y_test, y_pred, FIG_DIR / 'rf_actual_vs_predicted.png', 'Random Forest: Actual vs Predicted')
    plot_residuals(y_test, y_pred, FIG_DIR / 'rf_residuals.png', 'Random Forest: Residual Plot')
    fitted_preprocessor = best_model.named_steps['preprocessor']
    feature_names = get_feature_names(fitted_preprocessor, numeric_features, categorical_features)
    importances = best_model.named_steps['regressor'].feature_importances_
    plot_feature_importance(feature_names, importances, FIG_DIR / 'rf_feature_importance.png')
    print(f'\nCompleted successfully. Check outputs here: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
