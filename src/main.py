import numpy as np
import json
import pandas as pd

from .config import OUTPUT_DIR
from .data_loader import load_data
import pandas as pd
from .visualize import (
    plot_saleprice_distribution,
    plot_log_saleprice_distribution,
    plot_top_missing_values,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)
from .modeling import (
    fit_and_compare_models,
    prepare_training_data,
    build_preprocessor,
    get_transformed_feature_names,
    extract_linear_coefficients,
    extract_tree_importance,
)
from .features import remove_identifier_columns, add_engineered_features

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()

    # Basic EDA plots
    plot_saleprice_distribution(train_df)
    plot_log_saleprice_distribution(train_df)
    plot_top_missing_values(train_df)
    plot_correlation_heatmap(train_df)

    trained_models, best_params, results_df, predictions_store = fit_and_compare_models(train_df)

    print("\nModel comparison results:\n")
    print(results_df)

    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Plots for the best model
    y_true = predictions_store[best_model_name]["y_true"]
    y_pred = predictions_store[best_model_name]["y_pred"]
    plot_actual_vs_predicted(y_true, y_pred, best_model_name)
    plot_residuals(y_true, y_pred, best_model_name)

    # Extract feature names from the fitted preprocessing pipeline
    X, _, _ = prepare_training_data(train_df)
    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = get_transformed_feature_names(preprocessor, X.columns)

    linear_coef_df = extract_linear_coefficients(best_model, feature_names)
    if linear_coef_df is not None:
        linear_coef_df.head(30).to_csv(OUTPUT_DIR / "top_linear_coefficients.csv", index=False)
        print("\nTop linear coefficients saved to outputs/top_linear_coefficients.csv")

    tree_importance_df = extract_tree_importance(best_model, feature_names)
    if tree_importance_df is not None:
        tree_importance_df.head(30).to_csv(OUTPUT_DIR / "top_tree_importance.csv", index=False)
        plot_feature_importance(
            tree_importance_df["Importance"].values[:15],
            tree_importance_df["Feature"].values[:15],
            best_model_name,
            top_n=15
        )
        print("\nTop tree feature importance saved to outputs/top_tree_importance.csv")


    # Fit the best model on the full training set and create Kaggle-style test predictions
    test_features = add_engineered_features(remove_identifier_columns(test_df))
    X_full, _, y_full_log = prepare_training_data(train_df)
    best_model.fit(X_full, y_full_log)
    test_predictions = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": np.expm1(best_model.predict(test_features))
    })
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"\nBest model: {best_model_name}")
    print("Project finished successfully.")

if __name__ == "__main__":
    main()
