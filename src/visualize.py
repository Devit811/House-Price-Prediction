import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURE_DIR

def ensure_figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def plot_saleprice_distribution(train_df: pd.DataFrame):
    ensure_figure_dir()
    plt.figure(figsize=(8, 5))
    plt.hist(train_df["SalePrice"], bins=30)
    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "saleprice_distribution.png", dpi=200)
    plt.close()

def plot_log_saleprice_distribution(train_df: pd.DataFrame):
    ensure_figure_dir()
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(train_df["SalePrice"]), bins=30)
    plt.title("Distribution of log(1 + SalePrice)")
    plt.xlabel("log(1 + SalePrice)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "log_saleprice_distribution.png", dpi=200)
    plt.close()

def plot_top_missing_values(df: pd.DataFrame, top_n: int = 20):
    ensure_figure_dir()
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)

    if len(missing_pct) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(missing_pct.index[::-1], missing_pct.values[::-1])
    plt.title("Top Missing Value Percentages")
    plt.xlabel("Missing Percentage")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missing_values_top20.png", dpi=200)
    plt.close()

def plot_correlation_heatmap(train_df: pd.DataFrame, top_n: int = 15):
    ensure_figure_dir()
    numeric_df = train_df.select_dtypes(include=["int64", "float64"])
    corr_with_target = numeric_df.corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)
    top_features = corr_with_target.head(top_n).index.tolist()
    corr_matrix = numeric_df[top_features].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, aspect="auto")
    plt.xticks(range(len(top_features)), top_features, rotation=90)
    plt.yticks(range(len(top_features)), top_features)
    plt.title("Correlation Heatmap of Top Numeric Features")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap_top_features.png", dpi=200)
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, model_name: str):
    ensure_figure_dir()
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    filename = f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()

def plot_residuals(y_true, y_pred, model_name: str):
    ensure_figure_dir()
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0)
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    filename = f"residuals_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()

def plot_feature_importance(importances, feature_names, model_name: str, top_n: int = 15):
    ensure_figure_dir()
    if len(importances) != len(feature_names):
        return
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(9, 6))
    plt.barh(feature_importance.index[::-1], feature_importance.values[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top Feature Importance - {model_name}")
    plt.tight_layout()
    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()
