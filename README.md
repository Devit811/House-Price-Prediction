# House Price Prediction

A complete **machine learning regression project** for predicting house sale prices using the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

## Project Overview

This project predicts **house sale prices** using housing-related features such as:

- location and neighborhood
- living area and lot area
- overall quality and condition
- year built and remodel year
- basement, garage, and exterior features
- many other structural and property characteristics

The project is designed for a **final university machine learning assignment** and demonstrates:

- regression problem formulation
- exploratory data analysis (EDA)
- missing value treatment
- categorical encoding
- scaling of numerical features
- outlier handling
- feature engineering
- regularized linear models such as **Ridge** and **Lasso**
- model comparison using **RMSE, MAE, and R²**
- result interpretation in clear academic English

## Why this dataset is suitable

The Kaggle House Prices dataset is highly suitable for a regression project because:

1. **The target is continuous**  
   The target variable is `SalePrice`, which is a numeric value. This makes the task a regression problem.

2. **The dataset is rich and realistic**  
   It contains **79 explanatory features** that describe size, quality, age, neighborhood, condition, and many other housing factors.

3. **It supports strong feature engineering practice**  
   The data contains mixed variable types, missing values, skewed variables, and several feature interactions. This makes it excellent for practicing real machine learning workflow.

4. **It is ideal for regularization methods**  
   Since the dataset has many features, including encoded categorical variables, it becomes a strong case for **Ridge** and **Lasso Regression**, which help control overfitting and improve generalization.

## Machine Learning Type

This project is a **Regression** project because the goal is to predict a **continuous numerical value**, which is the final sale price of a house.

- **Target variable:** `SalePrice`

## Project Structure

```bash
house-price-prediction-project/
│
├── data/
│   ├── train.csv                  # Place Kaggle training file here
│   ├── test.csv                   # Place Kaggle test file here
│   └── README_DATA.md
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── main.py
│
├── reports/
│   ├── project_report.md
│   └── team_contribution.md
│
├── outputs/
│   └── figures/                   # Generated plots will be saved here after running the project
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset Setup Instructions

This project is designed to run **offline**.

Place these files into the `data/` folder:
- `train.csv`
- `test.csv`
- optional: `sample_submission.csv`
- optional: `data_description.txt`

Do **not** change the filenames.

If you already received the prepared ZIP version from this chat, those files may already be included.

## How to Run

### Option 1: Run the full pipeline as a script

```bash
python -m src.main
```

### Option 2: Use the notebook

Open:

```bash
notebooks/house_price_prediction.ipynb
```

and run the cells step by step.

## Main Workflow

The project includes the following stages:

1. Load training and test data
2. Perform EDA
3. Analyze missing values
4. Engineer useful features
5. Handle outliers
6. Build preprocessing pipeline
7. Train multiple regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet
   - Random Forest Regressor
   - Gradient Boosting Regressor
8. Tune hyperparameters with cross-validation
9. Compare models using:
   - RMSE
   - MAE
   - R²
10. Save plots and results
11. Recommend the best model

## Expected Important Features

Although the exact importance may vary after training, the most influential features usually include:

- `OverallQual`
- `GrLivArea`
- `Neighborhood`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `YearBuilt`
- `YearRemodAdd`
- `FullBath`
- engineered totals such as total area and total bathrooms

## Interpretation Focus

This project pays special attention to the influence of:

- **location** through `Neighborhood`
- **area** through `GrLivArea`, `LotArea`, basement area, and floor area
- **quality** through `OverallQual` and material-related variables
- **condition** through `OverallCond`
- **property age and remodeling** through `YearBuilt` and `YearRemodAdd`

## Notes on Data Leakage and Feature Removal

This project does not use `SalePrice` in feature engineering except as the target.  
The code also includes logic and notes for handling:

- near-constant or weakly useful features
- features with too many missing values
- extreme outliers that may distort linear models

The `Id` column is removed from modeling because it is an identifier, not a predictive feature.

## Team Contribution Template

See:

```bash
reports/team_contribution.md
```

You can edit the names and responsibilities for your team.

## GitHub Ready

This repository is organized so you can upload it directly to GitHub after adding the dataset files.

## Suggested GitHub Description

**House Price Prediction is a final Machine Learning project that applies regression models and feature engineering on the Kaggle House Prices dataset to predict sale prices and analyze the influence of location, area, and quality.**


## Actual Results from the Included Run

The project was executed on the uploaded Kaggle dataset files. The hold-out validation results are:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| ElasticNet | 19295.96 | 14000.01 | 0.9326 |
| Lasso Regression | 19301.28 | 13994.04 | 0.9326 |
| Gradient Boosting | 19302.13 | 13966.57 | 0.9326 |
| Ridge Regression | 19982.10 | 14439.49 | 0.9277 |
| Linear Regression | 21801.20 | 15445.38 | 0.9140 |

**Best model:** ElasticNet

Why it performed best in this run:
- it handled many encoded features better than plain linear regression
- it balanced coefficient shrinkage and feature selection
- it gave the strongest overall trade-off across RMSE, MAE, and R²

Generated outputs are saved in `outputs/`, including:
- `model_comparison.csv`
- `best_params.json`
- `test_predictions.csv`
- `top_linear_coefficients.csv`
- plots in `outputs/figures/`
