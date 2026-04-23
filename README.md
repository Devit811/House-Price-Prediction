# House Price Prediction

A complete, student-friendly machine learning regression project for predicting house sale prices using the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

## Project Overview

This project builds and compares three regression models for house price prediction:

1. **Linear Regression**
2. **Ridge Regression**
3. **Random Forest Regression**

The workflow follows a realistic university final project structure:

- exploratory data analysis (EDA)
- preprocessing and feature engineering
- model training
- hyperparameter tuning
- evaluation using RMSE, MAE, and R²
- visual comparison of models
- feature importance analysis

## Dataset

Recommended dataset: **House Prices - Advanced Regression Techniques**  
Source: Kaggle  
Files to place manually inside the `data/` folder:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

> Important: This project does **not** download the dataset automatically. You must place the dataset files manually in the `data/` folder.

## Machine Learning Type

This is a **Regression** project because the target variable, `SalePrice`, is a continuous numerical value.

## Project Structure

```text
house_price_prediction_project/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── src/
│   ├── linear_regression.py
│   ├── ridge_regression.py
│   └── random_forest.py
├── notebooks/
│   ├── 00_eda_preprocessing.ipynb
│   ├── 01_linear_regression.ipynb
│   ├── 02_ridge_regression.ipynb
│   └── 03_random_forest.ipynb
└── outputs/
    ├── figures/
    ├── metrics/
    └── models/
```

## Objectives

- Understand the house price prediction problem
- Explore and clean the dataset
- Build three regression models
- Tune the important hyperparameters
- Compare performance fairly
- Explain which features most influence predictions
- Present results in an academic and professional style

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Make sure your dataset files are already placed in the `data/` folder.

Run the models one by one:

```bash
python src/linear_regression.py
python src/ridge_regression.py
python src/random_forest.py
```

## Outputs Produced

Each script saves outputs such as:

- metrics JSON files in `outputs/metrics/`
- plots in `outputs/figures/`
- trained model files in `outputs/models/`
- prediction comparison CSV files in `outputs/`

Typical visualizations include:

- actual vs predicted scatter plot
- residual plot
- feature importance / coefficient plot
- model comparison chart

## Model Details

### 1. Linear Regression
- Baseline linear model
- Easy to interpret
- Good for understanding general linear relationships

### 2. Ridge Regression
- Regularized version of linear regression
- Helps reduce overfitting
- Useful when many features are correlated after one-hot encoding

### 3. Random Forest Regression
- Non-linear ensemble model
- Usually performs better on complex tabular data
- Can capture interactions without heavy manual feature engineering

## Evaluation Metrics

The models are compared using:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

For house prices, lower RMSE and MAE are better, while higher R² is better.

## Suggested Academic Interpretation

- Linear Regression is the simplest baseline.
- Ridge Regression usually improves generalization by shrinking coefficients.
- Random Forest often captures more complex housing patterns and may achieve better predictive performance.
- Feature importance helps explain which housing characteristics affect price the most.

## Team Contribution Note

You can edit this section with your real team members.

| Team Member | Contribution |
|---|---|
| Student A | Implemented Linear Regression, baseline evaluation, and documentation |
| Student B | Implemented Ridge Regression, hyperparameter tuning, and comparison analysis |
| Student C | Implemented Random Forest Regression, feature importance analysis, and final result summary |

If your group has only 2 members, one student can take two responsibilities.

## Notes for Final Submission

- Keep screenshots of plots for your report or slides.
- Run all notebooks before submission so outputs are visible.
- Update team names, student IDs, and final results after running on your dataset.
- You can extend the project with Extra Trees, XGBoost, or Lasso as future work.

## Possible Future Improvements

- More advanced feature engineering
- Log-transform the target variable
- Outlier treatment
- Stacking or boosting models
- Error analysis by neighborhood or house quality

## License

This project is for academic and learning purposes.


## Fixed package notes
- This updated package already includes `train.csv` and `test.csv` in the `data/` folder.
- Scripts now print progress step-by-step and show the full output save path.
- The code can also detect `train(1).csv` and `test(1).csv` if those names are used.
- Start with `python src/linear_regression.py` to test the setup quickly.
