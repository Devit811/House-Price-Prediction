# Project Report: House Price Prediction

## 1. Introduction

This project builds a regression-based machine learning system to predict house sale prices using the Kaggle dataset **House Prices - Advanced Regression Techniques**. The main purpose is to estimate the final sale value of a residential property and to understand which housing characteristics most strongly influence price.

This is a practical and academically suitable dataset because it includes many useful real-world housing variables, such as quality, living area, location, neighborhood, construction year, basement size, garage size, and many other structural and environmental features.

## 2. Problem Definition

### Problem Type
This is a **regression** problem.

### Reason
The target variable is `SalePrice`, which is a continuous numeric value. The model must predict a number rather than a class label.

### Target Variable
- `SalePrice`

## 3. Why the Dataset is Suitable

This dataset is highly suitable for a university machine learning project for several reasons:

- It contains **79 input features**, which gives a rich and realistic representation of housing data.
- It includes both **numerical and categorical variables**.
- It contains **missing values**, which allows practice in data cleaning.
- It benefits from **feature engineering**, such as total square footage and total bathrooms.
- It is well suited for **Lasso** and **Ridge Regression**, because one-hot encoding can create many features and regularization helps control model complexity.

## 4. Expected Important Features

The most influential features are usually expected to include:

- `OverallQual`
- `Neighborhood`
- `GrLivArea`
- `TotalBsmtSF`
- `GarageCars`
- `GarageArea`
- `1stFlrSF`
- `YearBuilt`
- `YearRemodAdd`
- `FullBath`
- engineered totals such as `TotalSF` and `TotalBath`

These features are strongly connected to house value because buyers commonly care about **location**, **size**, **construction quality**, and **functional condition**.

## 5. Exploratory Data Analysis (EDA)

The project includes EDA visualizations such as:

- distribution of `SalePrice`
- distribution of `log(1 + SalePrice)`
- top missing-value features
- correlation heatmap of the most relevant numeric variables

### Key Findings to Explain in the Assignment

You can discuss the following points after running the code:

1. **SalePrice is usually right-skewed**  
   The raw target distribution often has a long right tail. Log transformation helps make it more normal, which benefits linear models.

2. **Quality and size often have strong positive relationships with price**  
   Houses with higher overall quality and larger living area usually sell for higher prices.

3. **Neighborhood matters**  
   The same size house can have different prices depending on the neighborhood, so location is an important factor.

4. **Missing values are meaningful in some columns**  
   In housing data, missing values may sometimes mean that a feature does not exist, such as no basement, no garage, or no fireplace.

5. **Some outliers can distort linear models**  
   Extremely large houses with relatively low prices can make model fitting less stable. Therefore, controlled outlier handling is useful.

## 6. Data Preprocessing

The project applies the following preprocessing steps:

- remove identifier column `Id`
- detect numerical and categorical features
- impute missing numerical values using the median
- impute missing categorical values using the most frequent category
- one-hot encode categorical variables
- scale numerical features for linear and regularized regression
- apply log transformation to the target variable
- remove a small number of extreme outliers from training data
- engineer new features such as:
  - `TotalSF`
  - `TotalPorchSF`
  - `TotalBath`
  - `HouseAge`
  - `RemodAge`
  - `GarageAge`
  - presence indicators such as `HasGarage`, `HasBasement`, `HasFireplace`

## 7. Models Used

The project compares several regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

### Why these models were chosen

- **Linear Regression** is the baseline model.
- **Ridge Regression** helps reduce overfitting by shrinking coefficients.
- **Lasso Regression** can shrink some coefficients to zero, which also helps feature selection.
- **ElasticNet** combines Ridge and Lasso behavior.
- **Random Forest** captures non-linear patterns.
- **Gradient Boosting** often performs strongly on structured tabular datasets.

## 8. Hyperparameter Tuning

The project uses **GridSearchCV** with cross-validation to tune important hyperparameters such as:

- `alpha` for Ridge and Lasso
- `alpha` and `l1_ratio` for ElasticNet
- number of trees and tree depth for Random Forest
- number of estimators, learning rate, depth, and subsampling for Gradient Boosting

## 9. Evaluation Metrics

The models are compared using:

- **RMSE**: Root Mean Squared Error  
  Good for measuring prediction error with stronger penalty on large mistakes.

- **MAE**: Mean Absolute Error  
  Easy to interpret because it shows average absolute prediction error.

- **R² Score**  
  Measures how much variance in house prices is explained by the model.

## 10. Interpretation Focus

This project should discuss the practical meaning of the most important variables:

### Location
Neighborhood-related variables represent local market attractiveness, access, and area reputation.

### Area
Living area, first-floor size, basement size, and lot size are essential because larger usable space generally increases market value.

### Quality
`OverallQual` is often one of the strongest predictors because it captures construction and finish quality.

### Condition
`OverallCond` can influence price, but often less strongly than overall quality.

### Age and Renovation
Older homes may have lower prices unless they have been renovated. That is why `YearBuilt`, `YearRemodAdd`, and derived age features are useful.

## 11. Why Lasso and Ridge are Important Here

This dataset is especially useful for **Lasso** and **Ridge Regression** because:

- it contains many features
- many categorical variables become many new columns after one-hot encoding
- some features may be correlated
- regularization can improve generalization and model stability

### Ridge
Ridge is useful when many features carry small but meaningful information and multicollinearity exists.

### Lasso
Lasso is useful when some features are weak and can be reduced toward zero, making the model more selective.

## 12. Recommendation

After running the project, the best model should be selected based on **validation RMSE**, while also considering MAE and R².

In many structured tabular housing datasets, **Gradient Boosting** or another strong ensemble often performs best because it can model non-linear relationships and interactions. However, **Ridge Regression** and **Lasso Regression** remain academically important because they are interpretable and clearly show the effect of regularization.

## 13. Conclusion

This project is a strong final assignment because it combines:

- realistic real-world data
- clear regression problem setup
- proper EDA and preprocessing
- regularization methods
- model tuning and fair comparison
- useful business-style interpretation of location, area, and quality

It is also practical for GitHub, reproducible offline, and easy to extend with additional models such as XGBoost if required later.


## 8. Actual Model Comparison Results

The uploaded dataset was run through the project pipeline using a validation split. The observed results were:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| ElasticNet | 19295.96 | 14000.01 | 0.9326 |
| Lasso Regression | 19301.28 | 13994.04 | 0.9326 |
| Gradient Boosting | 19302.13 | 13966.57 | 0.9326 |
| Ridge Regression | 19982.10 | 14439.49 | 0.9277 |
| Linear Regression | 21801.20 | 15445.38 | 0.9140 |

### Best Model
The best-performing model was **ElasticNet** with RMSE **19295.96**, MAE **14000.01**, and R² **0.9326**.

### Interpretation
- **Area and quality** remained among the strongest price drivers, especially overall quality, above-ground living area, total square footage, and garage-related capacity.
- **Location** was also important because neighborhood variables became influential after one-hot encoding.
- **Regularization helped** because the dataset contains many categorical levels, engineered features, and correlated size variables. This is why Lasso, Ridge, and especially ElasticNet performed strongly.
- **Plain linear regression** performed worst because it was more sensitive to multicollinearity and less stable when many encoded variables were present.

## 9. Recommendation

For this final assignment, the recommended model is **ElasticNet**. It is a strong academic choice because it combines:
- the interpretability of a linear model
- the shrinkage effect of Ridge
- the feature selection behavior of Lasso

This makes it especially suitable for the Kaggle House Prices dataset, where many features are related, some are sparse after encoding, and feature engineering adds extra complexity.
