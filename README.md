# Breast Cancer ML Project 

## Breast Cancer Relapse-Free Survival (RFS) Prediction
This repository contains the source code and documentation for **Part 1** of the COMP4139 Machine Learning Final Assignment (2025). The objective of this task is to develop a machine learning regression pipeline to predict **Relapse-Free Survival (RFS)** time for breast cancer patients undergoing chemotherapy.

The project utilises a dataset comprising both clinical data (e.g., Age, ER status, Tumour Stage) and radiomics features extracted from medical imaging.

## Methodology

The pipeline follows a standard machine learning workflow using **Scikit-Learn** and **XGBoost**:

### 1. Data Preprocessing
**Preprocessing**:
* Missing values are filled using KNN Imputation.
* Data is scaled using `StandardScaler`.
**Feature Selection**:
* Used `SelectKBest` (f_regression) to filter out noise and keep only the most relevant features.
**Modeling**:
* Tested several regressors including Linear Regression, SVR, Random Forest, and XGBoost.


### 2. Model Selection
Several regression algorithms were evaluated using Cross-Validation:
- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### 3. Hyperparameter Tuning
Hyperparameters were tuned using `GridSearchCV` to minimise the **Mean Absolute Error (MAE)**.
The best performing model is automatically saved as `rfs_model.joblib`.
