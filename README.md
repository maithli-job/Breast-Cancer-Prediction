#  Breast Cancer Tumor Classification

##  Project Overview

This project is a Machine Learning classification system that predicts whether a breast tumor is **Benign** or **Malignant** using the Breast Cancer Wisconsin dataset.

The project uses different classification algorithms and compares their performance to select the best model.

A Streamlit web application is also developed to allow users to enter tumor measurements and get a prediction.

---

##  Objectives

- Load and analyze the Breast Cancer Wisconsin dataset.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and duplicate records.
- Select important features for classification.
- Analyze correlation and covariance.
- Train multiple Machine Learning classification models.
- Compare model performance.
- Select and save the best-performing model.
- Create a user-friendly Streamlit interface for prediction.

---

##  Dataset

The project uses the **Breast Cancer Wisconsin dataset**.

The selected features used for classification are:

- `radius_mean`
- `perimeter_mean`
- `area_mean`
- `concavity_mean`
- `concave points_mean`
- `radius_worst`
- `perimeter_worst`
- `area_worst`
- `concavity_worst`
- `concave points_worst`

### Target Variable

The target variable represents the tumor diagnosis:

- `0` → Benign
- `1` → Malignant

---

## Exploratory Data Analysis

The following EDA tasks are performed:

- First 5 and last 5 records
- Dataset shape
- Column names
- Data types
- Dataset information
- Missing value analysis
- Duplicate record analysis
- Numerical and categorical variable identification
- Distribution plots
- Box plots
- Scatter plots
- Correlation analysis
- Covariance analysis

---

##  Data Preprocessing

The following preprocessing steps are performed:

1. Remove unnecessary columns such as ID columns.
2. Check for missing values.
3. Remove duplicate records.
4. Select relevant features.
5. Separate features and target variable.
6. Split the dataset into training and testing sets.
7. Apply feature scaling where required.

---

##  Machine Learning Models

Five classification algorithms are implemented:

### 1. Logistic Regression

Used as a baseline classification model for predicting Benign and Malignant tumors.

### 2. Decision Tree

Uses decision rules based on feature values to classify tumors.

### 3. Random Forest

Uses multiple decision trees and combines their predictions to improve classification performance.

### 4. K-Nearest Neighbors (KNN)

Classifies a tumor based on the nearest training data points.

### 5. Support Vector Machine (SVM)

Finds an optimal decision boundary between the two tumor classes.

---

##  Model Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score

The model with the best performance is selected for the final prediction system.

---

##  Saved Model Files

The trained model and preprocessing files are saved using Joblib.

```text
breast_cancer_best_model.pkl
breast_cancer_scaler.pkl
breast_cancer_features.pkl
