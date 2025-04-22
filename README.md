# Tuberculosis Prediction Using Tabular Data and Machine Learning

## Overview

This project predicts the presence of Tuberculosis (TB) using clinical and diagnostic tabular data. It leverages machine learning to assist in early diagnosis and uses modern ML tooling (DVC, MLflow) to ensure reproducibility, experiment tracking, and model versioning.

## Key Features

- Binary classification to detect TB-positive vs. TB-negative cases
- In-depth Exploratory Data Analysis (EDA)
- Multiple ML algorithms: Logistic Regression, Random Forest, Support Vector Machine
- Model evaluation using accuracy, precision, recall, F1-score
- Hyperparameter tuning with RandomizedSearchCV
- Experiment tracking using **MLflow**
- Data & model versioning using **DVC**
- Final model saved and ready for deployment

## Tech Stack

- **Python**: Main programming language
- **Pandas, NumPy**: Data wrangling
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn: ML models and evaluation
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data and model version control
- **Pickle**: Model persistence

## Key Steps

1. **Data Management**
   - Store and version data using DVC
   - Load tabular data, clean missing values, encode features

2. **Exploratory Data Analysis (EDA)**
   - Understand feature distributions, class imbalance, correlations
   - Visualize key patterns and relationships

3. **Model Training**
   - Train and compare Logistic Regression, Random Forest, and Support Vector Machine
   - Use cross-validation and RandomizedSearchCV for tuning
   - Track experiments and metrics in MLflow

4. **Evaluation & Selection**
   - Evaluate models using classification report, confusion matrix

5. **Versioning & Reproducibility**
   - Use DVC to version datasets and trained models
   - Maintain reproducible pipelines

6. **Saving & Deployment**
   - Save final model with `pickle`
   - Optionally register the model in MLflow for future deployment

## Project Limitations

- Model not clinically validated on real-world healthcare systems
- Dataset size may limit generalization
- No explainability modules yet (e.g., SHAP, LIME)
- Class imbalance handling may require more refinement

## Conclusion

This project showcases a reproducible and well-tracked machine learning pipeline for TB prediction using tabular health records. With integration of MLflow and DVC, it emphasizes transparency and reproducibility—core values for any robust ML project. While the model shows strong performance on test data, further clinical testing is required for real-world applications.
