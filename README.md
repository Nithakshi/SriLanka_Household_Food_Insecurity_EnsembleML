# Predicting Household Food Insecurity in Sri Lanka Using Ensemble Machine Learning

## Project Overview

This research project investigates household food insecurity in Sri Lanka using ensemble machine learning classification techniques.

The project includes data preprocessing, model training, performance evaluation, feature importance analysis, and an interactive Streamlit dashboard.

## Machine Learning Models

1. Decision Tree
2. Bagging
3. Random Forest
4. AdaBoost
5. XGBoost
6. Stacking
7. Random Forest + Artificial Neural Network (RF + ANN Hybrid)

## Evaluation Metrics

Models are evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrices.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Jupyter Notebook

## Project Files

- `data_preprocessing.py` — Data cleaning and preparation
- `train_models.py` — Training seven machine learning models
- `model_evaluation.py` — Model performance evaluation
- `feature_importance.py` — Feature importance analysis
- `app.py` — Interactive Streamlit dashboard
- `Ensemble_Learning_Analysis.ipynb` — Research analysis notebook

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

## Dataset

The project uses household survey data from Sri Lanka. Raw survey files are not included in the public repository unless redistribution is permitted.

## Research Purpose

To compare ensemble learning techniques for predicting household food insecurity and analyze their effectiveness using classification performance metrics.