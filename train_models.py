
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'
SEED = 42


def make_preprocessor(numeric_columns, categorical_columns):
    steps = []
    if numeric_columns:
        steps.append(('numeric', SimpleImputer(strategy='constant', fill_value=0), numeric_columns))
    if categorical_columns:
        steps.append(('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns))
    return ColumnTransformer(steps, remainder='drop', sparse_threshold=0)


def make_pipeline(estimator, numeric_columns, categorical_columns):
    return Pipeline([
        ('preprocess', make_preprocessor(numeric_columns, categorical_columns)),
        ('model', estimator),
    ])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    data_path = RESULTS_DIR / 'cleaned_household_data.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'Run data_preprocessing.py first: {data_path}')
    data = pd.read_csv(data_path, low_memory=False)
    if 'food_insecure' not in data:
        raise ValueError('Missing target: food_insecure')
    if data['food_insecure'].isna().any() or not set(data['food_insecure'].unique()).issubset({0, 1}):
        raise ValueError('Target must contain only 0 and 1, with no missing values')
    X = data.drop(columns=['food_insecure']).copy()
    y = data['food_insecure'].astype(int)
    numeric_columns = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    # Mirror the categorical conversion in train_rf_ann_hybrid.py.
    for col in categorical_columns:
        X_train[col] = X_train[col].astype('string').fillna('__MISSING__').astype(str)
        X_test[col] = X_test[col].astype('string').fillna('__MISSING__').astype(str)

    def pipe(estimator):
        return make_pipeline(estimator, numeric_columns, categorical_columns)

    rf = lambda: pipe(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    ))
    models = {
        'Decision Tree': ('decision_tree', pipe(DecisionTreeClassifier(random_state=SEED))),
        'Bagging': ('bagging', pipe(BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=SEED),
            n_estimators=100, random_state=SEED, n_jobs=-1
        ))),
        'Random Forest': ('random_forest', rf()),
        'AdaBoost': ('adaboost', pipe(AdaBoostClassifier(
            n_estimators=100, random_state=SEED
        ))),
        'XGBoost': ('xgboost', pipe(XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            eval_metric='logloss', random_state=SEED, n_jobs=-1
        ))),
        'Stacking': ('stacking', StackingClassifier(
            estimators=[
                ('rf', rf()),
                ('xgb', pipe(XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    eval_metric='logloss', random_state=SEED, n_jobs=-1
                ))),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            stack_method='predict_proba', cv=3, n_jobs=1
        )),
    }
    # RF + ANN stacking hybrid: RF and MLP feed a logistic-regression meta-learner.
    # ANN scaling is applied AFTER the shared feature encoder, inside its pipeline.
    hybrid_rf = rf()
    hybrid_ann = Pipeline([
        ('preprocess', make_preprocessor(numeric_columns, categorical_columns)),
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
            learning_rate_init=0.001, max_iter=250, early_stopping=True,
            random_state=SEED
        )),
    ])
    models['RF + ANN Hybrid'] = ('rf_ann_hybrid', StackingClassifier(
        estimators=[('rf', hybrid_rf), ('ann', hybrid_ann)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        stack_method='predict_proba', cv=3, n_jobs=1,
    ))

    results = []
    predictions = pd.DataFrame({'actual': y_test.to_numpy()})
    for name, (key, model) in models.items():
        print(f'Training {name}...', flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.neural_network')
            model.fit(X_train, y_train)
        predicted = model.predict(X_test).astype(int)
        probability = model.predict_proba(X_test)[:, 1]
        predictions[f'{key}_pred'] = predicted
        predictions[f'{key}_prob'] = probability
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, predicted),
            'Precision': precision_score(y_test, predicted, zero_division=0),
            'Recall': recall_score(y_test, predicted, zero_division=0),
            'F1 Score': f1_score(y_test, predicted, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, probability),
        })
        cm = confusion_matrix(y_test, predicted, labels=[0, 1])
        pd.DataFrame(cm, index=['actual_0', 'actual_1'], columns=['predicted_0', 'predicted_1']).to_csv(
            RESULTS_DIR / f'{key}_confusion_matrix.csv'
        )
        joblib.dump(model, MODELS_DIR / f'{key}.pkl')
    comparison = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
    comparison.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    predictions.to_csv(RESULTS_DIR / 'test_predictions.csv', index=False)
    print('\nModel comparison:\n', comparison.to_string(index=False))
    print('\nSaved all seven models and their test results.')


if __name__ == '__main__':
    main()
