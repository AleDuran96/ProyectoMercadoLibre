from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_auc_score

# Scores
scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}

def entrenar_modelos(X_train, y_train, cv=10):
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=1000, random_state=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=100),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        cv_result = cross_validate(modelo, X_train, y_train, cv=cv, scoring=scoring)
        resultados[nombre] = {
            "precision_mean": cv_result['test_precision'].mean(),
            "recall_mean": cv_result['test_recall'].mean(),
            "roc_auc_mean": cv_result['test_roc_auc'].mean()
        }

    return resultados, modelos

def entrenar_final(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)
    return modelo

def entrenar_xgboost_final(X_train, y_train):
    modelo = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=100)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    }

    #param_grid = {
    #    "n_estimators": [100, 300, 500],
    #    "max_depth": [3, 5, 7, 10],
    #    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    #    "subsample": [0.8, 1.0],
    #    "colsample_bytree": [0.6, 0.8, 1.0]
    #}

    grid = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        scoring='recall',  # 'recall' o 'precision' o 'f1'
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print("Mejores parámetros para XGBoost:", grid.best_params_)

    return grid.best_estimator_

def entrenar_catboost_final(X_train, y_train):
    modelo = CatBoostClassifier(verbose=0, random_state=42)

    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1]
    }

    #param_grid = {
    #    "depth": [4, 6, 8, 10],
    #    "learning_rate": [0.01, 0.05, 0.1],
    #    "l2_leaf_reg": [1, 3, 5, 7],
    #    "iterations": [300, 500]
    #}

    grid = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        scoring='recall',  #'recall' o 'precision' o 'f1'
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print("Mejores parámetros para CatBoost:", grid.best_params_)

    return grid.best_estimator_
