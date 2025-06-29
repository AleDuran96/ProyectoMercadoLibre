from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def entrenar_modelos(X_train, y_train, cv=10):
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=1000, random_state=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=100),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy")
        resultados[nombre] = {
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std()
        }

    return resultados, modelos

def entrenar_final(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)
    return modelo
