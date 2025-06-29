from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def entrenar_modelos(X_train, y_train, cv=10):
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
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
