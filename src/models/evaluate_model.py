from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def evaluar_modelo_pre(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import numpy as np

def evaluar_modelo(modelo, X_test, y_test, threshold=0.3):
    # Validar si el modelo tiene predict_proba
    if hasattr(modelo, "predict_proba"):
        y_scores = modelo.predict_proba(X_test)[:, 1]  # Probabilidades clase positiva (1)
        y_pred = (y_scores >= threshold).astype(int)
    else:
        # fallback si el modelo no tiene predict_proba
        y_pred = modelo.predict(X_test)
        y_scores = None  # opcional, por si quieres evitar calcular AUC

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    if y_scores is not None:
        metrics["auc"] = roc_auc_score(y_test, y_scores)

    return metrics

def plot_roc_curve(model, X_test, y_test):
    # Obtener probabilidades (la clase positiva está en [:, 1])
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calcular FPR y TPR
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Graficar curva ROC
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
