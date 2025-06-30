# Obtener la ruta de los datos
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.load_data import load_data, codificar_variables
from src.config import DATA_PATH_BANK, DATA_PATH_COMPLETE
import pandas as pd
from src.features.balanceo import preprocess_data, split_data
from src.models.train_models import entrenar_modelos, entrenar_final, entrenar_xgboost_final, entrenar_catboost_final
from src.models.evaluate_model import evaluar_modelo, plot_roc_curve

def main():
   df = load_data(DATA_PATH_BANK)

   '''
   El dataframe se analizó en src/Notebooks/exploracion.ipynb
      -> El nombre de las columnas esta bien escrito
      -> Los valores de todas las columnas tienen el tipo de dato adecuado,
         sin embargo se deben codificar para efectos del modelo.
      -> No hay nulos para ningún campo.
      -> Hay balances negativos, pero estos no afectan dado que es válido tener
         balances negativos en cuentas bancarias.
      -> Se gráfican los campos para ver su distribución.
      -> El dataset no esta equilibrado, dado que para la variable Y el valor
         de "yes" tiene 521 registros, y el valor "no" tiene 4000.
   '''

   # Codificación del DataFrame
   '''
   Creación de variables:
   -> binarias
   -> ordinales
   -> dummies
   '''
   df_encoded = codificar_variables(df)

   # 1. Split con 30% para testeo
   X = df_encoded.drop("y", axis=1)
   y = df_encoded["y"]

   X_train, X_test, y_train, y_test = split_data(X, y)

   df_train = X_train.copy()
   df_train['y'] = y_train

   # 2. Balanceo SOLO para dataset de entrenamiento
   '''
   Dado que el dataset esta desbalanceado (11.5% Si, 88.5% No), es necesario
   entrenar el modelo con un dataset equilibrado, usamos SMOTE para balancearlo
   '''
   X_train_bal, y_train_bal = preprocess_data(df=df_train)

   # 3. Entrenar modelo final (XGBoost con Optimización de Hiperparámetros)
   '''
   Se realizaron varias pruebas de modelos:
   -> Regresión Logística
   -> Random Forest
   -> XGBoost
   -> CatBoost

   El modelo con mejores resultados es XGBoost y CatBoost
   '''
   modelo_xgboost = entrenar_xgboost_final(X_train_bal, y_train_bal)

   # 4. Evaluar en test
   '''
   Durante el preprocesamiento y pruebas, pensamos eliminar columnas como
   duration, pdays y poutcome que mostraban poca varianza y podrían generaban
   overfitting. Sin embargo al ver feature_importance y Recall/Precision,
   determinamos que no generan Overfiting

   Considerando que el dataframe general esta desbalanceado, el Accuracy y AUC
   pueden no ser las mejores métricas para evaluar el modelo, además debemos
   considerar el Recall, Precision y F1.
   
   Además encontramos que el umbral que mejor nos ayuda para mejorar el Recall
   sin dejar de lado Precision es de 0.3
   '''
   metricas_xgboost = evaluar_modelo(modelo_xgboost, X_test, y_test)
   print("Métricas Muestra 4K: " + str(metricas_xgboost))

   # 5. Gráfica de Curva ROC y análisis de métricas
   '''
   Resultados
      -> El recall elevado (0.78) indica que el modelo logra identificar
      correctamente la mayoría de los clientes que sí aceptarían el producto,
      lo cual es clave para campañas comerciales.
      -> El precision aceptable (0.31) implica que no se está desperdiciando
      demasiado esfuerzo en clientes que finalmente no aceptarán.
      -> El F1-score (0.548) refleja un equilibrio razonable entre ambos
      -> El AUC de 0.89 indica muy buena capacidad de discriminación del modelo
   '''
   plot_roc_curve(modelo_xgboost, X_test, y_test)

   # 6. Evaluamos el modelo en la muestra completa (40K registros)
   df_complete = load_data(DATA_PATH_COMPLETE)
   df_complete_encoded = codificar_variables(df_complete)
   # Separar X / y
   X = df_complete_encoded.drop("y", axis=1)
   y = df_complete_encoded["y"]
   # Evaluar en test
   metricas_df_complete = evaluar_modelo(modelo_xgboost, X, y)
   print("Métricas Muestra Completa: " + str(metricas_df_complete))

if __name__ == "__main__":
    main()



