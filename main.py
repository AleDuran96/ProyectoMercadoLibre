# Obtener la ruta de los datos
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PATH, DATA_PATH_COMPLETE
import pandas as pd
from src.data.data_analyzer import DataAnalyzer
from src.features.feature_engineer import FeatureEngineer
from src.models.model_predictor import ModelPredictor


def main():
   # Paso 1: EDA
   analyzer = DataAnalyzer(DATA_PATH)
   df_clean = analyzer.clean_data()

   print(df_clean["shipping_mode"].value_counts())

   # Paso 2: Feature Engineer
   fe = FeatureEngineer(df_clean)
   df_features = fe.engineer()
   # Guardar el dataframe procesado
   df_features.to_csv(r"C:\Users\aleja\Documents\Alejandro Duran Carpeta\Proyectos Data Science\Prueba Técnica Mercado Libre\data\processed\df_features.csv", index=False)

   # Paso 3: Modelado
   print("🤖 Entrenando modelos...")
   modeler = ModelPredictor(df_features, target='sold_quantity')
   modeler.split_data()
   # Entrenamiento de modelos con Optimización de Hiperparametros y validación cruzada
   modeler.train_xgboost()
   modeler.train_lightgbm()
   modeler.train_catboost()


if __name__ == "__main__":
   main()



