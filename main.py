# Obtener la ruta de los datos
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PATH, DATA_PATH_COMPLETE
import pandas as pd
from src.features.balanceo import preprocess_data, split_data
from src.models.train_models import entrenar_modelos, entrenar_final, entrenar_xgboost_final, entrenar_catboost_final
from src.models.evaluate_model import evaluar_modelo, plot_roc_curve
# main.py

from src.data.data_analyzer import DataAnalyzer

def main():
   # Paso 1: EDA
   analyzer = DataAnalyzer(DATA_PATH)
   df_clean = analyzer.clean_data()

   print(df_clean.shape)
   # Usar los métodos definidos


if __name__ == "__main__":
   main()



