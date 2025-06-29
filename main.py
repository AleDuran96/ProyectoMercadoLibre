# Obtener la ruta de los datos
from src.data.load_data import load_data, codificar_variables
from src.config import DATA_PATH_BANK
import pandas as pd

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
   df_encoded = codificar_variables(df)

   print(df_encoded.head())

if __name__ == "__main__":
    main()



