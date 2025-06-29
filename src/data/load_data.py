import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    """Carga un archivo CSV con los datos del banco."""
    return pd.read_csv(Path(path), delimiter=";")

def codificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()

    # Codificar variables binarias (yes/no)
    binarias = ["default", "housing", "loan", "y"]
    for col in binarias:
        df_encoded[col] = df_encoded[col].map({"yes": 1, "no": 0})

    # Ordinal - education
    edu_map = {
        "unknown": -1,  # o podrías usar imputación
        "primary": 0,
        "secondary": 1,
        "tertiary": 2
    }
    df_encoded["education"] = df_encoded["education"].map(edu_map)

    # Ordinal - month
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    df_encoded["month"] = df_encoded["month"].map(month_map)

    # Resto de categóricas a dummies
    categ_dummies = ["job", "marital", "contact", "poutcome"]
    df_encoded = pd.get_dummies(df_encoded, columns=categ_dummies, drop_first=True)
    
    # Convertir booleanos a enteros
    df_encoded = df_encoded.astype(int)
    
    return df_encoded
