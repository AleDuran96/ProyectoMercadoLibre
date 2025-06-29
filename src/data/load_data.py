import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    """Carga un archivo CSV con los datos del banco."""
    return pd.read_csv(Path(path), delimiter=";")
