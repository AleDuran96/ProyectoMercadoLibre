from dotenv import load_dotenv
from pathlib import Path
import os

# Cargar variables desde el archivo .env
load_dotenv()

# Obtener la ruta de los datos
DATA_PATH = Path(os.getenv("DATA_PATH"))
DATA_PATH_COMPLETE = Path(os.getenv("DATA_PATH_COMPLETE"))
