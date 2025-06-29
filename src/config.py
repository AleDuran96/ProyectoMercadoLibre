from dotenv import load_dotenv
from pathlib import Path
import os

# Cargar variables desde el archivo .env
load_dotenv()

# Obtener la ruta de los datos
DATA_PATH_BANK = Path(os.getenv("DATA_PATH_BANK"))
