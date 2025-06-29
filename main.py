# Obtener la ruta de los datos
from src.data.load_data import load_data
from src.config import DATA_PATH_BANK

def main():
    df = load_data(DATA_PATH_BANK)
    print(df.head())

if __name__ == "__main__":
    main()



