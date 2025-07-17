import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, file_path: str):
        """
        Inicializa el analizador de datos con la ruta del archivo.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Lee el dataset y lo guarda en un atributo interno.
        """
        self.df = pd.read_csv(self.file_path)
        print(f"✅ Dataset cargado con {self.df.shape[0]} filas y {self.df.shape[1]} columnas.")
        print(f"Primeras Filas: {self.df.head()}")

    def fix_data_types(self):
        """
        Convierte columnas clave al tipo de dato adecuado:
        - 'date_created' a datetime
        - Flags (booleanos) a bool
        - Precios y cantidades a numéricos
        """
        if self.df is not None:
            print("\n🔧 Corrigiendo tipos de datos...")

            # Convertir fecha a datetime sin zona horaria
            self.df['date_created'] = pd.to_datetime(self.df['date_created'], errors='coerce')
            if self.df['date_created'].dt.tz is not None:
                self.df['date_created'] = self.df['date_created'].dt.tz_localize(None)

            # Convertir columnas numéricas
            numeric_cols = ['base_price', 'price', 'initial_quantity', 'sold_quantity', 'available_quantity']
            for col in numeric_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Convertir flags a booleanos
            bool_cols = ['shipping_admits_pickup', 'shipping_is_free', 'is_new']
            for col in bool_cols:
                self.df[col] = self.df[col].astype('bool')

            print("✅ Tipos de datos corregidos.")
            print(f"Tipos de Datos {self.df.dtypes}")
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def remove_irrelevant_columns(self):
        """
        Elimina columnas irrelevantes para el modelo de predicción de 'sold_quantity'.
        """
        if self.df is not None:
            columnas_a_eliminar = [
                'id', 'title', 'tags', 'attributes', 'pictures', 'variations',
                'seller_id', 'status', 'sub_status', 'seller_country', 'seller_city'
            ]

            existentes = [col for col in columnas_a_eliminar if col in self.df.columns]
            self.df.drop(columns=existentes, inplace=True)
            print(f"\n🗑️ Columnas eliminadas: {existentes}")
        else:
            print("⚠️ Carga el dataset primero con load_data().")


    def summary_statistics(self):
        """
        Muestra resumen estadístico de columnas numéricas relevantes.
        """
        if self.df is not None:
            print("\n📊 Resumen estadístico de precios y cantidades vendidas:")
            print(self.df[['base_price', 'price', 'sold_quantity']].describe())
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def missing_values_report(self):
        """
        Muestra columnas con valores faltantes y su proporción.
        """
        if self.df is not None:
            print("\n🧩 Reporte de valores faltantes:")
            missing = self.df.isnull().sum()
            missing_percent = (missing / len(self.df)) * 100
            report = pd.DataFrame({
                'Missing Values': missing,
                'Percentage': missing_percent
            })
            report = report[report['Missing Values'] > 0]
            print(report.sort_values(by='Percentage', ascending=False))
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def handle_missing_values(self):
        """
        Ejemplo básico: elimina filas con precios nulos. Puedes expandir según tu análisis.
        """
        if self.df is not None:
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=['price', 'base_price'])
            print(f"\n🧹 Filas eliminadas por precios nulos: {initial_rows - len(self.df)}")
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def check_duplicates(self):
        """
        Muestra cantidad de duplicados exactos (en todas las columnas).
        """
        if self.df is not None:
            total_duplicates = self.df.duplicated().sum()
            print(f"\n🔍 Duplicados encontrados: {total_duplicates}")
            return self.df[self.df.duplicated()]
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def remove_duplicates(self):
        """
        Elimina duplicados exactos del dataset.
        """
        if self.df is not None:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)
            print(f"\n🧼 Duplicados eliminados: {before - after}")
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def check_inconsistencies(self):
        """
        Busca datos sospechosos como:
        - Precios negativos
        - base_price < price
        - Fechas antiguas (más de 10 años)
        - Cantidades negativas
        """
        if self.df is not None:
            inconsistencias = []

            # Precios negativos
            negativos = self.df[(self.df['price'] < 0) | (self.df['base_price'] < 0)]
            if not negativos.empty:
                inconsistencias.append(f"🔻 Precios negativos: {len(negativos)}")

            # base_price < price
            precios_invertidos = self.df[self.df['base_price'] < self.df['price']]
            if not precios_invertidos.empty:
                inconsistencias.append(f"📉 base_price menor a price: {len(precios_invertidos)}")

            # Fechas muy antiguas
            self.df['date_created'] = pd.to_datetime(self.df['date_created'], errors='coerce')
            fecha_corte = pd.Timestamp.now() - pd.DateOffset(years=10)
            fechas_invalidas = self.df[self.df['date_created'] < fecha_corte]
            if not fechas_invalidas.empty:
                inconsistencias.append(f"📅 Fechas de creación >10 años atrás: {len(fechas_invalidas)}")

            # Cantidades negativas
            cantidades_negativas = self.df[
                (self.df['sold_quantity'] < 0) | (self.df['initial_quantity'] < 0)
            ]
            if not cantidades_negativas.empty:
                inconsistencias.append(f"📦 Cantidades negativas: {len(cantidades_negativas)}")

            print("\n🔎 Inconsistencias detectadas:")
            if inconsistencias:
                for i in inconsistencias:
                    print(f"  - {i}")
            else:
                print("  ✅ No se encontraron inconsistencias.")
        else:
            print("⚠️ Carga el dataset primero con load_data().")

    def detect_price_outliers(self, method="zscore", threshold=3):
        """
        Detecta outliers en la columna 'price' con z-score o IQR.
        """
        if self.df is not None:
            prices = self.df['price']

            if method == "zscore":
                z_scores = np.abs(stats.zscore(prices))
                outliers = self.df[z_scores > threshold]
                print(f"\n🚨 Outliers detectados por Z-Score (umbral={threshold}): {len(outliers)}")

            elif method == "iqr":
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(prices < lower_bound) | (prices > upper_bound)]
                print(f"\n🚨 Outliers detectados por IQR: {len(outliers)}")

            else:
                print("⚠️ Método no soportado. Usa 'zscore' o 'iqr'.")

            return outliers

        else:
            print("⚠️ Carga el dataset primero con load_data().")
            return None

    def clean_data(self):
        self.load_data()
        self.fix_data_types()
        self.remove_irrelevant_columns()
        self.missing_values_report()
        self.handle_missing_values()
        self.check_duplicates()
        self.remove_duplicates()
        self.check_inconsistencies()
        self.summary_statistics()
        self.detect_price_outliers(method="zscore")

        return self.df
