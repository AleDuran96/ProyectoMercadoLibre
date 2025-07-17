import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()

    def group_category_id(self, threshold=200):
        print("🔢 Agrupando categorías con pocas ocurrencias...")
        cat_counts = self.df['category_id'].value_counts()
        valid_cats = cat_counts[cat_counts > threshold].index
        self.df['category_id'] = self.df['category_id'].apply(lambda x: x if x in valid_cats else 'otra_categoria')
        print("✅ Categorías agrupadas.")

    def group_provinces(self, threshold=1000):
        print("📍 Agrupando provincias poco frecuentes...")
        prov_counts = self.df['seller_province'].value_counts()
        valid_provs = prov_counts[prov_counts > threshold].index
        self.df['seller_province'] = self.df['seller_province'].apply(lambda x: x if x in valid_provs else 'Otra Provincia')
        print("✅ Provincias agrupadas.")

    def simplify_loyalty(self):
        print("🎖️ Simplificando niveles de lealtad del vendedor...")
        self.df['seller_loyalty'] = self.df['seller_loyalty'].replace({
            'gold_special': 'gold',
            'gold': 'gold',
            'gold_premium': 'gold',
            'gold_pro': 'gold',
            '70.0': 'gold',
            '80.0': 'gold',
            '100.0': 'gold',
            '5500.0': 'gold'
        })
        print("✅ Lealtad simplificada.")

    def clean_shipping_mode(self):
        print("🚚 Limpiando modos de envío no válidos...")
        valid_modes = ['me2', 'not_specified', 'custom', 'me1']
        self.df['shipping_mode'] = self.df['shipping_mode'].apply(lambda x: x if x in valid_modes else 'other')
        print("✅ Modos de envío limpiados.")

    def create_discount_features(self):
        print("💸 Creando variables de descuento...")
        self.df['discount_pct'] = (
            (self.df['base_price'] - self.df['price']) / self.df['base_price']
        ).replace([np.inf, -np.inf], 0).clip(lower=0).fillna(0)
        self.df['is_discounted'] = self.df['price'] < self.df['base_price']
        print("✅ Variables de descuento creadas.")

    def create_temporal_features(self):
        print("📅 Generando variables temporales...")
        self.df['days_since_created'] = (pd.Timestamp.now() - self.df['date_created']).dt.days
        self.df['month'] = self.df['date_created'].dt.month
        self.df['weekday'] = self.df['date_created'].dt.weekday
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6])
        print("✅ Variables temporales generadas.")

    def tag_promotion_and_holidays(self):
        print("🎯 Etiquetando días de promoción y temporadas festivas...")

        # Normaliza fechas para comparar solo YYYY-MM-DD
        dates = self.df['date_created'].dt.normalize()

        # --- Promoción ---
        promo_dates = pd.to_datetime([
            "2014-05-15", "2015-05-11",  # Hot Sale
            "2013-11-11", "2014-11-10", "2015-11-02",  # CyberMonday
            "2013-11-29", "2014-11-28", "2015-11-27"   # Black Friday
        ]).normalize()

        self.df['is_promotion_day'] = dates.isin(promo_dates)
        print("✅ Promotion Day aplicado.")

        # --- Holiday Season ---
        # Navidad / Año Nuevo
        is_navidad = (dates.dt.month == 12) | ((dates.dt.month == 1) & (dates.dt.day <= 6))

        # Invierno (julio en Argentina)
        is_invierno = dates.dt.month == 7

        # Semana Santa (rango precalculado)
        easter_ranges = {
            2013: pd.date_range("2013-03-24", "2013-03-31"),
            2014: pd.date_range("2014-04-13", "2014-04-20"),
            2015: pd.date_range("2015-03-29", "2015-04-05")
        }

        easter_days = pd.DatetimeIndex([])  # empieza vacío
        for days in easter_ranges.values():
            easter_days = easter_days.union(days)

        is_semana_santa = dates.isin(easter_days)

        # Combina todo
        self.df['is_holiday_season'] = is_navidad | is_invierno | is_semana_santa
        print("✅ Holiday Season aplicado.")
        print("🎉 Etiquetas de promociones y festivos agregadas.")

    def normalize_numerical_features(self):
        print("📏 Normalizando variables numéricas...")
        numeric_cols = ['base_price', 'price', 'initial_quantity', 'available_quantity',
                        'discount_pct', 'days_since_created']
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

    def prepare_categoricals(self):
        print("🔢 Codificando variables categóricas...")
        categorical_cols = ['buying_mode', 'shipping_mode', 'category_id', 'seller_province']
        for col in categorical_cols:
            self.df[col] = self.df[col].astype('category')

    def convert_booleans_to_int(self):
        print("🔁 Convirtiendo booleanos a enteros...")
        bool_cols = ['shipping_admits_pickup', 'shipping_is_free', 'is_new','is_discounted', 'is_weekend', 'is_promotion_day', 'is_holiday_season']
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)


    def encode_cyclic_features(self):
        """
        Transforma las variables cíclicas 'month' (1-12) y 'weekday' (0-6) 
        en componentes seno y coseno para representar su naturaleza circular.
        """
        print("🔄 Codificando variables cíclicas (month y weekday)...")

        # Verifica que las columnas existan
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        if 'weekday' in self.df.columns:
            self.df['weekday_sin'] = np.sin(2 * np.pi * self.df['weekday'] / 7)
            self.df['weekday_cos'] = np.cos(2 * np.pi * self.df['weekday'] / 7)

        print("✅ Codificación cíclica completada.")

    def drop_unused_columns(self):
        """
        Elimina columnas intermedias que ya no son necesarias después del feature engineering.
        """
        print("🧹 Eliminando columnas que ya no se requieren...")

        columns_to_drop = [
            'month',
            'weekday',
            'date_created',  # opcional: si ya sacaste toda la info temporal
        ]

        # Solo elimina las columnas que sí existen en el DataFrame
        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], inplace=True)

        print("✅ Columnas innecesarias eliminadas.")


    def engineer(self):
        print("\n⚙️ Iniciando Feature Engineering...\n")
        self.group_category_id()
        self.group_provinces()
        self.simplify_loyalty()
        self.clean_shipping_mode()
        self.create_discount_features()
        self.create_temporal_features()
        self.tag_promotion_and_holidays()
        self.normalize_numerical_features()
        self.prepare_categoricals()
        self.encode_cyclic_features()
        self.convert_booleans_to_int()
        self.drop_unused_columns()
        print("\n✅ Feature Engineering completado.\n")
        return self.df
