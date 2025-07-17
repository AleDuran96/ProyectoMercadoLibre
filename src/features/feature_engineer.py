# src/feature_engineer.py

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def group_category_id(self, threshold=200):
        cat_counts = self.df['category_id'].value_counts()
        valid_cats = cat_counts[cat_counts > threshold].index
        self.df['category_id'] = self.df['category_id'].apply(lambda x: x if x in valid_cats else 'otra_categoria')

    def group_provinces(self, threshold=1000):
        prov_counts = self.df['seller_province'].value_counts()
        valid_provs = prov_counts[prov_counts > threshold].index
        self.df['seller_province'] = self.df['seller_province'].apply(lambda x: x if x in valid_provs else 'Otra Provincia')

    def simplify_loyalty(self):
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

    def clean_shipping_mode(self):
        valid_modes = ['me2', 'not_specified', 'custom', 'me1']
        self.df['shipping_mode'] = self.df['shipping_mode'].apply(lambda x: x if x in valid_modes else 'other')

    def clean_shipping_flags(self):
        self.df['shipping_admits_pickup'] = self.df['shipping_admits_pickup'].astype(str)
        self.df['shipping_admits_pickup'] = self.df['shipping_admits_pickup'].apply(
            lambda x: 'True' if 'True' in x else ('False' if 'False' in x else 'unknown'))

        self.df['shipping_is_free'] = self.df['shipping_is_free'].astype(str)
        self.df['shipping_is_free'] = self.df['shipping_is_free'].apply(
            lambda x: 'True' if 'True' in x else ('False' if 'False' in x else 'unknown'))

    def clean_warranty(self):
        self.df['warranty'] = self.df['warranty'].fillna('no_warranty')
        self.df['warranty'] = self.df['warranty'].apply(
            lambda x: 'con_garantia' if isinstance(x, str) and 'sí' in x.lower() else 'sin_garantia')

    def engineer(self):
        self.group_category_id()
        self.group_provinces()
        self.simplify_loyalty()
        self.clean_shipping_mode()
        self.clean_shipping_flags()
        self.clean_warranty()
        return self.df
