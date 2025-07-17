import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class ModelPredictor:
    def __init__(self, df, target='sold_quantity'):
        self.df = df
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]

        self.df_xgboost = pd.get_dummies(df.copy(), drop_first=True)  # solo para XGBoost
        self.X_xgb = self.df_xgboost.drop(columns=[target])

        self.results = {}

    def split_data(self, test_size=0.3, random_state=42):
        print("✂️ Dividiendo datos en entrenamiento y prueba...")
        # Para CatBoost / LightGBM
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

        # Para XGBoost
        self.X_train_xgb, self.X_test_xgb, self.y_train_xgb, self.y_test_xgb = train_test_split(
            self.X_xgb, self.y, test_size=test_size, random_state=random_state)

        print("✅ División completada.")

    def evaluate_model(self, model_name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        self.results[model_name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        print(f"📊 Resultados para {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}")
        return self.results[model_name]

    def train_xgboost(self, param_grid=None):
        print("🚀 Entrenando XGBoost...")
        model = XGBRegressor(objective='reg:squarederror', random_state=42)

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.01]
            }

        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
        grid.fit(self.X_train_xgb, self.y_train_xgb)
        best_model = grid.best_estimator_

        print(f"🏆 Mejor modelo XGBoost: {grid.best_params_}")
        preds = best_model.predict(self.X_test_xgb)
        return self.evaluate_model('XGBoost', self.y_test_xgb, preds)

    def train_lightgbm(self, param_grid=None):
        print("💡 Entrenando LightGBM...")
        model = LGBMRegressor(random_state=42)

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [-1, 6],
                'learning_rate': [0.1, 0.01]
            }

        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
        grid.fit(self.X_train, self.y_train)
        best_model = grid.best_estimator_

        print(f"🏆 Mejor modelo LightGBM: {grid.best_params_}")
        preds = best_model.predict(self.X_test)
        return self.evaluate_model('LightGBM', self.y_test, preds)

    def train_catboost(self, param_grid=None):
        print("😺 Entrenando CatBoost...")
        model = CatBoostRegressor(verbose=0, random_state=42)

        if param_grid is None:
            param_grid = {
                'depth': [4],
                'learning_rate': [0.1],
                'iterations': [100]
            }

        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
        cat_features = self.X_train.select_dtypes(include='category').columns.tolist()
        grid.fit(self.X_train, self.y_train, **{'cat_features': cat_features})
        best_model = grid.best_estimator_

        print(f"🏆 Mejor modelo CatBoost: {grid.best_params_}")
        preds = best_model.predict(self.X_test)
        return self.evaluate_model('CatBoost', self.y_test, preds)

    def get_results(self):
        return pd.DataFrame(self.results).T.sort_values('RMSE')
