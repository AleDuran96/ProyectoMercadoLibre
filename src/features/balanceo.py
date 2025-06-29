import pandas as pd
from imblearn.over_sampling import SMOTE

def preprocess_data(df: pd.DataFrame):
    
    # Separar X / y
    X = df.drop("y", axis=1)
    y = df["y"]

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled