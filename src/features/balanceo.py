import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame):
    
    # Separar X / y
    X = df.drop("y", axis=1)
    y = df["y"]

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
