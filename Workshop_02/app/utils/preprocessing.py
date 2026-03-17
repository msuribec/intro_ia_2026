import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_insurance(df):
    """Preprocess insurance dataset for regression."""
    df = df.copy()
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    df['smoker'] = (df['smoker'] == 'yes').astype(int)
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    X = df.drop('charges', axis=1)
    y = df['charges']
    return X, y, list(X.columns)


def preprocess_digits(df, feature_names):
    """Preprocess Digits dataset. Already numerical, no missing values."""
    X = df[feature_names].copy()
    y = df['target'].copy()
    return X, y, list(feature_names)
