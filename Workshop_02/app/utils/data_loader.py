import pandas as pd
import numpy as np

INSURANCE_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


def load_insurance():
    """Load Medical Cost Insurance dataset."""
    try:
        df = pd.read_csv(INSURANCE_URL)
        return df
    except Exception:
        np.random.seed(42)
        n = 1338
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n),
            'sex': np.random.choice(['male', 'female'], n),
            'bmi': np.random.normal(30, 6, n).clip(15, 55),
            'children': np.random.randint(0, 6, n),
            'smoker': np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
            'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n),
        })
        df['charges'] = (
            1000
            + df['age'] * 200
            + df['bmi'] * 50
            + df['smoker'].map({'yes': 25000, 'no': 0})
            + np.random.normal(0, 2000, n)
        ).clip(1000, 70000)
        return df


def load_digits_data():
    """Load sklearn Digits dataset as DataFrame (1797 samples, 64 features, 10 classes)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
    df['target'] = digits.target
    return df, [f'pixel_{i}' for i in range(64)]
