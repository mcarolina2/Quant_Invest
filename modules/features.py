import pandas as pd
import numpy as np


def build_features(returns: pd.Series, horizon: int = 1) -> pd.DataFrame:
    df = pd.DataFrame({"ret": returns})

    # --- Lagged returns ---
    for lag in range(1, 11):
        df[f"ret_lag{lag}"] = df["ret"].shift(lag)

    # --- Rolling statistics ---
    for w in [5, 10, 21]:
        df[f"mean_{w}d"] = df["ret"].shift(1).rolling(w).mean()
        df[f"std_{w}d"] = df["ret"].shift(1).rolling(w).std()
        df[f"skew_{w}d"] = df["ret"].shift(1).rolling(w).skew()

    # --- Realized volatility ---
    df["rvol_5d"] = df["ret"].shift(1).rolling(5).std() * np.sqrt(252)
    df["rvol_21d"] = df["ret"].shift(1).rolling(21).std() * np.sqrt(252)

    # --- Momentum ---
    df["mom_21d"] = df["ret"].shift(1).rolling(21).sum()
    df["mom_63d"] = df["ret"].shift(1).rolling(63).sum()

    # --- RSI (14-day) ---
    delta = df["ret"].shift(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    # --- Target: forward return ---
    df["target"] = df["ret"].shift(-horizon).rolling(horizon).sum()

    df = df.drop(columns=["ret"]).dropna()
    return df


def train_test_split_ts(df: pd.DataFrame, test_ratio: float = 0.2):
    split = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split]
    test = df.iloc[split:]
    feature_cols = [c for c in df.columns if c != "target"]
    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]
    return X_train, y_train, X_test, y_test, feature_cols
