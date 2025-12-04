import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# SAFE SIMPLE FUNCTIONS (Namespace: simple_)
# ============================================================

def simple_load_data(ticker, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df.index = pd.to_datetime(df.index)
    return df


def simple_clean_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regularize calendar, forward-fill market gaps (weekends, holidays).
    """
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    # OHLC only → forward then backward fill
    df = df.ffill().bfill()

    return df


def simple_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic OHLC-compatible indicators:
      - returns
      - volatility
      - moving averages
      - price ratio
      - candle body
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    # --- price returns ---
    df["simp_ret_1d"] = close.pct_change(1)
    df["simp_ret_5d"] = close.pct_change(5)

    # --- moving average ---
    df["simp_ma_10"] = close.rolling(10).mean()

    # --- volatility ---
    df["simp_vol_10"] = close.rolling(10).std()

    # --- price / MA ---
    df["simp_price_over_ma10"] = close / df["simp_ma_10"]

    # --- candle body size ---
    df["simp_candle_body"] = close - open_

    # --- daily range ---
    df["simp_range"] = high - low

    # remove initial NaNs from rolling windows
    return df.dropna()


def simple_add_target(df: pd.DataFrame, target_col: str, horizon: int):
    """
    Future value of Close(t + horizon)
    """
    df = df.copy()
    df["simp_target_future"] = df[target_col].shift(-horizon)
    return df.dropna(subset=["simp_target_future"])


def simple_split(df: pd.DataFrame, test_size=0.2):
    n = len(df)
    idx = int(n * (1 - test_size))
    return df.iloc[:idx], df.iloc[idx:]


def simple_get_xy(df: pd.DataFrame, target_col="simp_target_future"):
    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def simple_build_xy_xgb(df, target_col="simp_target_future"):
    return df.drop(columns=[target_col]), df[target_col]


def simple_ts_splits(X: pd.DataFrame, y: pd.Series, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for tr, te in tscv.split(X):
        out.append({
            "simp_X_train": X.iloc[tr],
            "simp_X_test": X.iloc[te],
            "simp_y_train": y.iloc[tr],
            "simp_y_test": y.iloc[te],
        })
    return out


def simple_create_rnn_xy(data: np.ndarray | pd.DataFrame, seq_len: int, horizon: int):
    arr = np.array(data)
    X, y = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len : i+seq_len+horizon])
    return np.array(X), np.array(y)


def simple_rnn_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    seq_len: int,
    horizon: int,
    test_size: float = 0.2,
):
    idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:idx], X.iloc[idx:]
    y_train, y_test = y.iloc[:idx], y.iloc[idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_seq, y_train_seq = simple_create_rnn_xy(X_train_s, seq_len, horizon)
    X_test_seq, y_test_seq = simple_create_rnn_xy(X_test_s, seq_len, horizon)

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler


# ============================================================
# FINAL SIMPLE DATASET BUILDER (OHLC compatible)
# ============================================================

def simple_prepare_dataset(
    ticker: str,
    start: str,
    end: str,
    horizon: int,
    for_rnn=False,
    seq_len=20,
):
    df = simple_load_data(ticker, start, end)
    df = simple_clean_stock(df)
    df = simple_add_features(df)
    df = simple_add_target(df, "Close", horizon)

    train_df, test_df = simple_split(df)

    if not for_rnn:
        simp_X_train, simp_y_train = simple_build_xy_xgb(train_df)
        simp_X_test, simp_y_test = simple_build_xy_xgb(test_df)
        return simp_X_train, simp_y_train, simp_X_test, simp_y_test

    else:
        X_train, y_train = simple_get_xy(train_df)
        X_test, y_test = simple_get_xy(test_df)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        X_train_seq, y_train_seq = simple_create_rnn_xy(X_train_s, seq_len, horizon)
        X_test_seq, y_test_seq = simple_create_rnn_xy(X_test_s, seq_len, horizon)

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler
