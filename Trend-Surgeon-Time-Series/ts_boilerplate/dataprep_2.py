import numpy as np
import pandas as pd
import yfinance as yf
import holidays
from tqdm.auto import tqdm
from hmmlearn.hmm import GaussianHMM
import logging
import os
# ============================================================
# GLOBAL CONFIG
# ============================================================

feature_registry = {}   # Must always mirror df columns
TARGET_TICKER = "PPH"
SUPPORT_TICKERS = ["XPH"]
START_DATE = "2014-02-04"
END_DATE = "2025-08-03"
HORIZON = 30
USE_HMM = True
MIN_HISTORY = 20
FILLER = 99999   # sentinel for missing macro distances
PATH = "../docs/feature_documentation.md"
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION + REGISTRY HELPERS
# ============================================================

def register_feature(name, shift):
    global feature_registry
    feature_registry[name] = shift


def validate_feature(df, name):
    assert name in df.columns, f"{name} missing from df"
    assert len(df[name]) == len(df.index), f"{name} wrong length"
    assert not df[name].isna().all(), f"{name} all NaN"
    assert not np.isinf(df[name]).any(), f"{name} contains inf"
    assert name in feature_registry, f"{name} missing in registry"
    assert set(feature_registry.keys()).issubset(df.columns), "Registry has extra items"


# ============================================================
# SAFE WINDOW
# ============================================================

def compute_safe_window(start_date, end_date, min_history, horizon):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    safe_start = start - pd.Timedelta(days=min_history + 50)
    safe_end   = end   + pd.Timedelta(days=horizon + 50)
    return safe_start, safe_end


# ============================================================
# RAW OHLCV DOWNLOAD + CLEAN
# ============================================================

def download_prices(target, support_tickers, start_date, end_date):
    safe_start, safe_end = compute_safe_window(start_date, end_date, MIN_HISTORY, HORIZON)

    tickers = [target] + list(support_tickers)
    raw = yf.download(
        tickers=tickers,
        start=safe_start.strftime("%Y-%m-%d"),
        end=safe_end .strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    raw.columns = [f"{t}_{f}" for t, f in raw.columns]
    raw = raw[[c for c in raw.columns if "Adj" not in c]]
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().copy()
    return raw


def clean_raw(df, tickers):
    df = df.copy()

    # Identify OHLCV cols
    ohlcv = []
    for t in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            col = f"{t}_{field}"
            if col in df.columns:
                ohlcv.append(col)

    df[ohlcv] = df[ohlcv].replace([np.inf, -np.inf], np.nan)
    df[ohlcv] = df[ohlcv].ffill()
    df = df.dropna(subset=ohlcv, how="any")

    for col in ohlcv:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# REGISTRY INITIALIZATION
# ============================================================

def initialize_feature_registry(df, target, support):
    global feature_registry
    feature_registry = {}

    tickers = [target] + support

    # add raw OHLCV
    for t in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            name = f"{t}_{field}"
            register_feature(name, "no_shift")
            validate_feature(df, name)

    # canonical target_close
    df["target_close"] = df[f"{target}_Close"]
    register_feature("target_close", "no_shift")
    validate_feature(df, "target_close")

    return df


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def add_return(df, ticker):
    close = f"{ticker}_Close"
    name = f"{ticker}_Return_1d"
    df[name] = df[close].pct_change()
    register_feature(name, "shift_1")
    validate_feature(df, name)


def add_sma(df, ticker):
    close = f"{ticker}_Close"
    name = f"{ticker}_SMA_10"
    df[name] = df[close].rolling(10).mean()
    register_feature(name, "shift_1")
    validate_feature(df, name)


def add_rsi(df, ticker):
    close = f"{ticker}_Close"
    delta = df[close].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    name = f"{ticker}_RSI_14"
    df[name] = 100 - (100 / (1 + rs))
    register_feature(name, "shift_1")
    validate_feature(df, name)


def add_stoch(df, ticker):
    close = f"{ticker}_Close"
    low14 = df[close].rolling(14).min()
    high14 = df[close].rolling(14).max()

    kname = f"{ticker}_StochK"
    df[kname] = 100 * (df[close] - low14) / (high14 - low14)
    register_feature(kname, "shift_1")
    validate_feature(df, kname)

    dname = f"{ticker}_StochD"
    df[dname] = df[kname].rolling(3).mean()
    register_feature(dname, "shift_1")
    validate_feature(df, dname)


def add_entropy(df, ticker):
    ret = f"{ticker}_Return_1d"
    name = f"{ticker}_Entropy_20"
    df[name] = (df[ret] ** 2).rolling(20).sum()
    register_feature(name, "shift_1")
    validate_feature(df, name)


def add_hmm(df, ticker):
    ret = f"{ticker}_Return_1d"
    name = f"{ticker}_HMM"

    clean = df[ret].dropna()
    if len(clean) < 50:
        df[name] = np.nan
    else:
        model = GaussianHMM(n_components=3, n_iter=200)
        states = model.fit(clean.values.reshape(-1, 1)).predict(clean.values.reshape(-1, 1))
        pad = len(df) - len(states)
        df[name] = np.concatenate([np.full(pad, np.nan), states])

    register_feature(name, "shift_1")
    validate_feature(df, name)


def add_all_technicals(df, tickers, use_hmm=True):
    for t in tickers:
        add_return(df, t)
        add_sma(df, t)
        add_rsi(df, t)
        add_stoch(df, t)
        add_entropy(df, t)
        if use_hmm:
            add_hmm(df, t)
    return df


# ============================================================
# CROSS-TICKER FEATURES
# ============================================================

def generate_cross_ticker_features(df, target, support):
    tgt_close = f"{target}_Close"
    tgt_ret20 = f"{target}_Return_1d"  # simplified

    for s in support:
        sc = f"{s}_Close"

        name = f"{s}_Ratio_{target}"
        df[name] = df[sc] / df[tgt_close]
        register_feature(name, "shift_1")
        validate_feature(df, name)

    return df


# ============================================================
# CALENDAR + MACRO FEATURES
# ============================================================

def add_calendar(df):
    idx = df.index

    cols = {
        "day_of_week": idx.dayofweek,
        "day_of_month": idx.day,
        "month": idx.month,
        "quarter": idx.quarter
    }

    for name, series in cols.items():
        df[name] = series
        register_feature(name, "no_shift")
        validate_feature(df, name)

    return df


def add_holidays(df):
    us = holidays.US()
    df["is_holiday_adjacent"] = [
        int((d + pd.Timedelta(days=1) in us) or (d - pd.Timedelta(days=1) in us))
        for d in df.index
    ]
    register_feature("is_holiday_adjacent", "no_shift")
    validate_feature(df, "is_holiday_adjacent")
    return df


def add_macro_distances(df):
    """
    Dynamic macro event distances:
    - Expands events across all years covered by df.index
    - Prevents huge incorrect values (e.g., 3600+ days_to_cpi)
    - Uses the same feature names and structure as the original version
    """

    start = df.index.min()
    end   = df.index.max()

    years = range(start.year - 1, end.year + 2)   # pad 1 year on both sides

    # Same event month/day templates you currently have
    templates = {
        "cpi": [(1,11), (2,13), (3,12)],
        "nfp": [(1,5),  (2,2),  (3,8)],
    }

    def expand_dates(md_list):
        out = []
        for y in years:
            for m, d in md_list:
                try:
                    out.append(pd.Timestamp(year=y, month=m, day=d))
                except:
                    pass
        s = pd.to_datetime(out)
        # keep only reasonable vicinity
        s = s[(s >= start - pd.Timedelta(days=365)) &
              (s <= end   + pd.Timedelta(days=365))]
        return s.sort_values()

    for k, md_list in templates.items():

        ds = expand_dates(md_list)

        # ---------- distance to next event ----------
        def next_days(idx):
            out = []
            j = 0
            for d in idx:
                while j < len(ds) and ds[j] < d:
                    j += 1
                # if no next event → "far away" sentinel
                out.append((ds[j] - d).days if j < len(ds) else FILLER)
            return out

        # ---------- distance since previous event ----------
        def prev_days(idx):
            out = []
            j = 0
            for d in idx:
                while j < len(ds) and ds[j] <= d:
                    j += 1
                # if no previous event → sentinel
                out.append((d - ds[j-1]).days if j > 0 else FILLER)
            return out

        n = f"days_to_{k}"
        p = f"days_since_{k}"

        df[n] = next_days(df.index)
        df[p] = prev_days(df.index)

        register_feature(n, "no_shift")
        register_feature(p, "no_shift")

        validate_feature(df, n)
        validate_feature(df, p)

    return df



def add_calendar_macro(df):
    df = add_calendar(df)
    df = add_holidays(df)
    df = add_macro_distances(df)
    return df


# ============================================================
# SHIFT ENGINE
# ============================================================

def apply_shift_engine(df):
    df = df.copy()
    out = {}

    for col, rule in feature_registry.items():
        if rule == "no_shift":
            out[col] = df[col]
        elif rule == "shift_1":
            name = f"{col}_t-1"
            out[name] = df[col].shift(1)
        else:
            raise ValueError(f"Unknown rule {rule}")

    new_df = pd.DataFrame(out, index=df.index)
    return new_df


# ============================================================
# CLEAN FINAL DATASET
# ============================================================

def clean_final_dataset(df):
    df = df.copy()
    df = df.dropna(how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(thresh=10)
    return df

# ============================================================
# GENERATE MARKDOWN FEATURE DOC
# ============================================================


def generate_markdown_feature_doc(path, horizon):
    """
    Generates a Markdown table documenting each feature:
        - Original feature name (as stored in registry)
        - Shift rule
        - Final output column name(s)

    The registry must already be in its post-shift state.
    """

    global feature_registry

    lines = []
    lines.append("# Feature Transformation Table\n")
    lines.append("Automatically generated after running the feature pipeline.\n\n")
    lines.append("| Feature | Rule | Output Columns |\n")
    lines.append("|---------|------|----------------|\n")

    for feature, rule in feature_registry.items():

        # ----------------------------------------------
        # Determine output columns based on rule
        # ----------------------------------------------
        if rule == "no_shift":
            out_cols = feature

        elif rule == "shift_1":
            out_cols = f"{feature}_t-1"

        elif rule.startswith("shift_plus_"):
            # Expected format: "shift_plus_30"
            k = int(rule.replace("shift_plus_", ""))
            out_cols = ", ".join(f"{feature}_t+{i}" for i in range(1, k + 1))

        else:
            out_cols = "ERROR_UNKNOWN_RULE"

        lines.append(f"| `{feature}` | `{rule}` | `{out_cols}` |\n")

    # ----------------------------------------------
    # Ensure directory exists
    # ----------------------------------------------
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ----------------------------------------------
    # Write file
    # ----------------------------------------------
    with open(path, "w") as f:
        f.writelines(lines)

    print(f"Markdown feature documentation written to: {path}")



# ============================================================
# MASTER PIPELINE
# ============================================================

def build_feature_dataset():
    tickers = [TARGET_TICKER] + SUPPORT_TICKERS

    df = download_prices(TARGET_TICKER, SUPPORT_TICKERS, START_DATE, END_DATE)
    df = clean_raw(df, tickers)

    df = initialize_feature_registry(df, TARGET_TICKER, SUPPORT_TICKERS)
    df = add_all_technicals(df, tickers, USE_HMM)
    df = generate_cross_ticker_features(df, TARGET_TICKER, SUPPORT_TICKERS)
    df = add_calendar_macro(df)

    df_shift = apply_shift_engine(df)
    df_clean = clean_final_dataset(df_shift)
    generate_markdown_feature_doc(path=PATH, horizon=HORIZON)
    return df_clean.loc[START_DATE:END_DATE]


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    final_df = build_feature_dataset()
    print(final_df.shape)
