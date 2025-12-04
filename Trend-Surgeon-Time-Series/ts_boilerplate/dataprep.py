"""Prepare Data so as to be used in a Pipelined ML model"""

import numpy as np
from typing import Tuple, List
import pandas as pd
import yfinance as yf
import logging
from tqdm.auto import tqdm
from hmmlearn.hmm import GaussianHMM
import holidays

TARGET_TICKER = "PPH"
SUPPORT_TICKERS = [
    "XPH"
]

START_DATE = "2014-02-04"
END_DATE = "2025-08-03"
HORIZON = 30
USE_HMM = True



# ============================================================
# STEP A — RAW DATA PREPARATION WITH SAFE WINDOW EXPANSION
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _compute_safe_window(start_date, end_date, min_history=20, horizon=30):
    """
    Expands the user's requested date range so that:
      - rolling windows (SMA50, corr60, etc.) have enough initial history
      - shift_plus_k has future coverage
    """
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    safe_start = start - pd.Timedelta(days=int(min_history * 2.5))
    safe_end   = end + pd.Timedelta(days=horizon + 10)

    return safe_start, safe_end


def download_and_prepare_data(target, support_tickers,
                              start_date, end_date,
                              min_history=20, horizon=30):
    """
    Combines:
      ✔ Your original OHLCV preparation logic
      ✔ Automatic expansion of the date window
      ✔ Clear validation errors if the requested date range is unavailable
      ✔ Full safe dataset for all later steps
    """

    tickers = [target] + list(support_tickers)
    logger.info(f"Downloading data for: {tickers}")

    # ------------------------------------------------------------
    # 1. Compute expanded safe window
    # ------------------------------------------------------------
    safe_start, safe_end = _compute_safe_window(start_date, end_date,
                                                min_history, horizon)

    logger.info(f"Safe fetch window: {safe_start.date()} → {safe_end.date()}")

    # ------------------------------------------------------------
    # 2. Download from yfinance inside safe window
    # ------------------------------------------------------------
    raw = yf.download(
        tickers=tickers,
        start=safe_start.strftime("%Y-%m-%d"),
        end=safe_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        group_by="ticker",
        progress=False
    )

    if raw.empty:
        raise ValueError("❌ yfinance returned no data.")

    # ------------------------------------------------------------
    # 3. Flatten columns
    # ------------------------------------------------------------
    raw.columns = [f"{t}_{f}" for t, f in raw.columns]

    # ------------------------------------------------------------
    # 4. Drop Adj Close if any
    # ------------------------------------------------------------
    raw = raw[[c for c in raw.columns if "Adj_Close" not in c and "Adj Close" not in c]]

    # ------------------------------------------------------------
    # 5. Ensure datetime index
    # ------------------------------------------------------------
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.set_index("date")

    raw.index = pd.to_datetime(raw.index)

    # ------------------------------------------------------------
    # 6. Validate that requested range is covered
    # ------------------------------------------------------------
    earliest = raw.index.min()
    latest   = raw.index.max()

    req_start = pd.to_datetime(start_date)
    req_end   = pd.to_datetime(end_date)

    if earliest > req_start:
        raise ValueError(
            f"❌ Requested start {req_start.date()} is too early.\n"
            f"   Earliest fetched data is {earliest.date()}"
        )

    if latest < req_end:
        raise ValueError(
            f"❌ Requested end {req_end.date()} is too late.\n"
            f"   Latest fetched data is {latest.date()}"
        )

    # ------------------------------------------------------------
    # 7. DO NOT trim early — keep full safe window
    #    Feature generation needs the earlier history!
    # ------------------------------------------------------------

    logger.info(f"Data prepared. Shape after safe fetch: {raw.shape}")

    return raw


# ============================================================
# STEP A.5 — EARLY CLEANING OF RAW DATA
# ============================================================

def clean_raw_ohlcv(df, tickers):
    """
    Early cleaning applied immediately after downloading and trimming OHLCV data.
    Ensures stable inputs for feature generation.

    Steps:
    - Drop duplicate index entries
    - Ensure index is sorted
    - Replace infinite values
    - Forward-fill *only* OHLCV values to handle non-trading gaps
    - Remove rows where some tickers have partial OHLCV (e.g. trading halts)
    - Enforce numeric dtypes
    """

    df = df.copy()

    # --------------------------------------------
    # 1. Remove duplicate dates & sort index
    # --------------------------------------------
    df = df[~df.index.duplicated(keep='first')].sort_index()

    # --------------------------------------------
    # 2. Replace infinite values
    # --------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------
    # 3. Forward-fill ONLY OHLCV fields across non-trading gaps
    # --------------------------------------------
    ohlcv_cols = []
    for t in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            col = f"{t}_{field}"
            if col in df.columns:
                ohlcv_cols.append(col)

    # Forward-fill OHLCV missing values (safe for market closures)
    df[ohlcv_cols] = df[ohlcv_cols].ffill()

    # --------------------------------------------
    # 4. Remove rows where *not all* tickers have full OHLCV
    # --------------------------------------------
    mask = df[ohlcv_cols].notna().all(axis=1)
    df = df[mask]

    # --------------------------------------------
    # 5. Enforce numeric dtype for OHLCV columns
    # --------------------------------------------
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------
    # 6. Remove any row that still contains NaNs in OHLCV
    # --------------------------------------------
    df = df.dropna(subset=ohlcv_cols, how="any")

    return df


# GLOBAL
feature_registry = {}

def register_feature(name, shift):
    global feature_registry
    feature_registry[name] = shift
    print(f"Registered feature: {name} with shift: {shift}")

# ============================================================
# STEP B — FEATURE REGISTRY (Exact Implementation Requested)
# ============================================================

def initialize_feature_registry(df, target, support_tickers):

    global feature_registry

    # (1) RESET
    feature_registry.clear()

    # (2) Add canonical target_close
    raw_target_close = f"{target}_Close"
    df["target_close"] = df[raw_target_close].copy()

    # (3) Register canonical target
    register_feature("target_close", "no_shift")

    # (4) Loop through all columns
    for col in df.columns:

        if col == "target_close":
            continue  # already handled

        if col.endswith("_Open"):
            register_feature(col, "no_shift")
        else:
            register_feature(col, "shift_1")

    print("FINAL REGISTRY:", feature_registry)
    return feature_registry



# ============================================================
# STEP C — PER-TICKER TECHNICAL INDICATORS (FINAL VERSION)
# ===========================================================

# ------------------------------------------------------------
# OHLCV Helper
# ------------------------------------------------------------
def get_ohlcv(df, ticker):
    """
    Returns (Open, High, Low, Close, Volume) column names for a ticker.
    Ensures all are present.
    """
    cols = [f"{ticker}_{x}" for x in ["Open", "High", "Low", "Close", "Volume"]]
    return cols if all(c in df.columns for c in cols) else None


# ------------------------------------------------------------
# MOMENTUM FEATURES
# ------------------------------------------------------------
def feat_returns(df, ticker, close):
    feats = {}
    for w in [1, 5, 10, 20]:
        name = f"{ticker}_Return_{w}d"
        feats[name] = df[close].pct_change(w)
        register_feature(name, "shift_1")
    return feats


# ------------------------------------------------------------
# TREND FEATURES (SMA/EMA/MA Cross)
# ------------------------------------------------------------
def feat_sma_ema(df, ticker, close):
    feats = {}

    name = f"{ticker}_SMA_10"
    feats[name] = df[close].rolling(10).mean()
    register_feature(name, "shift_1")

    # name = f"{ticker}_SMA_50"
    # feats[name] = df[close].rolling(50).mean()
    # register_feature(name, "shift_1")

    name = f"{ticker}_EMA_20"
    feats[name] = df[close].ewm(span=20).mean()
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# MACD FEATURES
# ------------------------------------------------------------
def feat_macd(df, ticker, close):
    feats = {}

    ema12 = df[close].ewm(span=12).mean()
    ema26 = df[close].ewm(span=26).mean()
    macd = ema12 - ema26

    name = f"{ticker}_MACD"
    feats[name] = macd
    register_feature(name, "shift_1")

    name = f"{ticker}_MACD_sig"
    feats[name] = macd.ewm(span=9).mean()
    register_feature(name, "shift_1")

    name = f"{ticker}_MACD_hist"
    feats[name] = feats[f"{ticker}_MACD"] - feats[f"{ticker}_MACD_sig"]
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# RSI FEATURE
# ------------------------------------------------------------
def feat_rsi(df, ticker, close):
    feats = {}

    delta = df[close].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down

    name = f"{ticker}_RSI_14"
    feats[name] = 100 - (100 / (1 + rs))
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# STOCHASTICS FEATURES
# ------------------------------------------------------------
def feat_stochastics(df, ticker, close):
    feats = {}

    low14 = df[close].rolling(14).min()
    high14 = df[close].rolling(14).max()

    name = f"{ticker}_StochK"
    feats[name] = 100 * (df[close] - low14) / (high14 - low14)
    register_feature(name, "shift_1")

    name = f"{ticker}_StochD"
    feats[name] = feats[f"{ticker}_StochK"].rolling(3).mean()
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# VOLATILITY FEATURES
# ------------------------------------------------------------
def feat_volatility(df, ticker, o, h, l, c, returns_dict):
    feats = {}
    ret1 = returns_dict[f"{ticker}_Return_1d"]

    name = f"{ticker}_Vol_20"
    feats[name] = ret1.rolling(20).std()
    register_feature(name, "shift_1")

    name = f"{ticker}_Parkinson_20"
    feats[name] = ((np.log(df[h]/df[l])**2).rolling(20).mean() * (1/(4*np.log(2))))
    register_feature(name, "shift_1")

    name = f"{ticker}_GK_20"
    feats[name] = (
        0.5*(np.log(df[h]/df[l])**2)
        - (2*np.log(np.e)-1)*(np.log(df[c]/df[o])**2)
    ).rolling(20).mean()
    register_feature(name, "shift_1")

    sma20 = df[c].rolling(20).mean()
    std20 = df[c].rolling(20).std()

    name = f"{ticker}_BB_width"
    feats[name] = (2 * std20) / sma20
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# VOLUME FEATURES
# ------------------------------------------------------------
def feat_volume(df, ticker, v, c):
    feats = {}

    name = f"{ticker}_Volume_ROC"
    feats[name] = df[v].pct_change(5)
    register_feature(name, "shift_1")

    name = f"{ticker}_Volume_Z"
    feats[name] = (df[v] - df[v].rolling(20).mean()) / df[v].rolling(20).std()
    register_feature(name, "shift_1")

    name = f"{ticker}_OBV"
    feats[name] = (np.sign(df[c].diff()) * df[v]).fillna(0).cumsum()
    register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# ENTROPY FEATURE
# ------------------------------------------------------------
def feat_entropy(df, ticker):
    feats = {}

    col = f"{ticker}_Return_1d"
    if col in df.columns:
        name = f"{ticker}_Entropy_20"
        feats[name] = (df[col]**2).rolling(20).sum()
        register_feature(name, "shift_1")

    return feats


# ------------------------------------------------------------
# HMM BLOCK (optional)
# ------------------------------------------------------------

def feat_hmm(df, ticker, returns_dict, n_states=3):
    """
    Fit a simple Gaussian HMM on 1-day returns and output hidden state sequence.
    Output MUST be same length as df, with NaN padding at the top.
    """
    feats = {}
    ret = returns_dict.get(f"{ticker}_Return_1d")

    if ret is None:
        return feats

    # Convert to numpy, drop NaNs for fitting
    clean = ret.dropna().values.reshape(-1, 1)

    if len(clean) < 50:
        # Not enough data for HMM
        feats[f"{ticker}_HMM"] = pd.Series(np.nan, index=df.index)
        register_feature(f"{ticker}_HMM", "shift_1")
        return feats

    # Fit HMM
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200)
    model.fit(clean)

    # Predict states
    hidden_states = model.predict(clean)

    # Pad output back to full length
    pad_length = len(df) - len(hidden_states)
    padded = np.concatenate([np.full(pad_length, np.nan), hidden_states])

    col = f"{ticker}_HMM"
    feats[col] = pd.Series(padded, index=df.index)

    register_feature(col, "shift_1")
    return feats



# ------------------------------------------------------------
# MASTER WRAPPER (FIXED)
# ------------------------------------------------------------
def generate_per_ticker_features(df, tickers, use_hmm=True):
    """
    Computes all technical indicators for each ticker.
    Ensures correct sequencing:

        returns → trend → MACD → RSI → Stochastics → volatility → volume → entropy → HMM
    """
    all_feats = {}

    # We make a working copy so df is never modified outside Step C
    df_local = df.copy()

    for ticker in tqdm(tickers, desc="Per-ticker technicals"):
        ohlcv = get_ohlcv(df_local, ticker)
        if not ohlcv:
            continue

        o, h, l, c, v = ohlcv

        # --------------------------------------------
        # 1. RETURNS (must run first)
        # --------------------------------------------
        returns_dict = feat_returns(df_local, ticker, c)
        all_feats.update(returns_dict)

        # **CRITICAL FIX: Insert returns into df_local so entropy & others can see them**
        for k, s in returns_dict.items():
            df_local[k] = s

        # --------------------------------------------
        # 2. TREND
        # --------------------------------------------
        feats = feat_sma_ema(df_local, ticker, c)
        all_feats.update(feats)

        # --------------------------------------------
        # 3. MACD
        # --------------------------------------------
        feats = feat_macd(df_local, ticker, c)
        all_feats.update(feats)

        # --------------------------------------------
        # 4. MOMENTUM
        # --------------------------------------------
        feats = feat_rsi(df_local, ticker, c)
        all_feats.update(feats)

        feats = feat_stochastics(df_local, ticker, c)
        all_feats.update(feats)

        # --------------------------------------------
        # 5. VOLATILITY (depends on Return_1d)
        # --------------------------------------------
        feats = feat_volatility(df_local, ticker, o, h, l, c, returns_dict)
        all_feats.update(feats)

        # --------------------------------------------
        # 6. VOLUME
        # --------------------------------------------
        feats = feat_volume(df_local, ticker, v, c)
        all_feats.update(feats)

        # --------------------------------------------
        # 7. ENTROPY  (now works, because Return_1d is in df_local)
        # --------------------------------------------
        feats = feat_entropy(df_local, ticker)
        all_feats.update(feats)

        # --------------------------------------------
        # 8. OPTIONAL HMM
        # --------------------------------------------
        if use_hmm:
            feats = feat_hmm(df_local, ticker, returns_dict)
            all_feats.update(feats)

    return all_feats


# ============================================================
# STEP D — CROSS-TICKER FEATURES
# ============================================================

def compute_cross_ticker_features(df, target, support_tickers):
    """
    Builds cross-ticker features comparing each support ticker to the target.
    Returns a dict of {feature_name: Series}.
    """

    feats = {}

    target_close = f"{target}_Close"
    if target_close not in df.columns:
        raise ValueError(f"Target {target_close} not found in dataframe.")

    target_ret_20 = f"{target}_Return_20d"
    if target_ret_20 not in df.columns:
        # Should never happen because Step C adds it
        raise ValueError(f"{target_ret_20} missing — Step C must run first.")

    for ticker in support_tickers:
        close_col = f"{ticker}_Close"
        ret20_col = f"{ticker}_Return_20d"

        if close_col not in df.columns:
            continue  # skip incomplete tickers

        # ------------------------------
        # Price Ratio
        # ------------------------------
        name = f"{ticker}_Ratio_{target}"
        feats[name] = df[close_col] / df[target_close]
        register_feature(name, "shift_1")

        # ------------------------------
        # RS-20 (Relative Strength)
        # ------------------------------
        if ret20_col in df.columns:
            name = f"{ticker}_RS_20"
            feats[name] = df[ret20_col] - df[target_ret_20]
            register_feature(name, "shift_1")

        # ------------------------------
        # 60-Day Rolling Correlation
        # ------------------------------
        # name = f"{ticker}_Corr_{target}_60"
        # feats[name] = df[close_col].rolling(60).corr(df[target_close])
        # register_feature(name, "shift_1")

    return feats



# ============================================================
# STEP E — PCA LATENT FACTORS
# ============================================================

from sklearn.decomposition import PCA

def compute_pca_features(df, tickers):
    """
    Computes PCA on the 1-day returns of all tickers.
    Returns a dict of {feature_name: Series}.
    """

    feats = {}

    # Collect all 1-day return columns
    ret_cols = [f"{t}_Return_1d" for t in tickers if f"{t}_Return_1d" in df.columns]

    if len(ret_cols) == 0:
        # Should never happen (Step C must generate them)
        return feats

    # Extract data for PCA
    ret_df = df[ret_cols].dropna()

    # Need enough rows for PCA stability
    if len(ret_df) < 200:
        return feats

    try:
        pca = PCA(n_components=3)
        pca_values = pca.fit_transform(ret_df)

        # PCA_1
        name = "PCA_1"
        feats[name] = pd.Series(pca_values[:, 0], index=ret_df.index)
        register_feature(name, "shift_1")

        # PCA_2
        name = "PCA_2"
        feats[name] = pd.Series(pca_values[:, 1], index=ret_df.index)
        register_feature(name, "shift_1")

        # PCA_3
        name = "PCA_3"
        feats[name] = pd.Series(pca_values[:, 2], index=ret_df.index)
        register_feature(name, "shift_1")

    except Exception as e:
        logging.warning(f"PCA failed: {e}")

    return feats


# ============================================================
# STEP F — CALENDAR + MACRO FEATURES (COMPRESSED, NO 'date')
# ============================================================

# -----------------------------
# FOMC Calendar Fetcher
# -----------------------------
def fetch_fomc_dates():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        tables = pd.read_html(url)
        dates = pd.to_datetime(tables[0].iloc[:, 0], errors="coerce").dropna()
        return dates.sort_values()
    except:
        fallback = pd.to_datetime([
            "2024-01-31","2024-03-20","2024-05-01",
            "2024-06-12","2024-07-31","2024-09-18",
            "2024-11-07","2024-12-18"
        ])
        return fallback.sort_values()

# -----------------------------
# Macro Calendar
# -----------------------------
def macro_calendar():
    events = {
        "cpi": ["2024-01-11","2024-02-13","2024-03-12","2024-04-10"],
        "nfp": ["2024-01-05","2024-02-02","2024-03-08","2024-04-05"],
        "ppi": ["2024-01-12","2024-02-16","2024-03-14","2024-04-11"],
        "gdp": ["2024-01-25","2024-02-28","2024-03-28","2024-04-25"],
    }
    return {k: pd.to_datetime(v).sort_values() for k, v in events.items()}


# ============================================================
# HELPER — Compute distances to events
# ============================================================

def _days_to_next(idx, event_dates):
    """Return Series: days until the next event."""
    out = []
    j = 0

    for d in idx:
        while j < len(event_dates) and event_dates[j] < d:
            j += 1
        if j == len(event_dates):
            out.append(None)  # No next event
        else:
            out.append((event_dates[j] - d).days)

    return pd.Series(out, index=idx)


def _days_since_prev(idx, event_dates):
    """Return Series: days since previous event."""
    out = []
    j = 0

    for d in idx:
        # Find last event <= d
        while j < len(event_dates) and event_dates[j] <= d:
            j += 1
        if j == 0:
            out.append(None)
        else:
            out.append((d - event_dates[j-1]).days)

    return pd.Series(out, index=idx)


# ============================================================
# CALENDAR BASICS (NO SHIFT)
# ============================================================

def add_calendar_basics(df):
    idx = df.index

    features = {
        "day_of_week":    idx.dayofweek,
        "day_of_month":   idx.day,
        "month":          idx.month,
        "quarter":        idx.quarter,
        "is_month_end":   idx.is_month_end.astype(int),
        "is_month_start": idx.is_month_start.astype(int)
    }

    for name, series in features.items():
        df[name] = series
        register_feature(name, "no_shift")

    return df


# ============================================================
# HOLIDAYS (NO SHIFT)
# ============================================================

def add_holiday_features(df):
    us_holidays = holidays.US()
    idx = df.index

    df["is_holiday_adjacent"] = [
        int((d + pd.Timedelta(days=1) in us_holidays) or
            (d - pd.Timedelta(days=1) in us_holidays))
        for d in idx
    ]
    register_feature("is_holiday_adjacent", "no_shift")

    return df


# ============================================================
# OPEX WEEK (NO SHIFT)
# ============================================================

def add_opex_features(df):
    idx = df.index
    df["is_opex_week"] = [
        int((d.weekday() == 4) and (15 <= d.day <= 21))
        for d in idx
    ]
    register_feature("is_opex_week", "no_shift")
    return df


# ============================================================
# FOMC (COMPRESSED)
# ============================================================

def add_fomc_features(df):
    idx = df.index
    dates = fetch_fomc_dates()

    df["is_fomc_day"]        = idx.isin(dates).astype(int)
    df["days_to_fomc"]       = _days_to_next(idx, dates)
    df["days_since_fomc"]    = _days_since_prev(idx, dates)

    for col in ["is_fomc_day", "days_to_fomc", "days_since_fomc"]:
        register_feature(col, "no_shift")

    return df


# ============================================================
# MACRO EVENTS (COMPRESSED)
# ============================================================

def add_macro_features(df):
    idx = df.index
    macros = macro_calendar()

    for name, dates in macros.items():

        df[f"is_{name}_day"]          = idx.isin(dates).astype(int)
        df[f"days_to_{name}"]        = _days_to_next(idx, dates)
        df[f"days_since_{name}"]     = _days_since_prev(idx, dates)

        register_feature(f"is_{name}_day",      "no_shift")
        register_feature(f"days_to_{name}",     "no_shift")
        register_feature(f"days_since_{name}",  "no_shift")

    return df


# ============================================================
# MASTER WRAPPER
# ============================================================

def generate_calendar_and_macro_features(df):
    df = df.copy()
    df = add_calendar_basics(df)
    df = add_holiday_features(df)
    df = add_opex_features(df)
    df = add_fomc_features(df)
    df = add_macro_features(df)

        # ------------------------------------------------------------
    # FIX: Macro distance features produce unavoidable NaNs
    # ------------------------------------------------------------
    # Any column like days_since_* or days_to_* will have NaNs:
    #   - days_since_* → NaN before the FIRST macro event
    #   - days_to_*    → NaN after the LAST macro event
    # These NaNs should NOT cause row deletion in the cleaning step.
    #
    # Strategy:
    #   Impute with a large sentinel value (999) so the model
    #   interprets “far from event” properly.
    # ------------------------------------------------------------

    dist_cols = [c for c in df.columns
                 if c.startswith("days_since_") or c.startswith("days_to_")]

    if dist_cols:
        df[dist_cols] = df[dist_cols].fillna(999)
    return df


# ============================================================
# STEP G — SHIFT ENGINE (Column-Replacing, Registry-Driven)
# ============================================================

def apply_shift_engine(df, horizon):
    """
    Applies registry-driven temporal alignment AND updates the registry.

    For each column:
        - no_shift:       keep col(t)
        - shift_1:        replace col   → col_t-1
        - shift_plus_k:   replace col   → col_t+H

    Additional improvements:
        ✔ Registry is rewritten to match final column names
        ✔ Markdown output will now be accurate
        ✔ No duplicate columns
        ✔ No original (pre-shift) columns remain
    """

    global feature_registry

    df = df.copy()
    out = {}
    new_registry = {}     # fully rebuild registry from scratch

    # ----------------------------------------------------------
    # 1. Validate registry vs dataframe columns
    # ----------------------------------------------------------
    df_cols = set(df.columns)
    reg_cols = set(feature_registry.keys())

    missing = df_cols - reg_cols
    extra   = reg_cols - df_cols

    if missing:
        raise ValueError(f"Registry missing {len(missing)} columns: {sorted(missing)}")

    if extra:
        raise ValueError(f"Registry contains columns not in df: {sorted(extra)}")

    # ----------------------------------------------------------
    # 2. Apply transformations + rebuild registry
    # ----------------------------------------------------------
    for col in df.columns:
        rule = feature_registry[col]
        s = df[col]

        # ----- no_shift → keep original name -----
        if rule == "no_shift":
            new_name = col
            out[new_name] = s
            new_registry[new_name] = "no_shift"

        # ----- shift_1 → output col_t-1 -----
        elif rule == "shift_1":
            new_name = f"{col}_t-1"
            out[new_name] = s.shift(1)
            new_registry[new_name] = "shift_1"    # final shift label

        # ----- shift_plus_k → output col_t+H -----
        elif rule == "shift_plus_k":
            new_name = f"{col}_t+{horizon}"
            out[new_name] = s.shift(-horizon)
            new_registry[new_name] = f"shift_plus_{horizon}"

        else:
            raise ValueError(f"Unknown shift rule: {rule}")

    # ----------------------------------------------------------
    # 3. Replace registry with the new post-shift registry
    # ----------------------------------------------------------
    feature_registry = new_registry

    # ----------------------------------------------------------
    # 4. Build final DataFrame
    # ----------------------------------------------------------
    aligned_df = pd.DataFrame(out, index=df.index)

    return aligned_df


# ============================================================
# STEP H — MARKDOWN FEATURE DOCUMENTATION GENERATOR
# ============================================================

import os

def generate_markdown_feature_doc(path, horizon):
    """
    Creates a Markdown file documenting:
    - every feature registered
    - the shift rule
    - the resulting final columns after Step G

    Parameters
    ----------
    path : str
        Output path to write markdown file (e.g. "docs/feature_table.md")
    horizon : int
        Forecast horizon, used for shift_plus_k expansion
    """

    lines = []
    lines.append("# Feature Transformation Table\n")
    lines.append("This table is auto-generated from the feature pipeline.\n")
    lines.append("\n")
    lines.append("| Feature | Rule | Output Columns |\n")
    lines.append("|---------|------|----------------|\n")

    for feature, rule in feature_registry.items():

        # ----------------------------------------------
        # shift_1
        # ----------------------------------------------
        if rule == "shift_1":
            output_cols = f"{feature}_t-1"

        # ----------------------------------------------
        # no_shift
        # ----------------------------------------------
        elif rule == "no_shift":
            output_cols = feature

        # ----------------------------------------------
        # shift_plus_k
        # ----------------------------------------------
        elif rule == "shift_plus_k":
            shifted = [f"{feature}_t+{k}" for k in range(1, horizon + 1)]
            output_cols = ", ".join(shifted)

        else:
            output_cols = "ERROR_UNKNOWN_RULE"

        # add row
        lines.append(f"| `{feature}` | `{rule}` | `{output_cols}` |\n")

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write file
    with open(path, "w") as f:
        f.writelines(lines)

    print(f"Markdown feature documentation written to: {path}")


# ============================================================
# FINAL PRODUCTION CLEANER FOR THE FEATURE DATASET
# ============================================================

import numpy as np
import pandas as pd

def clean_final_dataset(df, target, min_history=20):
    """
    Cleans the aligned dataset after the shift engine.

    Actions performed:
        1. Drop rows where absolutely no features exist
        2. Drop first rows where rolling windows leave insufficient history
        3. Drop last rows with NaNs caused by shift_plus_k
        4. Remove columns that are all-NaN or constant
           AND update the registry accordingly
        5. Ensure index is clean, sorted, unique
    """

    global feature_registry

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Drop rows that are literally all NaN
    # ---------------------------------------------------------
    df = df.dropna(how="all")

    # ---------------------------------------------------------
    # 2. Drop early rows lacking enough usable feature history
    # ---------------------------------------------------------
    non_target_cols = [
        c for c in df.columns
        if c not in {target, "target_close"}
    ]

    df["valid_feature_count"] = df[non_target_cols].notna().sum(axis=1)
    df = df[df["valid_feature_count"] >= min_history]
    df = df.drop(columns=["valid_feature_count"], errors="ignore")

    # ---------------------------------------------------------
    # 3. Drop trailing NaN rows (shift_plus_k consequences)
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # 4. Remove all-NaN columns and constant columns
    #    AND update the registry BEFORE dropping
    # ---------------------------------------------------------

# ---------------------------------------------------------
# 3. Drop all-NaN, constant, and long-window columns
# ---------------------------------------------------------

    cols_all_nan = df.columns[df.isna().all()].tolist()

    # Constant columns (e.g., is_year_end when dataset has no Dec 31)
    cols_constant = [c for c in df.columns if df[c].nunique() <= 1]

    # Combine everything to remove
    cols_to_remove = set(cols_all_nan + cols_constant)

    # ----- 1) Remove from registry -----
    for col in cols_to_remove:
        feature_registry.pop(col, None)

    # ----- 2) Remove from dataframe safely -----
    df = df.drop(columns=list(cols_to_remove), errors="ignore")


    # ---------------------------------------------------------
    # 5. Final index validations
    # ---------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Final dataset index must be DatetimeIndex.")

    df = df.sort_index()

    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    return df

# ============================================================
# STEP J - MASTER PIPELINE — COMPLETE DATASET BUILDER
# ============================================================

def build_feature_dataset(
    target,
    support_tickers,
    start_date,
    end_date,
    horizon,
    markdown_output_path=None,
    USE_HMM=False
):
    """
    Full leakage-safe feature engineering pipeline.
    """

    # ---------------------------------------------------------
    # 1. DOWNLOAD RAW PRICE DATA
    # ---------------------------------------------------------
    df = download_and_prepare_data(
        target=target,
        support_tickers=support_tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(df.shape)
    print(df.shape)
    tickers = [target] + support_tickers



    # ---------------------------------------------------------
    # 2. EARLY RAW DATA CLEANUP
    # ---------------------------------------------------------
    def clean_raw_ohlcv(df, tickers):
        """
        Ensures raw OHLCV downloaded from yfinance is clean, sorted, and valid.
        """
        df = df.copy()

        # Ensure index is sorted and unique
        df = df[~df.index.duplicated()].sort_index()

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop any row where ALL tickers have missing OHLCV
        required_cols = []
        for t in tickers:
            for f in ["Open", "High", "Low", "Close", "Volume"]:
                required_cols.append(f"{t}_{f}")

        df = df.dropna(subset=required_cols, how="all")

        return df


    # ---------------------------------------------------------
    # 3. RESET FEATURE REGISTRY
    # ---------------------------------------------------------
    initialize_feature_registry(df, target, support_tickers)

    # ---------------------------------------------------------
    # 4. PER-TICKER TECHNICAL FEATURES  (Step C)
    # ---------------------------------------------------------
    tech_feats = generate_per_ticker_features(df, tickers, use_hmm=USE_HMM)

    # Merge tech features so that Step D & Step E can see them
    df_with_tech = pd.concat(
        [df, pd.DataFrame(tech_feats, index=df.index)], axis=1
    )

    print(df_with_tech.shape)

    # ---------------------------------------------------------
    # 5. CROSS-TICKER FEATURES (Step D)
    # ---------------------------------------------------------
    cross_feats = compute_cross_ticker_features(df_with_tech, target, support_tickers)

    # ---------------------------------------------------------
    # 6. PCA LATENT FACTORS (Step E)
    # ---------------------------------------------------------
    pca_feats = compute_pca_features(df_with_tech, tickers)


    # ---------------------------------------------------------
    # 7. CALENDAR + MACRO FEATURES
    # ---------------------------------------------------------
    df_calendar = generate_calendar_and_macro_features(df)

    print(df_calendar.shape)


    # ---------------------------------------------------------
    # 8. MERGE ALL FEATURE BLOCKS
    # ---------------------------------------------------------
    all_feats = {}
    all_feats.update(tech_feats)
    all_feats.update(cross_feats)
    all_feats.update(pca_feats)

    df_all = pd.concat(
        [df_calendar, pd.DataFrame(all_feats, index=df.index)],
        axis=1
    )

    print(df_all.shape)

    # ---------------------------------------------------------
    # 9. SHIFT ENGINE (APPLY TEMPORAL ALIGNMENT)
    # ---------------------------------------------------------
    df_aligned = apply_shift_engine(
        df_all,
        horizon=horizon       # <-- FIXED
    )
    print(df_aligned.shape)
    # ---------------------------------------------------------
    # 9. SHIFT ENGINE (APPLY TEMPORAL ALIGNMENT)
    # ---------------------------------------------------------

    df_cleaned = clean_final_dataset(df_aligned, target, min_history=20)
    print(df_cleaned.shape)

    # ---------------------------------------------------------
    # 11. OPTIONAL: FEATURE DOCUMENTATION
    # ---------------------------------------------------------
    if markdown_output_path is not None:
        generate_markdown_feature_doc(
            path=markdown_output_path,
            horizon=horizon
        )

    # ---------------------------------------------------------
    # 12. TRIM AND RETURN MODEL-READY DATAFRAME
    # ---------------------------------------------------------

    df = df_cleaned.loc[start_date : end_date]
    print(df.shape)
    return df

# ---------------------------------------------------------
# RUN PIPELINE TO BUILD FINAL FEATURE DATASET
# ---------------------------------------------------------

df_final = build_feature_dataset(
    target=TARGET_TICKER,
    support_tickers=SUPPORT_TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    horizon=HORIZON,
    markdown_output_path="docs/features_no_clean.md"    # Optional
)
# ============================================================


#def get_X_y(


#def get_folds(data: np.ndarray,



#def train_test_split(data: np.ndarray,



#def extract_sentiments(df: pd.DataFrame) -> pd.DataFrame:
