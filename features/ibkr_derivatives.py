"""
IBKR Derivatives Features — Option Implied Volatility & Variance Risk Premium
==============================================================================
Features basees sur des donnees exclusivement disponibles via IBKR :
  - Option Implied Volatility (IV) 1h par action
  - Historical Volatility (HV) calculee localement

Features calculees :
  iv_level      : niveau absolu de l'IV (fear gauge par action)
  iv_change     : variation horaire de l'IV  [IC=-0.066 sur AAPL — signal fort]
  iv_pct_90d    : rang percentile de l'IV sur 90 jours (VIX-like par action)
  vrp           : IV - HV realisee = Variance Risk Premium
                  VRP > 0 : options cheres  -> signal bearish
                  VRP < 0 : options bon marche -> signal bullish

Ces features sont impossibles a calculer avec Alpaca (pas d'acces options)
ou Yahoo (pas d'IV horaire par action).

Reference academique : Bollerslev & Todorov (2011), Carr & Wu (2009).
"""

import os
import numpy as np
import pandas as pd

from data_pipeline.data_storage import RAW_DIR

RAW_IV_DIR  = os.path.join(os.path.dirname(RAW_DIR), "raw_iv")
RAW_HV_DIR  = os.path.join(os.path.dirname(RAW_DIR), "raw_hv")
RAW_OPT_DIR = os.path.join(os.path.dirname(RAW_DIR), "raw_opt")
for _d in [RAW_IV_DIR, RAW_HV_DIR, RAW_OPT_DIR]:
    os.makedirs(_d, exist_ok=True)

IV_WINDOW_PCT    = 90 * 7   # 90 jours de barres 1h pour le percentile
IV_ZSCORE_1Y     = 252 * 6  # ~1 an de barres 1h
IV_CHANGE_WINDOW = 90 * 7   # 90 jours de barres 1h pour z-score iv_change
HV_WINDOW        = 20        # 20 barres 1h pour HV realisee (fallback local)
OPT_VOL_WINDOW   = 90        # 90 jours pour z-score volume options
IV_HISTORY_YRS   = 2         # 2 ans d'historique IV maximum via IBKR


def fetch_and_cache_iv(symbol: str, years: int = IV_HISTORY_YRS) -> pd.DataFrame:
    """Telecharge l'IV 1h via IBKR en chunks de 180 jours et met en cache."""
    import time
    from datetime import datetime, timezone, timedelta

    try:
        from execution.broker import _ib, _bars_to_df
        from ib_insync import Stock
    except ImportError:
        return pd.DataFrame()

    cache_path = os.path.join(RAW_IV_DIR, f"{symbol}_iv.parquet")

    # Mode incremental : si cache existant, seulement 30 derniers jours
    if os.path.exists(cache_path):
        existing = pd.read_parquet(cache_path)
        n_chunks = 1
        years_fetch = 1
    else:
        existing = pd.DataFrame()
        n_chunks = max(1, round(years * 365 / 180))
        years_fetch = years

    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)

    all_dfs = []
    end_dt = datetime.now(timezone.utc)

    for i in range(n_chunks):
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"
        try:
            bars = _ib().reqHistoricalData(
                contract, endDateTime=end_str, durationStr='180 D',
                barSizeSetting='1 hour', whatToShow='OPTION_IMPLIED_VOLATILITY',
                useRTH=True, formatDate=1,
            )
            if bars:
                df = _bars_to_df(bars)[['Close']].rename(columns={'Close': 'iv'})
                all_dfs.append(df)
        except Exception as e:
            print(f"     ⚠️ IV {symbol} chunk {i+1}: {e}")
        end_dt = end_dt - timedelta(days=179)
        if i < n_chunks - 1:
            time.sleep(1)

    if not all_dfs and existing.empty:
        return pd.DataFrame()

    frames = all_dfs + ([existing] if not existing.empty else [])
    df_full = pd.concat(frames)
    df_full = df_full[~df_full.index.duplicated(keep='last')].sort_index()

    df_full.to_parquet(cache_path, compression='snappy')
    return df_full


def fetch_and_cache_hv(symbol: str, days: int = 365) -> pd.DataFrame:
    """Telecharge et cache la HV journaliere IBKR (Parkinson estimator)."""
    cache_path = os.path.join(RAW_HV_DIR, f"{symbol}_hv.parquet")
    try:
        from execution.broker import fetch_hv_ibkr, IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            return pd.DataFrame()
        if os.path.exists(cache_path):
            existing = pd.read_parquet(cache_path)
            df_new = fetch_hv_ibkr(symbol, days=30)
            if not df_new.empty and not existing.empty:
                df = pd.concat([existing, df_new])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            else:
                df = existing
        else:
            df = fetch_hv_ibkr(symbol, days=days)
        if not df.empty:
            df.to_parquet(cache_path, compression='snappy')
        return df
    except Exception:
        return pd.DataFrame()


def fetch_and_cache_opt_vol(symbol: str, days: int = 365) -> pd.DataFrame:
    """Telecharge et cache le volume journalier des options IBKR."""
    cache_path = os.path.join(RAW_OPT_DIR, f"{symbol}_optvol.parquet")
    try:
        from execution.broker import fetch_option_volume, IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            return pd.DataFrame()
        if os.path.exists(cache_path):
            existing = pd.read_parquet(cache_path)
            df_new = fetch_option_volume(symbol, days=30)
            if not df_new.empty and not existing.empty:
                df = pd.concat([existing, df_new])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            else:
                df = existing
        else:
            df = fetch_option_volume(symbol, days=days)
        if not df.empty:
            df.to_parquet(cache_path, compression='snappy')
        return df
    except Exception:
        return pd.DataFrame()


def _load_iv(symbol: str) -> pd.DataFrame:
    path = os.path.join(RAW_IV_DIR, f"{symbol}_iv.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_hv_ibkr(symbol: str) -> pd.DataFrame:
    path = os.path.join(RAW_HV_DIR, f"{symbol}_hv.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_opt_vol(symbol: str) -> pd.DataFrame:
    path = os.path.join(RAW_OPT_DIR, f"{symbol}_optvol.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def compute_iv_features(df_1h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ajoute les features de volatilite implicite au DataFrame 1h.

    Les features utilisent shift(1) pour eviter le look-ahead bias :
    la barre T utilise l'IV de T-1 pour predire le retour de T.

    Pour les barres sans historique IV (> 2 ans), les features sont a zero.
    """
    df = df_1h.copy()
    for feat in ['iv_level', 'iv_change', 'iv_pct_90d', 'vrp', 'log_iv_level',
                 'iv_zscore_1y', 'iv_change_zscore']:
        df[feat] = 0.0

    iv_df = _load_iv(symbol)
    if iv_df.empty:
        return df

    # Aligner IV sur l'index 1h
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    if iv_df.index.tz is None:
        iv_df.index = iv_df.index.tz_localize('UTC')

    # Merge sur timestamps communs (forward-fill les gaps mineurs)
    iv_aligned = iv_df.reindex(df.index, method='ffill', limit=3)

    # HV realisee : rolling std des rendements * sqrt(252 * 6.5) annualisee
    ret = df['Close'].pct_change()
    hv_realized = ret.rolling(HV_WINDOW).std() * np.sqrt(252 * 6.5)

    # Construire les features (shift(1) pour eviter look-ahead)
    iv_series = iv_aligned['iv'].shift(1)

    df['iv_level']    = iv_series.fillna(0.0)
    df['iv_change']   = iv_series.diff().fillna(0.0)
    df['iv_pct_90d']  = iv_series.rolling(IV_WINDOW_PCT, min_periods=50).rank(pct=True).fillna(0.0)
    df['vrp']         = (iv_series - hv_realized.shift(1)).fillna(0.0)

    # log_iv_level : log-transform de l'IV (distribution log-normale → plus normale après transform)
    # Seulement là où IV est disponible — zéro sinon (barres sans IV conservent leur valeur neutre)
    df['log_iv_level'] = np.log(iv_series.clip(lower=1e-6)).fillna(0.0)

    # iv_zscore_1y : z-score de l'IV sur ~1 an glissant
    iv_roll_mean = iv_series.rolling(IV_ZSCORE_1Y, min_periods=100).mean()
    iv_roll_std  = iv_series.rolling(IV_ZSCORE_1Y, min_periods=100).std().replace(0, np.nan)
    df['iv_zscore_1y'] = ((iv_series - iv_roll_mean) / iv_roll_std).fillna(0.0)

    # iv_change_zscore : z-score de iv_change sur 90j
    iv_chg = iv_series.diff()
    chg_mean = iv_chg.rolling(IV_CHANGE_WINDOW, min_periods=50).mean()
    chg_std  = iv_chg.rolling(IV_CHANGE_WINDOW, min_periods=50).std().replace(0, np.nan)
    df['iv_change_zscore'] = ((iv_chg - chg_mean) / chg_std).fillna(0.0)

    return df


def compute_all_derivatives(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Point d'entree pour les features IV IBKR."""
    return compute_iv_features(df, symbol)
