"""
Intraday Microstructure Features — IBKR 15min bars
===================================================
Pour chaque barre 1h, agrege les 4 sous-barres 15min correspondantes
(barre precedente, shift(1)) pour capturer la microstructure intraday.

Ces features sont IMPOSSIBLES a calculer avec les donnees IEX d'Alpaca
(volume trop fragmenté, 2% du marche seulement).
Avec IBKR (volume consolide, 100% du marche), elles sont fiables.

Features calculees :
  intraday_vol       : std des rendements 15min -> volatilite reelle intraday
  intraday_trend     : correlation rendements 15min vs temps -> momentum intraday
  vol_frontload      : % volume dans les 2 premieres sous-barres -> pression debut d'heure
  price_reversal_rt  : taux de retournements de direction -> choppiness
  vwap_dev_intra     : ecart-type du prix vs VWAP 15min -> dispersion autour du prix juste
"""

import numpy as np
import pandas as pd
import os

from data_pipeline.data_storage import RAW_DIR

RAW_15MIN_DIR = os.path.join(os.path.dirname(RAW_DIR), "raw_15min")
os.makedirs(RAW_15MIN_DIR, exist_ok=True)


def _load_15min(symbol: str) -> pd.DataFrame:
    path = os.path.join(RAW_15MIN_DIR, f"{symbol}_15min.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def compute_intraday_features(df_1h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ajoute 5 features de microstructure intraday au DataFrame 1h.

    Chaque feature est calculee a partir des 4 sous-barres 15min
    de la BARRE PRECEDENTE (shift inherent : on utilise T-1 pour predire T).

    Si les donnees 15min sont absentes ou insuffisantes, les features
    sont remplies a zero (valeur neutre — le modele apprend a les ignorer).
    """
    df = df_1h.copy()
    # Initialiser les features a zero
    for feat in ['intraday_vol', 'intraday_trend', 'vol_frontload',
                 'price_reversal_rt', 'vwap_dev_intra']:
        df[feat] = 0.0

    df_15 = _load_15min(symbol)
    if df_15.empty or len(df_15) < 10:
        return df

    # S'assurer que les deux indices sont en UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    if df_15.index.tz is None:
        df_15.index = df_15.index.tz_localize('UTC')

    df_15 = df_15.sort_index()
    df_15['ret_15'] = df_15['Close'].pct_change()
    df_15['typical'] = (df_15['High'] + df_15['Low'] + df_15['Close']) / 3

    # Assigner chaque barre 15min a sa barre 1h parente (merge_asof backward)
    df_15_reset = df_15.reset_index().rename(columns={df_15.index.name or 'index': 'ts'})
    df_1h_reset = df.reset_index().rename(columns={df.index.name or 'index': 'ts'})

    merged = pd.merge_asof(
        df_15_reset[['ts', 'ret_15', 'Volume', 'typical', 'Close']],
        df_1h_reset[['ts']].rename(columns={'ts': 'parent_ts'}),
        left_on='ts', right_on='parent_ts',
        direction='backward'
    )
    merged = merged.dropna(subset=['parent_ts'])

    def _safe_corr(x):
        if len(x) < 3 or x.std() == 0:
            return 0.0
        return float(np.corrcoef(x.values, np.arange(len(x)))[0, 1])

    def _vwap_dev(grp):
        vwap = (grp['typical'] * grp['Volume']).sum() / (grp['Volume'].sum() + 1e-9)
        return ((grp['Close'] - vwap) / (vwap + 1e-9)).std()

    def _reversal_rt(x):
        signs = np.sign(x.values)
        return float((np.diff(signs) != 0).sum() / max(len(signs) - 1, 1))

    agg = merged.groupby('parent_ts').agg(
        intraday_vol    =('ret_15', 'std'),
        intraday_trend  =('ret_15', _safe_corr),
        price_reversal_rt=('ret_15', _reversal_rt),
        vol_first_half  =('Volume', lambda x: x.iloc[:max(1, len(x)//2)].sum()),
        vol_total       =('Volume', 'sum'),
    )
    agg['vol_frontload'] = agg['vol_first_half'] / (agg['vol_total'] + 1e-9)
    agg['vwap_dev_intra'] = merged.groupby('parent_ts').apply(_vwap_dev)

    # Shift(1) : la barre T utilise les features de la barre T-1
    agg = agg.shift(1)

    # Merger sur l'index 1h
    for feat in ['intraday_vol', 'intraday_trend', 'vol_frontload',
                 'price_reversal_rt', 'vwap_dev_intra']:
        if feat in agg.columns:
            df[feat] = df.index.map(agg[feat]).fillna(0.0).values

    return df


def compute_all_intraday(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Point d'entree pour les features intraday IBKR."""
    return compute_intraday_features(df, symbol)
