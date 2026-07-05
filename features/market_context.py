import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import load_data

# ============================================================
# MARKET CONTEXT FEATURES
# ============================================================
# Donne au modèle une vision du régime de marché global.
# TSLA est très corrélé au S&P500 et sensible à la peur (VIX).
# Sans ce contexte, le modèle trade TSLA "en aveugle" sur le marché.
#
# Toutes les features utilisent .shift(1) — anti look-ahead bias.
# ============================================================


def add_spy_features(df, spy_df):
    """
    Features basées sur SPY (S&P 500 ETF) :
    - Retours SPY multi-horizons (contexte directionnel)
    - Tendance SPY vs moyenne mobile (régime haussier/baissier)
    - Beta TSLA/SPY rolling (sensibilité au marché)
    - Force relative TSLA vs SPY (TSLA surperforme-t-il le marché ?)
    """
    spy = spy_df['Close'].reindex(df.index, method='ffill').shift(1)

    df['spy_ret_1']  = spy.pct_change(1)
    df['spy_ret_4']  = spy.pct_change(4)
    df['spy_ret_20'] = spy.pct_change(20)

    spy_ma20 = spy.rolling(20).mean()
    df['spy_above_ma20'] = (spy > spy_ma20).astype(int)

    tsla_ret = df['Close'].shift(1).pct_change()
    spy_ret  = spy.pct_change()

    cov_20   = tsla_ret.rolling(20).cov(spy_ret)
    var_20   = spy_ret.rolling(20).var()
    df['beta_20'] = cov_20 / (var_20 + 1e-9)

    df['rel_strength_20'] = tsla_ret.rolling(20).sum() - spy_ret.rolling(20).sum()

    return df


def add_vix_features(df, vix_df):
    """
    Features basées sur le VIX (indice de peur du marché) :
    - Niveau absolu du VIX (>20 = marché stressé)
    - Variation du VIX (le VIX monte → danger)
    - Z-score du VIX (relativise le niveau par rapport à l'historique récent)
    - Flag régime de peur (VIX > 20)
    """
    vix = vix_df['Close'].reindex(df.index, method='ffill').shift(1)

    df['vix_level']  = vix
    df['vix_change'] = vix.pct_change(1)

    vix_mean = vix.rolling(20).mean()
    vix_std  = vix.rolling(20).std()
    df['vix_zscore'] = (vix - vix_mean) / (vix_std + 1e-9)

    df['vix_high'] = (vix > 20).astype(int)

    return df


def compute_all_market_context(df, symbol=""):
    freq   = STRATEGY.get("frequency", "1h")
    spy_df = load_data(f"SPY_context_{freq}")
    vix_df = load_data(f"VIX_context_{freq}")

    n_added = 0

    if spy_df is not None:
        df = add_spy_features(df, spy_df)
        n_added += 6
    else:
        print(f"     ⚠️ SPY indisponible — features de contexte marché ignorées")

    if vix_df is not None:
        df = add_vix_features(df, vix_df)
        n_added += 4
    else:
        print(f"     ⚠️ VIX indisponible — features de peur ignorées")

    if n_added > 0:
        print(f"     🌐 Contexte marché ({n_added} features : SPY + VIX)")

    return df
