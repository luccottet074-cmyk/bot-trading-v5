import numpy as np
import pandas as pd

# ============================================================
# VOLATILITY FACTORS (Chapitre 4 - ML4T)
# ============================================================
# La volatilité est un signal clé en intraday : elle mesure l'incertitude
# et permet au modèle ML de détecter les régimes de marché (calme vs agité).
# Toutes les fonctions appliquent un .shift(1) pour éviter le Look-Ahead Bias.
# ============================================================

def add_rolling_volatility(df, windows=(10, 20, 60)):
    """
    Volatilité réalisée (écart-type des rendements log roulants).
    Mesure l'agitation récente du marché sur plusieurs horizons.
    """
    log_ret = np.log(df['Close'].shift(1) / df['Close'].shift(2))
    for w in windows:
        df[f'rvol_{w}'] = log_ret.rolling(w).std()
    return df


def add_atr(df, windows=(7, 14)):
    """
    ATR (Average True Range) — Mesure l'amplitude de mouvement réelle.
    Utilise High, Low, et le Close précédent pour capturer les gaps.
    """
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    c_prev = df['Close'].shift(2)

    tr = pd.concat([
        h - l,
        (h - c_prev).abs(),
        (l - c_prev).abs()
    ], axis=1).max(axis=1)

    for w in windows:
        df[f'atr_{w}'] = tr.rolling(w).mean()
    return df


def add_parkinson_volatility(df, window=20):
    """
    Volatilité de Parkinson (High/Low estimator).
    Plus précise que la vol. réalisée car elle utilise le range intraday.
    Formule : sqrt( 1/(4*ln(2)) * (ln(H/L))^2 ) en roulant.
    """
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    log_hl_sq = (np.log(h / (l + 1e-9))) ** 2
    df['park_vol'] = np.sqrt(log_hl_sq.rolling(window).mean() / (4 * np.log(2)))
    return df


def add_garman_klass(df, window=20):
    """
    Volatilité de Garman-Klass — La plus précise pour la donnée OHLCV.
    Combine les 4 prix (Open, High, Low, Close) pour réduire le bruit.
    """
    o = df['Open'].shift(1)
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    c = df['Close'].shift(1)

    term1 = 0.5 * (np.log(h / (l + 1e-9))) ** 2
    term2 = (2 * np.log(2) - 1) * (np.log(c / (o + 1e-9))) ** 2
    gk = (term1 - term2).rolling(window).mean()
    df['gk_vol'] = np.sqrt(gk.clip(lower=0))
    return df


def add_volatility_ratio(df, short=10, long=40):
    """
    Volatility Ratio : vol courte / vol longue.
    > 1 = Le marché s'accélère (breakout potentiel).
    < 1 = Le marché se calme (consolidation).
    """
    log_ret = np.log(df['Close'].shift(1) / df['Close'].shift(2))
    vol_short = log_ret.rolling(short).std()
    vol_long  = log_ret.rolling(long).std()
    df['vol_ratio'] = vol_short / (vol_long + 1e-9)
    return df


def add_rolling_variance(df, windows=(20, 60)):
    """
    Variance roulante des rendements (volatilité au carré).
    Parfois plus discriminante que l'écart-type pour certains modèles ML.
    """
    log_ret = np.log(df['Close'].shift(1) / df['Close'].shift(2))
    for w in windows:
        df[f'rvar_{w}'] = log_ret.rolling(w).var()
    return df


def compute_all_volatility(df):
    """Applique toutes les fonctions Volatilité d'un seul appel."""
    df = add_rolling_volatility(df)
    df = add_atr(df)
    df = add_parkinson_volatility(df)
    df = add_garman_klass(df)
    df = add_volatility_ratio(df)
    df = add_rolling_variance(df)
    return df
