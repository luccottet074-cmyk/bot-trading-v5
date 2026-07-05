import numpy as np
import pandas as pd

# ============================================================
# MOMENTUM FACTORS (Chapitre 4 - ML4T)
# ============================================================
# Le Momentum est l'un des facteurs les plus robustes en finance.
# Principe : "Les gagnants récents ont tendance à continuer à surperformer."
# Toutes les fonctions appliquent un .shift(1) pour éviter le Look-Ahead Bias.
# ============================================================

def add_momentum(df):
    """
    Retours bruts multi-horizons (adaptés à la fréquence 15 minutes).
    - 1 période  = 15 min
    - 4 périodes = 1 heure
    - 8 périodes = 2 heures
    - 16 périodes = 4 heures (une demi-journée)
    - 32 périodes = 1 journée entière de trading
    """
    c = df['Close'].shift(1)  # Décalage anti Look-Ahead Bias
    df['mom_1']  = c.pct_change(1)
    df['mom_4']  = c.pct_change(4)
    df['mom_8']  = c.pct_change(8)
    df['mom_16'] = c.pct_change(16)
    df['mom_32'] = c.pct_change(32)
    return df


def add_log_returns(df):
    """
    Retours logarithmiques multi-horizons.
    Plus stables mathématiquement que les pct_change, surtout sur longue période.
    log(P_t / P_{t-n}) = Stationnaire si les prix ne le sont pas.
    """
    c = df['Close'].shift(1)
    df['log_ret_1']  = np.log(c / c.shift(1))
    df['log_ret_4']  = np.log(c / c.shift(4))
    df['log_ret_16'] = np.log(c / c.shift(16))
    df['log_ret_32'] = np.log(c / c.shift(32))
    return df


def add_rsi(df, windows=(7, 14, 21)):
    """
    RSI (Relative Strength Index) sur 3 horizons.
    < 30 = survente (signal achat), > 70 = surachat (signal vente)
    """
    c = df['Close'].shift(1)
    for w in windows:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=w - 1, min_periods=w).mean()  # Lissage exponentiel (Wilder)
        avg_loss = loss.ewm(com=w - 1, min_periods=w).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    """
    MACD (Moving Average Convergence/Divergence).
    - Ligne MACD : EMA_12 - EMA_26
    - Signal : EMA_9 de la ligne MACD
    - Histogramme : MACD - Signal (mesure la vélocité du momentum)
    """
    c = df['Close'].shift(1)
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    df['macd_line']  = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist']  = df['macd_line'] - df['macd_signal']
    return df


def add_rate_of_change(df, windows=(5, 20)):
    """
    ROC (Rate of Change) : variation brute en % sur n périodes.
    Différent du pct_change : met l'accent sur la vitesse de changement.
    ROC = (Close_t - Close_{t-n}) / Close_{t-n} * 100
    """
    c = df['Close'].shift(1)
    for w in windows:
        df[f'roc_{w}'] = ((c - c.shift(w)) / (c.shift(w) + 1e-9)) * 100
    return df


def add_stochastic(df, k_window=14, d_window=3):
    """
    Oscillateur Stochastique (%K et %D).
    Mesure la position du Close par rapport au range High-Low récent.
    %K : Position relative actuelle.
    %D : Lissage de %K (= signal de confirmation du momentum).
    """
    high = df['High'].shift(1)
    low  = df['Low'].shift(1)
    close = df['Close'].shift(1)

    lowest_low   = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    df['stoch_k'] = ((close - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=d_window).mean()
    return df


def add_lagged_returns(df, lags=(1, 2, 5, 10, 21)):
    """
    Retours décalés (Lagged Returns) — Critique pour les modèles ML (Chapitre 4 - ML4T).

    Pourquoi c'est indispensable :
    Les modèles ML (Random Forest, XGBoost) n'ont pas de mémoire native.
    En leur passant les rendements des N dernières périodes, on leur donne
    un "historique" artificiel pour capturer la dynamique récente.

    Double protection anti-biais :
    - .pct_change() calcule le rendement de T-1 à T.
    - .shift(lag) décale ensuite de `lag` périodes en arrière.
    → La valeur en ligne T représente bien le rendement qui s'est terminé à T-lag.
    """
    ret = df['Close'].pct_change().shift(1)  # rendement de base, décalé d'1 pour le bias
    for lag in lags:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    return df


def compute_all_momentum(df):
    """Applique toutes les fonctions Momentum d'un seul appel."""
    df = add_momentum(df)
    df = add_log_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_rate_of_change(df)
    df = add_stochastic(df)
    df = add_lagged_returns(df)  # ← Lagged Returns (Chapitre 4)
    return df
