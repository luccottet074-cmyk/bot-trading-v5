import numpy as np
import pandas as pd

# ============================================================
# VOLUME FACTORS (Chapitre 4 - ML4T)
# ============================================================
# Le volume est la "conviction" derrière un mouvement de prix.
# Un prix qui monte AVEC du volume = signal fort.
# Un prix qui monte SANS volume = signal faible (probable faux breakout).
# Toutes les fonctions appliquent un .shift(1) pour éviter le Look-Ahead Bias.
# ============================================================

def add_volume_zscore(df, window=20):
    """
    Volume Z-Score : Normalise le volume par rapport à sa moyenne roulante.
    Un z-score > 2 = Spike de volume anormal (acheteurs/vendeurs agressifs).
    """
    v = df['Volume'].shift(1)
    mean = v.rolling(window).mean()
    std  = v.rolling(window).std()
    df['vol_zscore'] = (v - mean) / (std + 1e-9)
    return df


def add_volume_percentile_rank(df, window=50):
    """
    Rang Percentile du Volume sur une fenêtre glissante.
    Indique si le volume actuel est dans le top 20% ou dans le bas 20% historique.
    Très utile car insensible aux changements d'échelle (splits, etc.).
    """
    v = df['Volume'].shift(1)
    df['vol_rank'] = v.rolling(window).rank(pct=True)
    return df


def add_obv(df):
    """
    OBV (On-Balance Volume) — Accumulation/Distribution du volume.
    Si le Close monte → on ajoute le volume.
    Si le Close baisse → on soustrait le volume.
    La tendance de l'OBV prédit souvent la tendance du prix.
    """
    c = df['Close'].shift(1)
    v = df['Volume'].shift(1)
    direction = np.sign(c.diff())
    df['obv'] = (direction * v).fillna(0).cumsum()
    return df


def add_chaikin_money_flow(df, window=20):
    """
    Chaikin Money Flow (CMF) — Variante du Chaikin Oscillator.
    Mesure si l'argent "entre" ou "sort" du marché via le rapport prix/volume.
    CMF > 0 = Pression acheteuse dominante.
    CMF < 0 = Pression vendeuse dominante.
    """
    h = df['High'].shift(1)
    l = df['Low'].shift(1)
    c = df['Close'].shift(1)
    v = df['Volume'].shift(1)

    # Money Flow Multiplier : position du Close dans le range High-Low
    mfm = ((c - l) - (h - c)) / (h - l + 1e-9)
    # Money Flow Volume
    mfv = mfm * v
    df['cmf'] = mfv.rolling(window).sum() / (v.rolling(window).sum() + 1e-9)
    return df


def add_volume_spike(df, window=20, threshold=2.0):
    """
    Volume Spike Indicator : Signal binaire (0/1).
    Vaut 1 quand le volume dépasse `threshold` fois sa moyenne roulante.
    Capture les moments d'activité institutionnelle intense.
    """
    v = df['Volume'].shift(1)
    vol_mean = v.rolling(window).mean()
    df['vol_spike'] = (v > threshold * vol_mean).astype(int)
    return df


def compute_all_volume(df):
    """Applique toutes les fonctions Volume d'un seul appel."""
    df = add_volume_zscore(df)
    df = add_volume_percentile_rank(df)
    df = add_obv(df)
    df = add_chaikin_money_flow(df)
    df = add_volume_spike(df)
    return df
