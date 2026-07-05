import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone('America/New_York')


def add_opening_features(df):
    """Features specifiques a la barre d'ouverture RTH IBKR (9:30 ET).

    IBKR fournit une barre labellee heure=9 ET (9:30-10:30 ET) absente d'Alpaca.
    Ces 3 features capturent la dynamique d'ouverture et sont propagees
    a toutes les barres de la meme journee de trading :

    - open_gap        : ecart Open_9h30 / Close_veille (gap overnight)
    - open_vol_ratio  : volume d'ouverture / volume median des barres normales (J-1)
    - open_range_pct  : amplitude High-Low de la barre d'ouverture / Close
    """
    # Convertir en ET pour identifier l'heure de chaque barre
    if df.index.tz is None:
        idx_et = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert(ET)
    else:
        idx_et = df.index.tz_convert(ET)

    df_et = df.copy()
    df_et.index = idx_et
    df_et['_date'] = idx_et.date
    df_et['_hour'] = idx_et.hour

    open_mask = df_et['_hour'] == 9

    # Pas assez de barres d'ouverture -> features neutres (Alpaca ou donnees incompletes)
    if open_mask.sum() < 10:
        df['open_gap']       = 0.0
        df['open_vol_ratio'] = 1.0
        df['open_range_pct'] = 0.0
        return df

    # --- open_gap : (Open_9h30 - Close_barre_precedente) / Close_barre_precedente ---
    # La barre precedant la barre 9h30 est la derniere barre RTH de la veille
    prev_close = df_et['Close'].shift(1).loc[open_mask]
    open_price = df_et.loc[open_mask, 'Open']
    open_gap_s = (open_price - prev_close) / (prev_close.abs() + 1e-9)

    # --- open_vol_ratio : Volume_9h30 / Volume_median_barres_normales(J-1) ---
    normal_mask = df_et['_hour'].isin([10, 11, 12, 13, 14])
    vol_median_by_date = (
        df_et.loc[normal_mask]
        .groupby('_date')['Volume']
        .median()
        .shift(1)   # J-1 pour eviter look-ahead
    )
    open_dates    = df_et.loc[open_mask, '_date']
    vol_ref_open  = open_dates.map(vol_median_by_date)
    open_vol_s    = df_et.loc[open_mask, 'Volume'] / (vol_ref_open + 1)

    # --- open_range_pct : (High - Low)_9h30 / Close_9h30 ---
    h = df_et.loc[open_mask, 'High']
    l = df_et.loc[open_mask, 'Low']
    c = df_et.loc[open_mask, 'Close']
    open_range_s = (h - l) / (c.abs() + 1e-9)

    # Associer chaque feature a sa date de trading
    open_gap_d   = pd.Series(open_gap_s.values,  index=open_dates.values)
    open_vol_d   = pd.Series(open_vol_s.values,   index=open_dates.values)
    open_range_d = pd.Series(open_range_s.values, index=open_dates.values)

    # Propager a toutes les barres du meme jour (forward-fill intraday)
    df_et['open_gap']       = df_et['_date'].map(open_gap_d).fillna(0.0)
    df_et['open_vol_ratio'] = df_et['_date'].map(open_vol_d).fillna(1.0)
    df_et['open_range_pct'] = df_et['_date'].map(open_range_d).fillna(0.0)

    # Shift(1) : la barre 9h30 du jour J ne doit pas utiliser ses propres features
    # -> elle voit les features d'ouverture du jour J-1
    df_et['open_gap']       = df_et['open_gap'].shift(1).fillna(0.0)
    df_et['open_vol_ratio'] = df_et['open_vol_ratio'].shift(1).fillna(1.0)
    df_et['open_range_pct'] = df_et['open_range_pct'].shift(1).fillna(0.0)

    # Reassigner sur l'index original
    df['open_gap']       = df_et['open_gap'].values
    df['open_vol_ratio'] = df_et['open_vol_ratio'].values
    df['open_range_pct'] = df_et['open_range_pct'].values

    return df


def compute_all_opening(df):
    """Point d'entree pour les features d'ouverture IBKR."""
    return add_opening_features(df)
