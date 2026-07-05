import numpy as np
import pandas as pd

# ============================================================
# SEASONALITY & TIME FEATURES (Chapitre 4 - ML4T)
# ============================================================
# Les marchés financiers ont des cycles et des "effets calendaires" répétables.
# Ex: Plus de volume le lundi matin, moins le vendredi soir.
# Ces features permettent au modèle ML de "savoir l'heure" et de s'adapter.
# PAS de .shift(1) ici : l'heure et le jour de la bougie sont connus au moment T.
# ============================================================

def add_time_features(df):
    """
    Variables calendaires brutes.
    Extrait les composantes temporelles de l'index DatetimeIndex.
    """
    df['hour']         = df.index.hour
    df['day_of_week']  = df.index.dayofweek     # 0=Lundi … 4=Vendredi
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month']        = df.index.month
    return df


def add_cyclic_encoding(df):
    """
    Encodage cyclique (sin/cos) des variables temporelles.
    Problème sans encodage : le modèle pense que 23h et 0h sont "loin",
    alors qu'ils sont consécutifs. Le sin/cos corrige ça.
    """
    # Heure (cycle de 24h)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Jour de la semaine (cycle de 5 jours ouvrés)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # Mois (cycle de 12 mois)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def add_market_session_flags(df):
    """
    Flags booléens des effets de session.
    Capture les anomalies intraday connues (plus de vol à l'ouverture et à la clôture).
    Basé sur l'heure UTC (Paris = UTC+1 hiver, UTC+2 été).
    Ouverture Euronext : ~08h00 UTC, Clôture : ~16h30 UTC.
    """
    h = df.index.hour

    # Première heure de trading (forte volatilité d'ouverture)
    df['is_open_hour']  = ((h >= 8) & (h < 9)).astype(int)

    # Dernière heure de trading (forte volatilité de clôture)
    df['is_close_hour'] = ((h >= 15) & (h < 17)).astype(int)

    # Milieu de journée (faible volatilité, spread élevé)
    df['is_lunch']      = ((h >= 12) & (h < 14)).astype(int)
    return df


def add_calendar_effects(df):
    """
    Effets calendaires reconnus dans la littérature (anomalies de marché).
    - Monday Effect : Le lundi est statistiquement le pire jour de la semaine.
    - Turn of Month : Les 2 derniers et 3 premiers jours du mois surperforment
      (Flux institutionnels : pensions de retraite, rééquilibrages de fonds).
    """
    # Monday Effect (0 = Lundi)
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)

    # Turn-of-Month Effect (jours 28-31 et 1-3)
    d = df.index.day
    df['is_turn_of_month'] = ((d >= 28) | (d <= 3)).astype(int)
    return df


def add_eod_features(df):
    """
    Minutes restantes avant la clôture NYSE (16h00 ET = 21h00 UTC en été).
    Permet au modèle d'apprendre que les trades tardifs ont un profil différent
    (spreads élargis, gap overnight).
    """
    # 16h ET = 20h UTC (heure d'été) / 21h UTC (heure d'hiver)
    # On utilise 20h UTC comme approximation (heure d'été couvre la majorité de la période)
    close_utc = 20
    idx_utc = df.index.tz_convert('UTC') if df.index.tzinfo is not None else df.index
    minutes_since_midnight = idx_utc.hour * 60 + idx_utc.minute
    close_minutes = close_utc * 60
    df['minutes_to_close'] = np.clip(close_minutes - minutes_since_midnight, 0, None)
    df['is_last_hour']     = (df['minutes_to_close'] < 60).astype(int)
    df['is_last_2h']       = (df['minutes_to_close'] < 120).astype(int)
    return df


def compute_all_seasonality(df):
    """Applique toutes les fonctions de Saisonnalité d'un seul appel."""
    df = add_time_features(df)
    df = add_cyclic_encoding(df)
    df = add_market_session_flags(df)
    df = add_calendar_effects(df)
    df = add_eod_features(df)
    return df
