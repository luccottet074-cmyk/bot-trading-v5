import pandas as pd

def run_pre_ml_validation(df, target_col='target_return', feature_cols=None):
    """
    Module 6: Validation Statistique avant Entraînement (Pre-ML Validation).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("⚠️ 'statsmodels' non installé — test ADF ignoré.")
        return True

    if target_col not in df.columns:
        print(f"❌ Colonne cible '{target_col}' introuvable.")
        return False

    series = df[target_col].dropna()
    if len(series) < 30:
        print("❌ Pas assez de données pour le test ADF.")
        return False

    result = adfuller(series)
    p_value = result[1]

    if p_value >= 0.05:
        print(f"🚨 Série NON stationnaire (ADF p={p_value:.4f}) — risque d'instabilité du modèle.")
        return False
    return True
