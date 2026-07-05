import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import load_data, save_data

# --- Import de toutes les familles de facteurs ---
from features.momentum       import compute_all_momentum
from features.volatility     import compute_all_volatility
from features.volume         import compute_all_volume
from features.seasonality    import compute_all_seasonality
from features.denoising      import compute_all_denoising
from features.market_context import compute_all_market_context
from features.intraday         import compute_all_intraday
from features.ibkr_derivatives import compute_all_derivatives
from features.ibkr_opening     import compute_all_opening


def build_alpha_factors(df, symbol=""):
    """
    Orchestrateur Central des Alpha Factors (Chapitre 4 - ML4T).

    Applique séquentiellement toutes les familles de facteurs sur le DataFrame brut.
    Chaque module peut être activé/désactivé indépendamment.
    Les NaN finaux sont supprimés pour garantir un dataset propre pour le ML.
    """
    initial_cols = len(df.columns)

    df = compute_all_momentum(df)
    df = compute_all_volatility(df)
    df = compute_all_volume(df)
    df = compute_all_seasonality(df)
    df = compute_all_denoising(df)
    df = compute_all_market_context(df, symbol=symbol)
    df = compute_all_intraday(df, symbol=symbol)       # microstructure IBKR 15min
    df = compute_all_derivatives(df, symbol=symbol)    # IV / VRP IBKR (exclusif)
    df = compute_all_opening(df)                       # features ouverture RTH IBKR

    df = df.dropna()
    new_cols = len(df.columns) - initial_cols
    print(f"✅ [{symbol}] {new_cols} alpha factors ({len(df)} lignes valides)")
    return df


def run_alpha_factors():
    """Point d'entrée principal appelé par run_all.py."""
    for symbol in STRATEGY["universe"]:
        filename = f"{symbol}_{STRATEGY['frequency']}"
        df = load_data(filename)

        if df is None:
            print(f"❌ [{symbol}] Données brutes introuvables — lancez d'abord data_pipeline.")
            continue

        df_alpha = build_alpha_factors(df, symbol=symbol)
        save_data(df_alpha, f"{filename}_alpha")


if __name__ == "__main__":
    run_alpha_factors()
