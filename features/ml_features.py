import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import load_data, save_data

def generate_targets_and_clean(df):
    """
    Module 7: Prévention du Leakage.
    1. Crée la cible (Target) en regardant le futur.
    2. Ne fait AUCUNE normalisation (StandardScaler) globale pour éviter le "Data Leakage". 
       La normalisation sera faite DANS le module de machine learning (après la coupure Train/Test).
    """
    data = df.copy()
    
    target_name = STRATEGY["target_horizon"]
    freq = STRATEGY.get("frequency", "15m")

    if freq == "15m":
        horizon = 4 if target_name == "1h_forward_return" else 1
    elif freq == "1h":
        horizon = 1 if target_name == "1h_forward_return" else 4
    else:
        horizon = 60 if target_name == "1h_forward_return" else 15

    # Le seul endroit où l'on regarde le futur (-horizon), c'est pour créer la réponse.
    data['target_return'] = data['Close'].shift(-horizon) / data['Close'] - 1

    if STRATEGY["signal_type"] == "directional_return":
        data['target'] = np.where(data['target_return'] > 0, 1, -1)
        data = data.dropna(subset=['target_return'])
    elif STRATEGY["signal_type"] == "continuous_return":
        # Cible continue : retour futur brut (régression)
        data['target'] = data['target_return']
        data = data.dropna(subset=['target_return'])

    return data

def run_ml_features():
    for symbol in STRATEGY["universe"]:
        filename = f"{symbol}_{STRATEGY['frequency']}_alpha"
        df = load_data(filename)

        if df is None:
            print(f"❌ [{symbol}] Facteurs introuvables — lancez d'abord alpha_factors.")
            continue

        df_final = generate_targets_and_clean(df)
        save_data(df_final, f"{symbol}_ML_Dataset")
        mode = "continue" if STRATEGY["signal_type"] == "continuous_return" else "binaire"
        print(f"✅ [{symbol}] Dataset ML prêt — {len(df_final.columns)} colonnes, cible {mode}")

if __name__ == "__main__":
    run_ml_features()
