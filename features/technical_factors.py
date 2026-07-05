import os
import sys
import numpy as np
import pandas as pd

# Ajouter la racine au sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import DATA_DIR, load_data, save_data

def generate_technical_factors(df):
    """
    Module 7: Prévention du Look-Ahead Bias & Calcul Technique.
    Toutes les features sont décalées (.shift(1)) pour simuler la réalité :
    Au moment T, on ne peut prendre de décision qu'avec les données de T-1.
    """
    print("  ⚙️ Calcul des Facteurs Techniques (Technical Factors)...")
    data = df.copy()
    
    # Sécurisation anti Look-Ahead Bias : on décale tout le prix brut d'une période
    # pour le calcul des indicateurs (Ainsi le "Close" de la ligne T est en fait le Close de T-1)
    safe_data = data.shift(1)
    
    data["ret_1m"] = safe_data["Close"].pct_change(1)
    data["ret_5m"] = safe_data["Close"].pct_change(5)
    data["ret_15m"] = safe_data["Close"].pct_change(15)
    
    data["vol_20m"] = data["ret_1m"].rolling(window=20).std()
    
    high_low = safe_data['High'] - safe_data['Low']
    high_close = np.abs(safe_data['High'] - safe_data['Close'].shift())
    low_close = np.abs(safe_data['Low'] - safe_data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['atr_14'] = true_range.rolling(14).mean()
    
    data["volume_change"] = safe_data["Volume"].pct_change(1)
    data["volume_sma_20"] = safe_data["Volume"].rolling(20).mean()
    data["volume_ratio"] = safe_data["Volume"] / (data["volume_sma_20"] + 1e-8)
    
    bb_mean = safe_data['Close'].rolling(window=20).mean()
    bb_std = safe_data['Close'].rolling(window=20).std()
    data['bb_bandwidth'] = (bb_std * 4) / bb_mean
    
    delta = safe_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = safe_data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = safe_data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    
    # On garde les prix originaux pour le calcul de rentabilité future, 
    # mais les features ne regardent que le passé !
    data = data.dropna()
    return data

def run_technical_factors():
    print("=== DÉBUT DE L'INGÉNIERIE FINANCIÈRE (TECHNICAL FACTORS) ===")
    for symbol in STRATEGY["universe"]:
        filename = f"{symbol}_{STRATEGY['frequency']}"
        df = load_data(filename)
        
        if df is None:
            print(f"⚠️ Données brutes introuvables pour {symbol}.")
            continue
            
        df_features = generate_technical_factors(df)
        save_data(df_features, f"{filename}_tech_factors")
        print(f"✅ Facteurs techniques (Bias-Free) générés : {len(df_features.columns)} colonnes.")

if __name__ == "__main__":
    run_technical_factors()
