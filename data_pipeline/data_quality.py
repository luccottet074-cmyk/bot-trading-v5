import pandas as pd
import numpy as np

def run_quality_checks(df, symbol="Dataset"):
    if df is None or df.empty:
        return {"error": "Le dataframe est vide ou inexistant."}
        
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    missing_values = df.isna().sum().to_dict()
    total_missing = sum(missing_values.values())
    
    duplicate_rows = int(df.duplicated().sum())
    
    date_gaps = {}
    if not df.index.empty:
        date_gaps = df.index.to_series().diff().value_counts().to_dict()
        date_gaps = {str(k): v for k, v in date_gaps.items()}
        
    negative_prices = 0
    zero_volume = 0
    if 'Close' in df.columns:
        negative_prices = int((df['Close'] < 0).sum())
    if 'Volume' in df.columns:
        zero_volume = int((df['Volume'] == 0).sum())
        
    unsorted_index = not df.index.is_monotonic_increasing

    report = {
        "missing_columns": missing_cols,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "negative_prices": negative_prices,
        "zero_volume": zero_volume,
        "unsorted_index": unsorted_index,
        "date_gaps": date_gaps
    }
    
    if total_missing > 0:
        print(f"     ⚠️ {total_missing} valeurs manquantes détectées.")
    if duplicate_rows > 0:
        print(f"     ⚠️ {duplicate_rows} lignes en doublon.")
    if negative_prices > 0:
        print(f"     🚨 PRIX NÉGATIFS DÉTECTÉS ({negative_prices} occurrences) !")
    if unsorted_index:
        print(f"     🚨 L'index temporel n'est pas trié chronologiquement !")
        
    return report
