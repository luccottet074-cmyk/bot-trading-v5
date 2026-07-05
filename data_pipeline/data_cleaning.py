import pandas as pd

def clean_data(df):
    df_clean = df.copy()
    
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_clean = df_clean[[c for c in cols_to_keep if c in df_clean.columns]]
    
    if not df_clean.index.is_monotonic_increasing:
        df_clean = df_clean.sort_index()
        
    df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
    df_clean = df_clean.ffill().bfill()
    
    if 'Close' in df_clean.columns:
        df_clean = df_clean[df_clean['Close'] > 0]
        
    if 'Volume' in df_clean.columns:
        df_clean = df_clean[df_clean['Volume'] > 0]
    
    return df_clean
