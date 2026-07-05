import pandas as pd

def align_time_series(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    return df

def merge_multiple_tickers(dfs_dict):
    print("  🔄 Fusion des DataFrames...")
    aligned = pd.concat(dfs_dict.values(), axis=1, keys=dfs_dict.keys(), join='inner')
    return aligned
