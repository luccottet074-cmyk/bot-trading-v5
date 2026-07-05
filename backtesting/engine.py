import os
import sys
import pandas as pd
import numpy as np

# Ajouter la racine au sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import DATA_DIR, PARQUET_DIR, CSV_DIR
from backtesting.risk import apply_stop_loss

# === PARAMÈTRES DU BACKTEST ===
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
SLIPPAGE = 0.0005
TOTAL_COST = FEE_RATE + SLIPPAGE

# Stop Loss / Take Profit par ticker (SL, TP)
# GOOGL/MSFT : DD faible → on laisse courir les profits plus longtemps
_TICKER_RISK = {
    'AAPL':  (0.02, 0.03),
    'MSFT':  (0.02, 0.04),
    'GOOGL': (0.02, 0.05),
    'COST':  (0.02, 0.03),
}
_DEFAULT_RISK = (0.02, 0.03)

# Fenêtre MA du filtre downtrend par ticker.
# COST : MA_20 (plus réactive) car MA_50 bloque trop longtemps les phases de recovery.
# Autres : MA_50 standard.
_TICKER_MA_WINDOW = {
    'AAPL':  30,
    'MSFT':  50,
    'GOOGL': 50,
    'COST':  20,
}
_DEFAULT_MA_WINDOW = 50

def run_backtest(symbol):
    print(f"Exécution du Backtest pour {symbol}...")
    
    signals_parquet = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
    signals_csv     = os.path.join(CSV_DIR,     f"{symbol}_Signals.csv")
    
    if os.path.exists(signals_parquet):
        df = pd.read_parquet(signals_parquet)
    elif os.path.exists(signals_csv):
        df = pd.read_csv(signals_csv, index_col=0, parse_dates=True)
    else:
        print(f"⚠️ Fichier de signaux introuvable pour {symbol}.")
        return
    
    # 1. Anti Look-Ahead Bias — position graduée (0 / 0.5 / 1.0)
    #    + Rétention de signal : si signal=0 mais actif dans les N_HOLD_BARS précédentes,
    #      on maintient la dernière taille connue (évite les sorties prématurées).
    N_HOLD_BARS = 2

    if 'Signal_Size' in df.columns:
        raw = df['Signal_Size'].shift(1).fillna(0.0)
    else:
        raw = pd.Series(
            np.where(df['Signal'].shift(1).fillna(0) == 1, 1.0, 0.0), index=df.index
        )

    retained = raw.copy()
    for i in range(1, N_HOLD_BARS + 1):
        # Condition : toutes les i-1 barres précédentes dans raw étaient à 0
        all_recent_zero = pd.Series(True, index=df.index)
        for j in range(1, i):
            all_recent_zero &= (raw.shift(j).fillna(0.0) == 0.0)
        had_signal = raw.shift(i).fillna(0.0) > 0.0
        fill_mask  = (retained == 0.0) & all_recent_zero & had_signal
        retained[fill_mask] = raw.shift(i).fillna(0.0)[fill_mask]

    df['Position'] = retained

    # 2. Filtre de régime de marché (basé uniquement sur le passé — no lookahead)
    #    2 conditions bloquantes :
    #    - Chaos     : vol_20 > 1.5x vol_100
    #    - Downtrend : Close < MA_N  (N configurable par ticker)
    ma_window = _TICKER_MA_WINDOW.get(symbol, _DEFAULT_MA_WINDOW)

    ret = df['Close'].pct_change()
    vol_20  = ret.rolling(20).std().shift(1)
    vol_100 = ret.rolling(100).std().shift(1)
    ma_n    = df['Close'].rolling(ma_window).mean().shift(1)

    high_vol  = vol_20 > 1.5 * vol_100
    downtrend = df['Close'] < ma_n

    risky_regime = (high_vol | downtrend).fillna(True)

    df['Regime'] = (~risky_regime).astype(int)
    df['Position'] = df['Position'] * df['Regime']

    # Vol-adjusted sizing : réduit la taille de 50 % quand la vol courte (10 barres)
    # dépasse la vol longue (100 barres) sans atteindre le seuil du filtre régime (×1.5).
    # Complémentaire au filtre : capte les zones de vol modérément élevée.
    vol_10    = ret.rolling(10).std().shift(1)
    vol_ratio = (vol_10 / vol_100.replace(0, np.nan)).fillna(1.0)
    moderate_vol = (vol_ratio > 1.0) & (vol_ratio < 1.5) & (df['Position'] > 0)
    df.loc[moderate_vol, 'Position'] = df.loc[moderate_vol, 'Position'] * 0.5
    n_vol_reduced = int(moderate_vol.sum())

    n_filtered  = (df['Regime'] == 0).sum()
    n_vol       = high_vol.sum()
    n_down      = downtrend.sum()
    active      = df[df['Position'] > 0]
    avg_size    = active['Position'].mean() * 100 if len(active) > 0 else 0
    print(f"   Régime : {n_filtered}/{len(df)} barres filtrées ({n_filtered/len(df)*100:.1f}% en pause)"
          f"  [chaos:{n_vol} | down:{n_down}]"
          f"  |  Vol réduite : {n_vol_reduced} barres"
          f"  |  Taille moy. position : {avg_size:.0f}%")

    # 3. Rendements bruts
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return_Gross'] = df['Position'] * df['Market_Return']
    
    # 3. Stop Loss 2 % + Take Profit 3 % (= 1.5 × SL)
    sl_pct, tp_pct = _TICKER_RISK.get(symbol, _DEFAULT_RISK)
    df = apply_stop_loss(df, stop_loss_pct=sl_pct, take_profit_pct=tp_pct)
    
    # 4. Transactions et Frais
    df['Trade'] = df['Position'].diff().fillna(0)
    df['Transaction_Costs'] = np.abs(df['Trade']) * TOTAL_COST
    
    df['Strategy_Return_Net'] = df['Strategy_Return_Gross'] - df['Transaction_Costs']
    
    # 5. Évolution du Capital
    df['Portfolio_Value'] = INITIAL_CAPITAL * (1 + df['Strategy_Return_Net']).cumprod()
    df['Buy_Hold_Value'] = INITIAL_CAPITAL * (1 + df['Market_Return']).cumprod()
    
    df = df.dropna()
    
    parquet_out = os.path.join(PARQUET_DIR, f"{symbol}_Backtest.parquet")
    df.to_parquet(parquet_out, compression='snappy')
    print(f"✅ Backtest terminé et sauvegardé")
    
    final_portfolio = df['Portfolio_Value'].iloc[-1]
    print(f"💰 Capital Final (Stratégie) : {final_portfolio:.2f} $")

def run_panel_backtest(symbols):
    """
    Exécute le backtest pour chaque ticker puis construit une equity curve combinée.
    Allocation égale : le capital total est réparti équitablement entre tous les tickers.
    """
    for symbol in symbols:
        run_backtest(symbol)

    # Chargement des backtests individuels
    dfs = {}
    for symbol in symbols:
        path = os.path.join(PARQUET_DIR, f"{symbol}_Backtest.parquet")
        if os.path.exists(path):
            dfs[symbol] = pd.read_parquet(path)

    if len(dfs) < 2:
        print("⚠️ Panel : moins de 2 backtests disponibles, agrégation impossible.")
        return

    # Intersection des index : dates communes à tous les tickers
    common_idx = None
    for df in dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    n_assets = len(dfs)

    # Chargement des poids Kelly (générés par features/ic_kelly.py avant engine)
    kelly_path = os.path.join(PARQUET_DIR, "kelly_weights.parquet")
    if os.path.exists(kelly_path):
        kw_df  = pd.read_parquet(kelly_path).set_index('symbol')
        raw_kw = {sym: float(kw_df.loc[sym, 'kelly_weight']) for sym in dfs if sym in kw_df.index}
        total  = sum(raw_kw.values())
        eq     = 1.0 / n_assets
        weights = {sym: raw_kw.get(sym, eq) / total for sym in dfs}
    else:
        weights = {sym: 1.0 / n_assets for sym in dfs}

    panel = pd.DataFrame(index=common_idx)
    panel['Panel_Return_Net'] = sum(
        dfs[sym]['Strategy_Return_Net'].reindex(common_idx).fillna(0) * weights[sym] for sym in dfs
    )
    panel['BH_Return'] = sum(
        dfs[sym]['Market_Return'].reindex(common_idx).fillna(0) * weights[sym] for sym in dfs
    )
    panel['Portfolio_Value'] = INITIAL_CAPITAL * (1 + panel['Panel_Return_Net']).cumprod()
    panel['Buy_Hold_Value']  = INITIAL_CAPITAL * (1 + panel['BH_Return']).cumprod()

    panel_path = os.path.join(PARQUET_DIR, "Panel_Backtest.parquet")
    panel.to_parquet(panel_path, compression='snappy')

    final_val = panel['Portfolio_Value'].iloc[-1]
    bh_val    = panel['Buy_Hold_Value'].iloc[-1]
    ret_pct   = (final_val / INITIAL_CAPITAL - 1) * 100
    bh_pct    = (bh_val   / INITIAL_CAPITAL - 1) * 100

    print(f"\n{'='*55}")
    print(f"📊 PANEL BACKTEST — {n_assets} tickers ({', '.join(dfs.keys())})")
    print(f"{'='*55}")
    print(f"  Capital ML (pondéré) : {final_val:.2f} $  ({ret_pct:+.2f}%)")
    print(f"  Capital B&H (pond.)  : {bh_val:.2f} $  ({bh_pct:+.2f}%)")
    print(f"  Sauvegardé           : Panel_Backtest.parquet")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_panel_backtest(STRATEGY["universe"])
