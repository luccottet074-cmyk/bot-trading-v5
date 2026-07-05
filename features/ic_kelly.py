"""
Kelly weighting basé sur l'IC (Information Coefficient) par ticker.
IC = corrélation entre le rendement prédit et le rendement réel à t+1.
Plus l'IC est élevé, plus le ticker reçoit de capital dans le panel.
"""
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import PARQUET_DIR

KELLY_CAP   = 1.5   # poids max = 1.5× equal-weight
KELLY_FLOOR = 0.5   # poids min = 0.5× equal-weight


def _compute_ic(symbol):
    """Retourne l'IC du ticker sur tout le jeu de test, ou None si données insuffisantes."""
    sig_path = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
    if not os.path.exists(sig_path):
        return None

    df = pd.read_parquet(sig_path)
    if 'Predicted_Return' not in df.columns or 'Close' not in df.columns:
        return None

    actual = df['Close'].pct_change().shift(-1)
    pred   = df['Predicted_Return']
    common = pred.dropna().index.intersection(actual.dropna().index)

    if len(common) < 30:
        return None

    ic = np.corrcoef(pred.loc[common], actual.loc[common])[0, 1]
    return float(ic) if not np.isnan(ic) else None


def compute_kelly_weights(symbols):
    """
    Calcule les poids Kelly normalisés pour chaque ticker.
    Retourne un dict {symbol: weight} avec sum(weights) = 1.
    """
    ics = {}
    for sym in symbols:
        ic = _compute_ic(sym)
        ics[sym] = max(ic, 0.0) if ic is not None else 0.0
        label = f"{ics[sym]:+.4f}" if ic is not None else "N/A"
        print(f"   IC {sym}: {label}")

    n        = len(symbols)
    total_ic = sum(ics.values())

    if total_ic == 0:
        raw = {sym: 1.0 for sym in symbols}
        print("   ⚠️  IC tous nuls → fallback equal-weight")
    else:
        raw = {sym: ics[sym] / total_ic * n for sym in symbols}

    capped  = {sym: float(np.clip(raw[sym], KELLY_FLOOR, KELLY_CAP)) for sym in symbols}
    total   = sum(capped.values())
    weights = {sym: capped[sym] / total for sym in symbols}

    df_w = pd.DataFrame([
        {'symbol': sym, 'kelly_weight': weights[sym], 'ic': ics[sym]}
        for sym in symbols
    ])
    out_path = os.path.join(PARQUET_DIR, "kelly_weights.parquet")
    df_w.to_parquet(out_path, compression='snappy')

    eq = 1.0 / n
    print(f"\n   {'Ticker':<8}  {'IC':>7}  {'Poids':>8}  {'vs equal-w':>10}")
    print("   " + "-" * 38)
    for sym in symbols:
        diff = weights[sym] / eq - 1
        print(f"   {sym:<8}  {ics[sym]:>7.4f}  {weights[sym]:>7.3f}×  ({diff:+.1%})")

    return weights


if __name__ == "__main__":
    symbols = STRATEGY['universe']
    print(f"\n{'='*50}")
    print("📐 KELLY WEIGHTING — IC par ticker")
    print(f"{'='*50}")
    compute_kelly_weights(symbols)
    print("✅ kelly_weights.parquet sauvegardé")
