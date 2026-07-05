import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore', category=stats.ConstantInputWarning if hasattr(stats, 'ConstantInputWarning') else UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import load_data, save_data

# Colonnes à exclure de l'évaluation (pas des features)
NON_FEATURE_COLS = {'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'target_return'}

N_QUANTILES = 5
ROLLING_WINDOW = 20


def compute_ic(df, features, target_col='target_return'):
    """
    Information Coefficient (IC) = corrélation de Spearman entre chaque feature et le forward return.
    IC proche de 0 = facteur sans signal. IC > 0.05 = facteur exploitable.
    """
    records = []
    target = df[target_col].dropna()

    for feat in features:
        if feat not in df.columns:
            continue
        aligned = df[feat].reindex(target.index).dropna()
        common = target.reindex(aligned.index).dropna()
        aligned = aligned.reindex(common.index)

        if len(common) < 30:
            continue

        ic, pval = stats.spearmanr(aligned, common)
        records.append({
            'feature':  feat,
            'ic_mean':  round(ic, 4),
            'ic_abs':   round(abs(ic), 4),
            'p_value':  round(pval, 4),
            'significant': pval < 0.05
        })

    ic_df = pd.DataFrame(records).sort_values('ic_abs', ascending=False).reset_index(drop=True)
    return ic_df


def compute_rolling_ic(df, features, target_col='target_return', window=ROLLING_WINDOW):
    """
    IC glissant sur `window` périodes pour chaque feature.
    Mesure la stabilité temporelle du signal : un bon facteur a un IC stable, pas erratique.
    """
    records = []
    target = df[target_col]

    for feat in features[:10]:  # Top 10 seulement pour éviter un rapport trop lourd
        if feat not in df.columns:
            continue
        rolling_ics = []
        for i in range(window, len(df)):
            window_df = df.iloc[i - window:i]
            x = window_df[feat].dropna()
            y = target.reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(y) < 10:
                rolling_ics.append(np.nan)
                continue
            ic, _ = stats.spearmanr(x, y)
            rolling_ics.append(ic)

        ic_series = pd.Series(rolling_ics)
        records.append({
            'feature':     feat,
            'ic_mean':     round(ic_series.mean(), 4),
            'ic_std':      round(ic_series.std(), 4),
            'ic_ir':       round(ic_series.mean() / (ic_series.std() + 1e-9), 4),  # Information Ratio du facteur
            'ic_positive_pct': round((ic_series > 0).mean() * 100, 1)
        })

    return pd.DataFrame(records).sort_values('ic_ir', ascending=False).reset_index(drop=True)


def compute_quantile_returns(df, features, target_col='target_return', n_quantiles=N_QUANTILES):
    """
    Pour chaque feature, divise les observations en N quantiles et calcule le rendement moyen par quantile.
    Un bon facteur montre une relation monotone : Q1 (bas) → rendement bas, Q5 (haut) → rendement haut.
    Le spread Q5-Q1 mesure la puissance de séparation du facteur.
    """
    records = []
    for feat in features[:15]:  # Top 15
        if feat not in df.columns:
            continue
        tmp = df[[feat, target_col]].dropna()
        if len(tmp) < n_quantiles * 10:
            continue
        try:
            tmp['quantile'] = pd.qcut(tmp[feat], q=n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            continue
        q_returns = tmp.groupby('quantile')[target_col].mean()
        spread = q_returns.iloc[-1] - q_returns.iloc[0] if len(q_returns) >= 2 else np.nan
        row = {'feature': feat, 'spread_Q5_Q1': round(spread, 6)}
        for q in range(n_quantiles):
            row[f'Q{q+1}'] = round(q_returns.get(q, np.nan), 6)
        records.append(row)

    return pd.DataFrame(records).sort_values('spread_Q5_Q1', ascending=False).reset_index(drop=True)


def run_factor_evaluation():
    for symbol in STRATEGY["universe"]:
        df = load_data(f"{symbol}_ML_Dataset")
        if df is None:
            print(f"❌ [{symbol}] Dataset introuvable — lancez d'abord ml_features.")
            continue

        features = [c for c in df.columns if c not in NON_FEATURE_COLS]
        if 'target_return' not in df.columns:
            print(f"❌ [{symbol}] Colonne 'target_return' absente.")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic_df = compute_ic(df, features)
            stability_df = compute_rolling_ic(df, ic_df['feature'].tolist())
            quant_df = compute_quantile_returns(df, ic_df['feature'].tolist())

        save_data(ic_df, f"{symbol}_factor_ic")
        save_data(stability_df, f"{symbol}_factor_stability")
        save_data(quant_df, f"{symbol}_factor_quantiles")

        top3 = ic_df.head(3)
        ic_summary = " | ".join(
            f"{r['feature']}={r['ic_mean']:+.3f}{'**' if r['significant'] else ''}"
            for _, r in top3.iterrows()
        )
        print(f"✅ [{symbol}] Facteurs évalués — Top IC: {ic_summary}")


if __name__ == "__main__":
    run_factor_evaluation()
