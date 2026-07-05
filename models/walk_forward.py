import os
import sys
import warnings
import math
from collections import Counter
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

warnings.filterwarnings('ignore')

from strategy_definition import STRATEGY
from data_pipeline.data_storage import load_data, save_data
from models.training import prepare_data, ic_score

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("⚠️ scikit-learn non installé.")
    exit()

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

N_FOLDS = 5

# Horizon forward en barres — même logique que ml_features.py
_freq   = STRATEGY.get('frequency', '1h')
_target = STRATEGY.get('target_horizon', '1h_forward_return')
if _freq == '15m':
    FORWARD_HORIZON = 4 if _target == '1h_forward_return' else 1
elif _freq == '1m':
    FORWARD_HORIZON = 60 if _target == '1h_forward_return' else 15
else:  # 1h, 1d
    FORWARD_HORIZON = 1 if _target == '1h_forward_return' else 4


def run_walk_forward():
    summary_rows = []
    for symbol in STRATEGY["universe"]:
        df = load_data(f"{symbol}_ML_Dataset")
        if df is None:
            print(f"❌ [{symbol}] Dataset introuvable — lancez d'abord ml_features.")
            continue

        X_raw, y = prepare_data(df)
        n = len(X_raw)
        test_size  = n // (N_FOLDS + 1)
        init_train = n - N_FOLDS * test_size

        results = []

        horizon = FORWARD_HORIZON
        embargo = max(horizon, math.ceil(0.01 * test_size))

        for fold in range(N_FOLDS):
            train_end  = init_train + fold * test_size
            test_start = train_end
            test_end   = min(test_start + test_size, n)

            # --- Sans purging (méthode actuelle) ---
            X_tr = X_raw.iloc[:train_end]
            X_te = X_raw.iloc[test_start:test_end]
            y_tr = y.iloc[:train_end]
            y_te = y.iloc[test_start:test_end]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=20,
                                       max_features='sqrt', random_state=42, n_jobs=-1)
            rf.fit(X_tr_s, y_tr)
            ic_rf = ic_score(y_te, rf.predict(X_te_s))

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr)
            ic_ridge = ic_score(y_te, ridge.predict(X_te_s))

            ic_xgb = 0.0
            if XGB_AVAILABLE:
                xgb = XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective='reg:squarederror', verbosity=0,
                    random_state=42, n_jobs=-1
                )
                xgb.fit(X_tr_s, y_tr)
                ic_xgb = ic_score(y_te, xgb.predict(X_te_s))

            fold_scores = {"RFR": ic_rf, "Ridge": ic_ridge}
            if XGB_AVAILABLE:
                fold_scores["XGB"] = ic_xgb
            best_fold_model = max(fold_scores, key=fold_scores.get)
            ic_best = fold_scores[best_fold_model]

            # --- Avec purging + embargo (López de Prado) ---
            # Purging : retire les `horizon` dernières barres d'entraînement
            # dont les labels chevauchent la fenêtre de test.
            purge_end = max(train_end - horizon, 1)
            X_tr_p = X_raw.iloc[:purge_end]
            y_tr_p = y.iloc[:purge_end]
            # Embargo : exclut les premières `embargo` barres du test de l'éval IC
            # (corrélation résiduelle avec les features proches de la frontière).
            X_te_p = X_te.iloc[embargo:] if len(X_te) > embargo else X_te
            y_te_p = y_te.iloc[embargo:] if len(y_te) > embargo else y_te

            scaler_p = StandardScaler()
            X_tr_p_s = scaler_p.fit_transform(X_tr_p)
            X_te_p_s = scaler_p.transform(X_te_p)

            rf_p = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=20,
                                         max_features='sqrt', random_state=42, n_jobs=-1)
            rf_p.fit(X_tr_p_s, y_tr_p)
            ic_rf_p = ic_score(y_te_p, rf_p.predict(X_te_p_s))

            ridge_p = Ridge(alpha=10.0)
            ridge_p.fit(X_tr_p_s, y_tr_p)
            ic_ridge_p = ic_score(y_te_p, ridge_p.predict(X_te_p_s))

            ic_xgb_p = 0.0
            if XGB_AVAILABLE:
                xgb_p = XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective='reg:squarederror', verbosity=0,
                    random_state=42, n_jobs=-1
                )
                xgb_p.fit(X_tr_p_s, y_tr_p)
                ic_xgb_p = ic_score(y_te_p, xgb_p.predict(X_te_p_s))

            fold_scores_p = {"RFR": ic_rf_p, "Ridge": ic_ridge_p}
            if XGB_AVAILABLE:
                fold_scores_p["XGB"] = ic_xgb_p
            ic_best_p = fold_scores_p[best_fold_model]  # même modèle pour comparaison équitable

            # Baseline momentum
            mom = X_raw.iloc[test_start:test_end].get('mom_1', pd.Series(0, index=y_te.index))
            ic_mom = ic_score(y_te, mom.values)

            results.append({
                'fold':        fold + 1,
                'debut':       df.index[test_start].date(),
                'fin':         df.index[test_end - 1].date(),
                'n_bars':      test_end - test_start,
                'ic':          round(ic_best, 4),
                'ic_purged':   round(ic_best_p, 4),
                'n_purged':    horizon,
                'n_embargoed': embargo,
                'baseline':    round(ic_mom, 4),
                'delta':       round(ic_best - ic_mom, 4),
                'beats':       ic_best > 0,
                'model':       best_fold_model,
            })

        mean_ic    = np.mean([r['ic'] for r in results])
        mean_ic_p  = np.mean([r['ic_purged'] for r in results])
        mean_base  = np.mean([r['baseline'] for r in results])
        mean_delta = np.mean([r['delta'] for r in results])
        n_wins     = sum(r['beats'] for r in results)
        dominant   = Counter(r['model'] for r in results).most_common(1)[0][0]

        if n_wins >= 4:
            verdict = "✅ ROBUSTE"
        elif n_wins >= 3:
            verdict = "⚠️  INSTABLE"
        else:
            verdict = "❌ FRAGILE"

        summary_rows.append((symbol, mean_ic, mean_ic_p, mean_base, mean_delta, n_wins, verdict))

        results_df = pd.DataFrame(results)
        save_data(results_df, f"{symbol}_walk_forward")

    if len(summary_rows) > 1:
        print("\n" + "="*62)
        print(f"{'RÉSUMÉ WALK-FORWARD':^62}")
        print("="*62)
        print(f"  {'Ticker':<8}  {'IC moy':>7}  {'Naïf':>7}  {'Delta':>8}  {'Folds':>5}  Verdict")
        print("  " + "-"*54)
        for sym, ic, ic_p, base, delta, wins, verdict in summary_rows:
            print(f"  {sym:<8}  {ic:>+7.3f}  {base:>+7.3f}  {delta:>+8.3f}  {wins}/{N_FOLDS}   {verdict}")
        print("="*62)

        # Comparaison purging/embargo (López de Prado) vs méthode actuelle
        print(f"\n  Purging/Embargo — biais résiduel (horizon={FORWARD_HORIZON}b, embargo={max(FORWARD_HORIZON, math.ceil(0.01 * (len(X_raw) // (N_FOLDS + 1))))}b)")
        print(f"  {'Ticker':<8}  {'Sans purge':>10}  {'Avec purge':>10}  {'Delta':>8}")
        print("  " + "-"*44)
        for sym, ic, ic_p, base, delta, wins, verdict in summary_rows:
            bias = ic_p - ic
            print(f"  {sym:<8}  {ic:>+10.4f}  {ic_p:>+10.4f}  {bias:>+8.4f}")


if __name__ == "__main__":
    run_walk_forward()
