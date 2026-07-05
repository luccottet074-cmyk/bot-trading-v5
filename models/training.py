import os
import sys
import pandas as pd
import numpy as np
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

warnings.filterwarnings('ignore')

from strategy_definition import STRATEGY
from data_pipeline.data_storage import DATA_DIR, PARQUET_DIR, CSV_DIR, load_data, save_data
from models.pre_ml_validation import run_pre_ml_validation

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    import joblib
except ImportError:
    print("⚠️ scikit-learn ou joblib n'est pas installé.")
    exit()

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBClassifierWrapper:
    """Conservé pour compatibilité avec les anciens fichiers .pkl — non utilisé en régression."""
    def __init__(self, xgb_inner, label_encoder=None):
        self.xgb = xgb_inner
        self._le = label_encoder
        self.classes_ = label_encoder.classes_ if label_encoder else None

    def predict(self, X):
        pred = self.xgb.predict(X)
        return self._le.inverse_transform(pred) if self._le else pred

    def predict_proba(self, X):
        return self.xgb.predict_proba(X)


def ic_score(y_true, y_pred):
    """Information Coefficient : corrélation de Spearman entre prédiction et réalité."""
    arr_t = np.array(y_true, dtype=float)
    arr_p = np.array(y_pred, dtype=float)
    mask  = np.isfinite(arr_t) & np.isfinite(arr_p)
    if mask.sum() < 3:
        return 0.0
    corr, _ = spearmanr(arr_t[mask], arr_p[mask])
    return 0.0 if np.isnan(corr) else float(corr)


# Nombre de features à conserver après sélection (règle empirique ML4T : max 20-30)
TOP_N_FEATURES = 25


def prepare_data(df):
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_return', 'target']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['target']
    return X, y


def select_top_features(rf_model, feature_names, top_n=TOP_N_FEATURES):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    return top_features, importances


def remove_correlated_features(X_df, feature_names, importances, threshold=0.85):
    """
    Supprime les features trop corrélées en conservant la plus importante (Gini RF).
    Algorithme : trie par importance décroissante. Pour chaque feature, si elle est
    corrélée > threshold avec une feature déjà conservée, elle est éliminée.
    Résultat : un ensemble de features non-redondantes, classées par importance.
    """
    corr = pd.DataFrame(X_df, columns=feature_names).corr().abs()

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    kept, removed = [], []
    for feat, _ in ranked:
        dominated = any(corr.loc[feat, k] > threshold for k in kept)
        if dominated:
            dominant = next(k for k in kept if corr.loc[feat, k] > threshold)
            removed.append((feat, dominant, round(float(corr.loc[feat, dominant]), 2)))
        else:
            kept.append(feat)

    return kept, removed




def train_and_evaluate(symbol):
    df = load_data(f"{symbol}_ML_Dataset")
    if df is None:
        print(f"⚠️ Dataset introuvable pour {symbol}.")
        return

    X_raw, y = prepare_data(df)

    is_valid = run_pre_ml_validation(df, target_col='target_return')
    if not is_valid:
        print("⚠️ Attention: Risque d'instabilité en production.")

    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    print(f"[{symbol}] Train: {len(X_train_scaled)} | Test: {df.index[split_idx].date()} → {df.index[-1].date()}")

    # Sélection de features via RF régresseur
    rf_selector = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_scaled, y_train)
    all_features = list(X_raw.columns)
    importances  = rf_selector.feature_importances_

    kept_features, removed_pairs = remove_correlated_features(
        X_train_raw.values, all_features, importances
    )
    kept_imp   = {f: imp for f, imp in zip(all_features, importances) if f in kept_features}
    top_features = sorted(kept_imp, key=kept_imp.get, reverse=True)[:TOP_N_FEATURES]

    importance_df = pd.DataFrame({
        'feature':    all_features,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    save_data(importance_df, f"{symbol}_feature_importance")
    importance_df.to_csv(os.path.join(CSV_DIR, f"{symbol}_feature_importance.csv"), index=False)

    n_removed = len(removed_pairs)
    top3_feat = ", ".join(top_features[:3])
    print(f"[{symbol}] 🔗 {n_removed} colinéarités  |  Top : {top3_feat}...")

    col_list    = list(X_raw.columns)
    top_indices = [col_list.index(f) for f in top_features]
    X_train = X_train_scaled[:, top_indices]
    X_test  = X_test_scaled[:,  top_indices]

    # Baseline : IC du momentum naïf
    mom_test = X_raw.iloc[split_idx:]['mom_1'] if 'mom_1' in X_raw.columns else pd.Series(0, index=y_test.index)
    ic_mom = ic_score(y_test.values, mom_test.values)

    # Random Forest Regressor
    tscv = TimeSeriesSplit(n_splits=3)
    rf_model = RandomForestRegressor(
        n_estimators=150, max_depth=6, min_samples_leaf=20,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    cv_scores_rf = []
    for train_idx, val_idx in tscv.split(X_train):
        rf_model.fit(X_train[train_idx], y_train.iloc[train_idx])
        cv_scores_rf.append(ic_score(y_train.iloc[val_idx], rf_model.predict(X_train[val_idx])))
    rf_model.fit(X_train, y_train)
    ic_rf = ic_score(y_test, rf_model.predict(X_test))

    # Ridge Regression (équivalent à LR avec régularisation L2)
    ridge_model = Ridge(alpha=10.0)
    cv_scores_ridge = []
    for train_idx, val_idx in tscv.split(X_train):
        ridge_model.fit(X_train[train_idx], y_train.iloc[train_idx])
        cv_scores_ridge.append(ic_score(y_train.iloc[val_idx], ridge_model.predict(X_train[val_idx])))
    ridge_model.fit(X_train, y_train)
    ic_ridge = ic_score(y_test, ridge_model.predict(X_test))

    # XGBoost Regressor
    ic_xgb = 0.0
    cv_scores_xgb = []
    xgb_model = None
    if XGB_AVAILABLE:
        xgb_depth = 4
        xgb_model = XGBRegressor(
            n_estimators=150, max_depth=xgb_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective='reg:squarederror', verbosity=0,
            random_state=42, n_jobs=-1
        )
        for train_idx, val_idx in tscv.split(X_train):
            xgb_model.fit(X_train[train_idx], y_train.iloc[train_idx])
            cv_scores_xgb.append(ic_score(y_train.iloc[val_idx], xgb_model.predict(X_train[val_idx])))
        xgb_model.fit(X_train, y_train)
        ic_xgb = ic_score(y_test, xgb_model.predict(X_test))

    # Sélection du meilleur modèle par IC OOS
    candidates = [("RFR", rf_model, ic_rf, cv_scores_rf),
                  ("Ridge", ridge_model, ic_ridge, cv_scores_ridge)]
    if XGB_AVAILABLE and xgb_model is not None:
        candidates.append(("XGB", xgb_model, ic_xgb, cv_scores_xgb))
    best_name, best_model, best_ic, cv_best_scores = max(candidates, key=lambda x: x[2])
    cv_best = np.mean(cv_best_scores)

    delta = best_ic - ic_mom
    mark  = "✅" if best_ic > 0 and delta > 0 else "⚠️"
    xgb_part = f" XGB:{ic_xgb:+.3f}" if XGB_AVAILABLE else ""
    print(f"[{symbol}] RFR:{ic_rf:+.3f} Ridge:{ic_ridge:+.3f}{xgb_part}"
          f" → {best_name} IC={best_ic:+.3f} ({delta:+.3f} vs naïf) {mark}")

    # Sauvegarde — modèle courant (écrasé à chaque run)
    joblib.dump(best_model,   os.path.join(PARQUET_DIR, f"{symbol}_best_model.pkl"))
    joblib.dump(scaler,       os.path.join(PARQUET_DIR, f"{symbol}_scaler.pkl"))
    joblib.dump(top_features, os.path.join(PARQUET_DIR, f"{symbol}_top_features.pkl"))

    # Sauvegarde ensemble — les 3 modèles + leurs ICs pour combinaison pondérée
    joblib.dump(rf_model,    os.path.join(PARQUET_DIR, f"{symbol}_model_rf.pkl"))
    joblib.dump(ridge_model, os.path.join(PARQUET_DIR, f"{symbol}_model_ridge.pkl"))
    if XGB_AVAILABLE and xgb_model is not None:
        joblib.dump(xgb_model, os.path.join(PARQUET_DIR, f"{symbol}_model_xgb.pkl"))
    ensemble_ics = {
        'RFR':   max(ic_rf,    0.0),
        'Ridge': max(ic_ridge, 0.0),
        'XGB':   max(ic_xgb,   0.0) if XGB_AVAILABLE else 0.0,
    }
    joblib.dump(ensemble_ics, os.path.join(PARQUET_DIR, f"{symbol}_ensemble_ics.pkl"))

    # Versionnage — copie datée pour traçabilité (jamais écrasée)
    from datetime import date
    version      = date.today().strftime('%Y%m%d')
    versions_dir = os.path.join(PARQUET_DIR, 'versions')
    os.makedirs(versions_dir, exist_ok=True)
    joblib.dump(best_model,   os.path.join(versions_dir, f"{symbol}_best_model_{version}.pkl"))
    joblib.dump(top_features, os.path.join(versions_dir, f"{symbol}_top_features_{version}.pkl"))

    return symbol, best_name, best_ic, ic_mom, ic_rf, ic_ridge, ic_xgb


if __name__ == "__main__":
    results = []
    for symbol in STRATEGY["universe"]:
        r = train_and_evaluate(symbol)
        if r:
            results.append(r)

    if results:
        print("\n" + "="*62)
        print(f"{'RÉSUMÉ ENTRAÎNEMENT (RÉGRESSION)':^62}")
        print("="*62)
        print(f"  {'Ticker':<8} {'Modèle':<6}  {'IC OOS':>7}  {'vs Naïf':>8}  {'CV IC':>7}")
        print("  " + "-"*54)
        for sym, best_name, best_ic, ic_mom, ic_rf, ic_ridge, ic_xgb in results:
            delta = best_ic - ic_mom
            mark  = "✅" if best_ic > 0 and delta > 0 else "⚠️"
            print(f"  {sym:<8} {best_name:<6}  {best_ic:>+7.3f}  {delta:>+8.3f}   {mark}")
        print("="*62)
