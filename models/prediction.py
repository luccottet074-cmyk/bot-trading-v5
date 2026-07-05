import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

# Ajouter la racine au sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

warnings.filterwarnings('ignore')

from strategy_definition import STRATEGY
from data_pipeline.data_storage import DATA_DIR, PARQUET_DIR, load_data
from models.training import prepare_data, XGBClassifierWrapper

def generate_signals(symbol):
    print(f"Génération des signaux pour {symbol}...")
    
    df = load_data(f"{symbol}_ML_Dataset")
    model_path  = os.path.join(PARQUET_DIR, f"{symbol}_best_model.pkl")
    scaler_path = os.path.join(PARQUET_DIR, f"{symbol}_scaler.pkl")
    
    if df is None:
        print(f"⚠️ Dataset introuvable pour {symbol}.")
        return
        
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ Modèle ou Scaler introuvable pour {symbol}. Lancez models/training.py d'abord.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    top_features = joblib.load(os.path.join(PARQUET_DIR, f"{symbol}_top_features.pkl"))

    X_raw, _ = prepare_data(df)

    split_idx = int(len(X_raw) * 0.8)
    X_test_raw = X_raw.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()

    X_test_scaled = scaler.transform(X_test_raw)
    col_list = list(X_raw.columns)
    top_indices = [col_list.index(f) for f in top_features]
    X_test = X_test_scaled[:, top_indices]

    # Ensemble : moyenne pondérée par IC des 3 modèles (si disponibles)
    ens_ics_path = os.path.join(PARQUET_DIR, f"{symbol}_ensemble_ics.pkl")
    _model_files = {'RFR': f"{symbol}_model_rf.pkl", 'Ridge': f"{symbol}_model_ridge.pkl", 'XGB': f"{symbol}_model_xgb.pkl"}
    ensemble_used = False
    if os.path.exists(ens_ics_path):
        ens_ics = joblib.load(ens_ics_path)
        members = {}
        for name, fname in _model_files.items():
            p = os.path.join(PARQUET_DIR, fname)
            if os.path.exists(p) and ens_ics.get(name, 0.0) > 0:
                members[name] = (joblib.load(p), ens_ics[name])
        if len(members) >= 2:
            total_ic = sum(ic for _, ic in members.values())
            predicted_returns = sum(m.predict(X_test) * ic / total_ic for m, ic in members.values())
            model_type = f"Ensemble({'+'.join(members.keys())})"
            ensemble_used = True

    if not ensemble_used:
        predicted_returns = model.predict(X_test)
        model_type = type(model).__name__

    # Seuils adaptés au type de modèle et au ticker
    # RF compresse les prédictions (max ~0.002) → seuil plus bas
    # Ridge/XGB ont une plage plus large (jusqu'à 0.03) → seuil standard
    _TICKER_THRESHOLDS = {
        # (min_ret, high_ret)
        'COST': (0.0015, 0.003),   # E[r|pred>0.001] < coût → lever le seuil
    }
    if symbol in _TICKER_THRESHOLDS:
        min_ret, high_ret = _TICKER_THRESHOLDS[symbol]
    elif 'Forest' in model_type:
        min_ret, high_ret = 0.0003, 0.0006   # RF : compression → seuil bas
    else:
        min_ret  = STRATEGY.get('target_threshold', 0.001)
        high_ret = min_ret * 2

    print(f"   Modèle : {model_type}  →  seuils {min_ret:.4f} / {high_ret:.4f}")

    signals_df = pd.DataFrame(index=df_test.index)
    signals_df['Close']            = df_test['Close']
    signals_df['Predicted_Return'] = predicted_returns

    # Signal : trade si retour prédit > coût de transaction
    signals_df['Signal'] = (signals_df['Predicted_Return'] > min_ret).astype(int)

    # Position sizing continu : rampe linéaire 0.5 → 1.0 entre min_ret et high_ret
    raw_size = (signals_df['Predicted_Return'] - min_ret) / (high_ret - min_ret) * 0.5 + 0.5
    signals_df['Signal_Size'] = np.where(
        signals_df['Predicted_Return'] < min_ret,
        0.0,
        raw_size.clip(0.5, 1.0)
    )

    parquet_out = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
    signals_df.to_parquet(parquet_out, compression='snappy')
    print(f"✅ [{symbol}] {len(signals_df)} signaux générés")

def generate_panel_signals(symbols):
    """
    Génère des signaux cross-sectionnels en utilisant le modèle panel unique.
    Chaque ticker est évalué avec le même modèle universel entraîné sur le panel.
    On utilise la coupure temporelle du panel pour garantir un test OOS cohérent.
    """
    paths = {
        'model':      os.path.join(PARQUET_DIR, "panel_best_model.pkl"),
        'scaler':     os.path.join(PARQUET_DIR, "panel_scaler.pkl"),
        'features':   os.path.join(PARQUET_DIR, "panel_top_features.pkl"),
        'all_feat':   os.path.join(PARQUET_DIR, "panel_all_features.pkl"),
        'ticker_map': os.path.join(PARQUET_DIR, "panel_ticker_map.pkl"),
        'split_date': os.path.join(PARQUET_DIR, "panel_split_date.pkl"),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Fichier panel introuvable : {key}. Lancez models/training.py d'abord.")
            return

    model        = joblib.load(paths['model'])
    scaler       = joblib.load(paths['scaler'])
    top_features = joblib.load(paths['features'])
    all_features = joblib.load(paths['all_feat'])
    ticker_map   = joblib.load(paths['ticker_map'])
    split_date   = joblib.load(paths['split_date'])

    classes = list(model.classes_)
    print(f"   Classes du modèle panel : {classes}")
    if 1 not in classes:
        print("⚠️ Classe Hausse (1) absente du modèle panel.")
        return
    idx_hausse = classes.index(1)
    threshold = 0.53 if 'Forest' in type(model).__name__ else 0.58
    print(f"   Modèle panel : {type(model).__name__}  →  seuil {threshold}")

    # Indices des top features dans la liste complète (même ordre qu'à l'entraînement)
    top_indices = [all_features.index(f) for f in top_features if f in all_features]

    for symbol in symbols:
        if symbol not in ticker_map:
            print(f"⚠️ [{symbol}] Absent du ticker_map panel — ignoré.")
            continue

        df = load_data(f"{symbol}_ML_Dataset")
        if df is None:
            print(f"⚠️ [{symbol}] Dataset introuvable.")
            continue

        X_raw, _ = prepare_data(df)
        X_raw    = X_raw.copy()
        X_raw['ticker_id'] = ticker_map[symbol]

        # Aligner les colonnes sur l'ordre exact utilisé pendant l'entraînement du panel
        for f in all_features:
            if f not in X_raw.columns:
                X_raw[f] = 0.0
        X_raw = X_raw[all_features]

        # Période de test = données après la coupure temporelle du panel
        df_test    = df[df.index > split_date].copy()
        X_test_raw = X_raw[X_raw.index > split_date]

        if len(X_test_raw) == 0:
            print(f"⚠️ [{symbol}] Aucune donnée après split_date ({split_date.date()}).")
            continue

        X_test_scaled = scaler.transform(X_test_raw)
        X_test        = X_test_scaled[:, top_indices]

        probabilities = model.predict_proba(X_test)

        signals_df = pd.DataFrame(index=df_test.index)
        signals_df['Close']   = df_test['Close']
        signals_df['Prob_Up'] = probabilities[:, idx_hausse]
        signals_df['Signal']  = (signals_df['Prob_Up'] > threshold).astype(int)

        parquet_out = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
        signals_df.to_parquet(parquet_out, compression='snappy')
        print(f"✅ [{symbol}] {len(signals_df)} signaux (panel model)")


def apply_cross_sectional_filter(symbols, top_n=3):
    """
    Filtre cross-sectionnel : à chaque barre, parmi tous les signaux long actifs,
    ne conserve que les top_n tickers avec la Prob_Up la plus haute.
    Les signaux long hors top_n sont ramenés à 0 (flat).
    """
    signal_dfs = {}
    for symbol in symbols:
        path = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
        if os.path.exists(path):
            signal_dfs[symbol] = pd.read_parquet(path)

    if len(signal_dfs) < 2:
        return

    # Matrice de signal pour le ranking (Predicted_Return si régression, sinon Signal)
    rank_col = 'Predicted_Return' if 'Predicted_Return' in next(iter(signal_dfs.values())).columns else 'Signal'
    prob_matrix = pd.DataFrame({sym: df[rank_col] for sym, df in signal_dfs.items()})

    # Rang de chaque ticker à chaque barre (1 = Prob_Up la plus haute)
    rank_matrix = prob_matrix.rank(axis=1, ascending=False, method='first')

    total_filtered = 0
    for symbol, df in signal_dfs.items():
        common_idx  = df.index.intersection(rank_matrix.index)
        was_long    = df.loc[common_idx, 'Signal'] == 1
        rank_too_low = rank_matrix.loc[common_idx, symbol] > top_n
        to_filter   = was_long & rank_too_low

        total_filtered += int(to_filter.sum())
        df.loc[common_idx[to_filter], 'Signal'] = 0
        if 'Signal_Size' in df.columns:
            df.loc[common_idx[to_filter], 'Signal_Size'] = 0.0

        path = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
        df.to_parquet(path, compression='snappy')

    print(f"✅ Filtre cross-sectionnel top-{top_n} : {total_filtered} signaux filtrés "
          f"sur {len(symbols)} tickers")


if __name__ == "__main__":
    for symbol in STRATEGY["universe"]:
        generate_signals(symbol)
    apply_cross_sectional_filter(STRATEGY["universe"], top_n=3)
