import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

warnings.filterwarnings('ignore')

from strategy_definition import STRATEGY
from data_pipeline.data_storage import PARQUET_DIR, load_data
from models.training import prepare_data

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _get_shap_values(model, X_test, X_train=None):
    """
    Calcule les SHAP values. Pour la régression, retourne directement le tableau 2D.
    Pour la classification (legacy), extrait la slice de la classe +1.
    """
    inner_model = model.xgb if hasattr(model, 'xgb') else model
    inner_type  = type(inner_model).__name__

    if "RandomForest" in inner_type or "XGB" in inner_type:
        explainer = shap.TreeExplainer(inner_model)
        sv = explainer.shap_values(X_test)
    else:
        if X_train is None:
            raise ValueError("LinearExplainer nécessite X_train comme background.")
        explainer = shap.LinearExplainer(inner_model, X_train)
        sv = explainer.shap_values(X_test)

    # Régression : sv est déjà 2D (n_samples, n_features)
    if not hasattr(model, 'classes_'):
        return np.array(sv)

    # Classification legacy : extraire la slice classe +1
    classes = list(model.classes_)
    idx_pos = classes.index(1) if 1 in classes else -1
    if isinstance(sv, list):
        return np.array(sv[idx_pos])
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, idx_pos]
    return np.array(sv)


def run_shap_analysis():
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP non installé — exécutez : pip install shap")
        return

    summary_rows = []

    for symbol in STRATEGY["universe"]:
        model_path   = os.path.join(PARQUET_DIR, f"{symbol}_best_model.pkl")
        scaler_path  = os.path.join(PARQUET_DIR, f"{symbol}_scaler.pkl")
        feat_path    = os.path.join(PARQUET_DIR, f"{symbol}_top_features.pkl")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, feat_path]):
            print(f"⚠️ [{symbol}] Modèle introuvable — lancez models/training.py d'abord.")
            continue

        df = load_data(f"{symbol}_ML_Dataset")
        if df is None:
            continue

        model        = joblib.load(model_path)
        scaler       = joblib.load(scaler_path)
        top_features = joblib.load(feat_path)
        model_type   = type(model).__name__

        X_raw, _ = prepare_data(df)
        split_idx = int(len(X_raw) * 0.8)

        col_list    = list(X_raw.columns)
        top_indices = [col_list.index(f) for f in top_features]

        X_train_scaled = scaler.transform(X_raw.iloc[:split_idx])
        X_test_scaled  = scaler.transform(X_raw.iloc[split_idx:])
        X_train = X_train_scaled[:, top_indices]
        X_test  = X_test_scaled[:,  top_indices]

        X_train_df = pd.DataFrame(X_train, columns=top_features)
        X_test_df  = pd.DataFrame(X_test,  columns=top_features)

        sv = _get_shap_values(model, X_test_df, X_train=X_train_df)

        mean_abs = np.abs(sv).mean(axis=0)
        mean_dir = sv.mean(axis=0)

        shap_df = pd.DataFrame({
            'feature':    top_features,
            'shap_abs':   mean_abs,
            'shap_mean':  mean_dir,
        }).sort_values('shap_abs', ascending=False).reset_index(drop=True)

        shap_df['direction'] = shap_df['shap_mean'].apply(
            lambda x: "↑ haussier" if x > 0 else "↓ baissier"
        )

        # Sauvegarde
        shap_df.to_parquet(
            os.path.join(PARQUET_DIR, f"{symbol}_shap.parquet"), compression='snappy'
        )

        # Concordance avec Gini
        gini_df  = load_data(f"{symbol}_feature_importance")
        gini_top = gini_df.head(5)['feature'].tolist() if gini_df is not None else []
        shap_top = shap_df.head(5)['feature'].tolist()
        overlap  = len(set(gini_top) & set(shap_top))

        _inner       = model.xgb if hasattr(model, 'xgb') else model
        display_type = type(_inner).__name__

        top_dir = shap_df.iloc[0]['direction']
        summary_rows.append((symbol, display_type, shap_top[0], shap_df['shap_abs'].iloc[0], overlap, top_dir))

    # Résumé final
    print("\n" + "="*72)
    print(f"{'RÉSUMÉ SHAP — FEATURE DOMINANTE PAR TICKER':^72}")
    print("="*72)
    print(f"  {'Ticker':<6}  {'Modèle':<22}  {'Feature dominante':<28}  {'SHAP':>6}  Gini  Direction")
    print("  " + "-"*66)
    for sym, mtype, top_feat, top_val, ovlp, top_dir in summary_rows:
        warn = " ⚠️" if ovlp <= 2 else "   "
        print(f"  {sym:<6}  {mtype:<22}  {top_feat:<28}  {top_val:.4f}  {ovlp}/5{warn}  {top_dir}")
    print("="*72)


if __name__ == "__main__":
    run_shap_analysis()
