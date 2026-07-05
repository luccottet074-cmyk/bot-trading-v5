import os
import sys
import warnings
import pandas as pd

# --- Chemins globaux ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# Ajouter la racine au sys.path pour importer strategy_definition
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_loader import load_data
from data_pipeline.data_storage import save_data, file_exists, load_data as load_parquet
from data_pipeline.data_quality import run_quality_checks
from data_pipeline.data_cleaning import clean_data
from data_pipeline.data_alignment import align_time_series

warnings.filterwarnings('ignore')

# Symboles de contexte marché (téléchargés via Yahoo, jamais tradés)
CONTEXT_SYMBOLS = {"SPY": "SPY", "VIX": "^VIX"}


def download_context_data():
    freq   = STRATEGY["frequency"]
    period = STRATEGY["lookback_period"]
    days   = int(''.join(filter(str.isdigit, period)))

    # SPY via IBKR avec cache raw/
    try:
        import time
        from data_pipeline.data_storage import RAW_DIR
        from execution.broker import fetch_bars, IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync non disponible")

        spy_cache = os.path.join(RAW_DIR, "SPY_raw.parquet")
        if os.path.exists(spy_cache):
            age_h = (time.time() - os.path.getmtime(spy_cache)) / 3600
            if age_h < 20:
                spy_df = pd.read_parquet(spy_cache)
                print(f"   ✅ [SPY] Cache IBKR ({age_h:.0f}h) — {len(spy_df)} barres")
            else:
                days_since = max(2, int(age_h / 24) + 2)
                df_existing = pd.read_parquet(spy_cache)
                df_new = fetch_bars('SPY', days=days_since)
                if not df_new.empty:
                    spy_df = pd.concat([df_existing, df_new])
                    spy_df = spy_df[~spy_df.index.duplicated(keep='last')].sort_index()
                    from datetime import timezone
                    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days)
                    spy_df = spy_df[spy_df.index >= cutoff]
                    spy_df.to_parquet(spy_cache, compression='snappy')
                    print(f"   ✅ [SPY] IBKR +{len(df_new)} barres → {len(spy_df)} total")
                else:
                    spy_df = df_existing
        else:
            spy_df = fetch_bars('SPY', days=days)
            spy_df.to_parquet(spy_cache, compression='snappy')
            print(f"   ✅ [SPY] {len(spy_df)} barres de contexte (IBKR)")

        if not spy_df.empty:
            save_data(spy_df, f"SPY_context_{freq}")
        else:
            raise ValueError("SPY vide")
    except Exception as e:
        print(f"   ⚠️ [SPY] Alpaca indisponible ({e}) — fallback Yahoo")
        spy_df_fb = load_data('SPY', period='730d', interval=freq, source="yahoo")
        if not spy_df_fb.empty:
            save_data(spy_df_fb, f"SPY_context_{freq}")
            print(f"   ✅ [SPY] {len(spy_df_fb)} barres de contexte (Yahoo fallback)")

    # VIX via Yahoo Finance — source stable et reproductible pour run_all.py
    # (IBKR VIX requiert abonnement CBOE et produit des résultats variables selon dispo)
    try:
        import yfinance as yf
        vix_daily = yf.Ticker('^VIX').history(period='max', interval='1d', auto_adjust=True)
        if isinstance(vix_daily.columns, __import__('pandas').MultiIndex):
            vix_daily.columns = vix_daily.columns.droplevel(1)
        vix_daily = vix_daily[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        spy_saved = load_parquet(f"SPY_context_{freq}")
        if spy_saved is not None and not vix_daily.empty:
            vix_resampled = vix_daily.reindex(spy_saved.index, method='ffill').dropna(subset=['Close'])
            if not vix_resampled.empty:
                save_data(vix_resampled, f"VIX_context_{freq}")
                print(f"   ✅ [VIX] {len(vix_resampled)} barres (Yahoo)")
    except Exception as e:
        print(f"   ⚠️ [VIX] Yahoo indisponible ({e})")


def run_data_pipeline():
    for symbol in STRATEGY["universe"]:
        filename = f"{symbol}_{STRATEGY['frequency']}"

        data_source = STRATEGY.get("data_source", "yahoo")
        df_raw = load_data(symbol, period=STRATEGY["lookback_period"], interval=STRATEGY["frequency"], source=data_source)

        if df_raw.empty:
            print(f"❌ [{symbol}] Aucune donnée trouvée sur aucune source.")
            continue

        quality_report = run_quality_checks(df_raw, symbol=symbol)
        df_clean = clean_data(df_raw)
        df_aligned = align_time_series(df_clean)

        if df_aligned.empty:
            print(f"❌ [{symbol}] DataFrame vide après nettoyage.")
            continue

        final_path = save_data(df_aligned, filename)
        print(f"✅ [{symbol}] {len(df_aligned)} barres ({STRATEGY['frequency']}, {STRATEGY['lookback_period']})")

    print("   Téléchargement des données de contexte (SPY, VIX)...")
    download_context_data()

    # Barres 15min IBKR pour les features intraday (2 ans, cache incremental)
    _download_15min_data()

    # IV 1h IBKR pour les features de volatilite implicite (2 ans, cache incremental)
    _download_iv_data()

    # HV daily IBKR (Parkinson) — cache incremental, utilisé si vrp_ibkr réactivé
    _download_hv_data()


def _download_15min_data(years: int = 2):
    """Télécharge et met en cache les barres 15min IBKR pour les features intraday."""
    try:
        import time
        from execution.broker import fetch_bars_15min, IBKR_AVAILABLE
        from features.intraday import RAW_15MIN_DIR
        if not IBKR_AVAILABLE:
            return

        print("   Téléchargement données 15min IBKR (features intraday)...")
        for symbol in STRATEGY["universe"]:
            cache_path = os.path.join(RAW_15MIN_DIR, f"{symbol}_15min.parquet")
            # Incremental : si cache existe, seulement les 35 derniers jours
            if os.path.exists(cache_path):
                df_existing = pd.read_parquet(cache_path)
                df_new = fetch_bars_15min(symbol, years=1)  # 1 chunk recent = 30 jours
                if not df_new.empty and not df_existing.empty:
                    df = pd.concat([df_existing, df_new])
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                else:
                    df = df_existing if not df_existing.empty else df_new
            else:
                # Premier téléchargement : 2 ans
                df = fetch_bars_15min(symbol, years=years)

            if not df.empty:
                df.to_parquet(cache_path, compression='snappy')
                print(f"     ✅ {symbol} 15min — {len(df)} barres")
    except Exception as e:
        print(f"     ⚠️ 15min indisponible ({e}) — features intraday desactivees")


def _download_iv_data():
    """Telecharge et met en cache l'IV 1h IBKR (2 ans, incremental)."""
    try:
        from features.ibkr_derivatives import fetch_and_cache_iv
        from execution.broker import IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            return
        print("   Téléchargement IV 1h IBKR (Variance Risk Premium)...")
        for symbol in STRATEGY["universe"]:
            df = fetch_and_cache_iv(symbol)
            if not df.empty:
                print(f"     ✅ {symbol} IV — {len(df)} barres")
            else:
                print(f"     ⚠️ {symbol} IV — indisponible")
    except Exception as e:
        print(f"     ⚠️ IV indisponible ({e}) — features VRP desactivees")


def _download_hv_data():
    """Telecharge et cache la HV daily IBKR (Parkinson estimator, 1 an)."""
    try:
        from features.ibkr_derivatives import fetch_and_cache_hv
        from execution.broker import IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            return
        print("   Téléchargement HV daily IBKR (Parkinson)...")
        for symbol in STRATEGY["universe"]:
            df = fetch_and_cache_hv(symbol)
            if not df.empty:
                print(f"     ✅ {symbol} HV — {len(df)} barres")
            else:
                print(f"     ⚠️ {symbol} HV — indisponible")
    except Exception as e:
        print(f"     ⚠️ HV indisponible ({e})")


def _download_opt_volume_data():
    """Telecharge et cache le volume options daily IBKR (1 an)."""
    try:
        from features.ibkr_derivatives import fetch_and_cache_opt_vol
        from execution.broker import IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            return
        print("   Téléchargement Option Volume daily IBKR...")
        for symbol in STRATEGY["universe"]:
            df = fetch_and_cache_opt_vol(symbol)
            if not df.empty:
                print(f"     ✅ {symbol} OPT_VOL — {len(df)} barres")
            else:
                print(f"     ⚠️ {symbol} OPT_VOL — indisponible")
    except Exception as e:
        print(f"     ⚠️ Option Volume indisponible ({e})")


if __name__ == "__main__":
    run_data_pipeline()
