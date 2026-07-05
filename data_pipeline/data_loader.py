import pandas as pd
import yfinance as yf
import os

# --- Imports des bibliothèques professionnelles ---
# pandas_datareader est désactivé car il contient un bug (TypeError: deprecate_kwarg) avec Pandas 2.2+
web = None

try:
    import nasdaqdatalink
except ImportError:
    nasdaqdatalink = None

try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    TimeSeries = None

try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None

# --- Chargement sécurisé des clés API depuis le fichier .env ---
# Le fichier .env se trouve à la racine du projet (1 niveau au-dessus)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            # On ignore les lignes vides et les commentaires
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    pass

def load_from_yahoo(symbol, period="60d", interval="15m"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"   ❌ Erreur Yahoo: {e}")
        return pd.DataFrame()


def load_from_stooq(symbol):
    print(f"   🔄 Téléchargement depuis Stooq (Fallback) pour {symbol}...")
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url, index_col="Date", parse_dates=True)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        return df.sort_index()
    except Exception as e:
        print(f"   ❌ Erreur Stooq: {e}")
        return pd.DataFrame()


def load_from_quandl(symbol):
    if nasdaqdatalink is None:
        print("   ❌ Erreur: 'nasdaq-data-link' n'est pas installé.")
        return pd.DataFrame()
        
    api_key = os.environ.get("QUANDL_API_KEY")
    if not api_key:
        print("   🔒 Accès Refusé : La clé d'API QUANDL_API_KEY est introuvable sur votre ordinateur.")
        return pd.DataFrame()
        
    try:
        nasdaqdatalink.ApiConfig.api_key = api_key
        # Exemple de format: WIKI/AAPL (les bases dépendent de votre abonnement)
        df = nasdaqdatalink.get(f"WIKI/{symbol}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    except Exception as e:
        print(f"   ❌ Erreur Quandl: {e}")
        return pd.DataFrame()


def load_from_alphavantage(symbol, interval="15min"):
    if TimeSeries is None:
        print("   ❌ Erreur: 'alpha_vantage' n'est pas installé.")
        return pd.DataFrame()
        
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("   🔒 Accès Refusé : La clé d'API ALPHAVANTAGE_API_KEY est introuvable.")
        return pd.DataFrame()
        
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        df, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
        # AlphaVantage nomme ses colonnes '1. open', '2. high', etc.
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    except Exception as e:
        print(f"   ❌ Erreur AlphaVantage: {e}")
        return pd.DataFrame()


def load_from_polygon(symbol, period="60d", interval="15m"):
    if RESTClient is None:
        print("   ❌ Erreur: 'polygon-api-client' n'est pas installé.")
        return pd.DataFrame()

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("   🔒 Accès Refusé : La clé d'API POLYGON_API_KEY est introuvable.")
        return pd.DataFrame()

    interval_map = {
        "1m":  (1,  "minute"),
        "5m":  (5,  "minute"),
        "15m": (15, "minute"),
        "1h":  (1,  "hour"),
        "1d":  (1,  "day"),
    }
    multiplier, timespan = interval_map.get(interval, (15, "minute"))

    try:
        client = RESTClient(api_key=api_key, timeout=30)
        import datetime
        end_date = datetime.date.today()
        days = int(''.join(filter(str.isdigit, period)))
        start_date = end_date - datetime.timedelta(days=days)

        aggs = []
        for a in client.list_aggs(symbol, multiplier, timespan, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')):
            aggs.append(a)

        if not aggs:
            return pd.DataFrame()

        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    except Exception as e:
        print(f"   ❌ Erreur Polygon: {e}")
        return pd.DataFrame()


def _load_bars_cached(symbol, period="180d", interval="1h", label="IBKR"):
    try:
        import sys, os, time
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from data_pipeline.data_storage import RAW_DIR
        from execution.broker import fetch_bars, IBKR_AVAILABLE
        if not IBKR_AVAILABLE:
            print("   ❌ ib_insync non disponible.")
            return pd.DataFrame()

        cache_path = os.path.join(RAW_DIR, f"{symbol}_raw.parquet")
        days_full  = int(''.join(filter(str.isdigit, period)))

        if os.path.exists(cache_path):
            df_existing = pd.read_parquet(cache_path)
            age_h = (time.time() - os.path.getmtime(cache_path)) / 3600

            if age_h < 20:
                print(f"   ✅ Cache {label} ({age_h:.0f}h) — {len(df_existing)} barres")
                return df_existing

            # Téléchargement incrémental : seulement les nouvelles barres
            days_since = max(2, int(age_h / 24) + 2)
            df_new = fetch_bars(symbol, days=days_since)
            if not df_new.empty:
                df = pd.concat([df_existing, df_new])
                df = df[~df.index.duplicated(keep='last')].sort_index()
                # Garde uniquement la fenêtre utile (days_full)
                from datetime import timezone
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days_full)
                df = df[df.index >= cutoff]
                df.to_parquet(cache_path, compression='snappy')
                print(f"   ✅ {label} +{len(df_new)} barres ajoutées → {len(df)} total")
                return df
            return df_existing

        # Premier téléchargement complet
        df = fetch_bars(symbol, days=days_full)
        if not df.empty:
            df.to_parquet(cache_path, compression='snappy')
        return df
    except Exception as e:
        print(f"   ❌ Erreur IBKR: {e}")
        return pd.DataFrame()


def load_from_ibkr(symbol, period="1825d", interval="1h"):
    """Charge les données depuis IBKR via broker.fetch_bars (cache incrémental)."""
    return _load_bars_cached(symbol, period, interval, label="IBKR")


def load_data(symbol, period="60d", interval="15m", source="yahoo"):
    """
    Module 4: Chargeur Universel (Hub de connexion aux API).
    """
    print(f"📡 Requête de données pour {symbol} (Source demandée: {source})...")

    df = pd.DataFrame()

    if source in ("alpaca", "ibkr"):
        df = load_from_ibkr(symbol, period, interval)
    elif source == "yahoo":
        df = load_from_yahoo(symbol, period, interval)
    elif source == "stooq":
        df = load_from_stooq(symbol)
    elif source == "quandl":
        df = load_from_quandl(symbol)
    elif source == "alphavantage":
        df = load_from_alphavantage(symbol, "15min" if interval == "15m" else interval)
    elif source == "polygon":
        df = load_from_polygon(symbol, period, interval)

    if df.empty and source != "yahoo":
        print("   ⚠️ Source demandée indisponible ou Clé API manquante. Bascule sur Yahoo (Plan B)...")
        df = load_from_yahoo(symbol, period, interval)
        
    if df.empty and source == "yahoo":
        print("   ⚠️ Yahoo est en panne. Bascule sur Stooq (Plan B)...")
        df = load_from_stooq(symbol)
        
    return df
