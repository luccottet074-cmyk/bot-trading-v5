"""
Broker interface — Interactive Brokers via ib_insync
====================================================
Requiert IB Gateway (port 4001/4002) ou TWS (port 7496/7497) actif.
Configurer IBKR_PORT dans .env selon le mode :
  7497 → TWS paper trading
  7496 → TWS live
  4002 → IB Gateway paper
  4001 → IB Gateway live (recommandé en production)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

import math as _math
import pandas as pd
import pytz
import holidays as _holidays
from datetime import datetime

try:
    import ib_insync
    ib_insync.util.patchAsyncio()
    from ib_insync import IB, Stock, MarketOrder
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

ALPACA_AVAILABLE = IBKR_AVAILABLE  # alias conservé pour compatibilité ascendante

IBKR_HOST      = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT      = int(os.getenv('IBKR_PORT', '7497'))
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))

_ib_instance = None


def _ib() -> 'IB':
    """Retourne la connexion IB, en (re)connectant si nécessaire."""
    global _ib_instance
    if _ib_instance is None or not _ib_instance.isConnected():
        _ib_instance = IB()
        _ib_instance.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
        _ib_instance.reqMarketDataType(3)  # données différées 15 min si pas d'abonnement temps réel
    return _ib_instance


def _safe_price(ticker, avg_price: float = 0.0) -> float:
    """Retourne le premier prix valide (non-NaN, > 0) parmi les champs du ticker.
    Fallback sur avg_price si aucun champ n'est disponible (cas compte paper sans abonnement).
    """
    for p in [ticker.marketPrice(), ticker.last, ticker.bid, ticker.close, avg_price]:
        try:
            v = float(p)
            if not _math.isnan(v) and v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return 0.0


def get_account() -> dict:
    # Prend la devise de base du compte (EUR pour comptes européens, USD pour US)
    all_vals = [(v.tag, v.currency, v.value) for v in _ib().accountValues()
                if v.tag in ('NetLiquidation', 'TotalCashValue', 'GrossPositionValue')]
    # Priorité USD, sinon première devise avec NetLiquidation non nul
    base_ccy = 'USD'
    for tag, ccy, val in all_vals:
        if tag == 'NetLiquidation' and float(val) > 0:
            base_ccy = ccy
            break
    vals = {v.tag: v.value for v in _ib().accountValues() if v.currency == base_ccy}
    return {
        'portfolio_value': float(vals.get('NetLiquidation', 0)),
        'cash':            float(vals.get('TotalCashValue', 0)),
        'equity':          float(vals.get('GrossPositionValue', 0)),
    }


def get_positions() -> dict:
    result = {}
    for pos in _ib().positions():
        if pos.contract.secType != 'STK':
            continue
        symbol    = pos.contract.symbol
        qty       = float(pos.position)
        avg_price = float(pos.avgCost)

        contract = Stock(symbol, 'SMART', 'USD')
        _ib().qualifyContracts(contract)
        [ticker]  = _ib().reqTickers(contract)
        cur_price = _safe_price(ticker, avg_price)

        mv  = qty * cur_price
        upl = (cur_price - avg_price) * qty
        result[symbol] = {
            'qty':             qty,
            'market_value':    mv,
            'avg_price':       avg_price,
            'current_price':   cur_price,
            'unrealized_pl':   upl,
            'unrealized_plpc': (cur_price - avg_price) / avg_price * 100 if avg_price else 0,
        }
    return result


def is_market_open() -> bool:
    """Vérifie si le marché US est ouvert (9h30-16h00 ET, lun-ven, hors jours fériés NYSE)."""
    ET  = pytz.timezone('America/New_York')
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    if now.date() in _holidays.NYSE():
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def _bar_size(freq: str) -> str:
    return {
        '1m': '1 min', '5m': '5 mins', '15m': '15 mins',
        '1h': '1 hour', '1d': '1 day',
    }.get(freq, '1 hour')


def fetch_bars(symbol: str, days: int = 185, freq: str = None) -> pd.DataFrame:
    if freq is None:
        from strategy_definition import STRATEGY
        freq = STRATEGY.get('frequency', '1h')

    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)

    if days > 365:
        years = max(1, round(days / 365))
        duration = f'{years} Y'
    else:
        duration = f'{days} D'
    bars = _ib().reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=_bar_size(freq),
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return pd.DataFrame()

    rows = [{'date': b.date, 'Open': b.open, 'High': b.high,
             'Low': b.low, 'Close': b.close, 'Volume': b.volume}
            for b in bars]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])

    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')

    return df.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def _bars_to_df(bars) -> pd.DataFrame:
    """Convertit une liste de barres IBKR en DataFrame OHLCV indexe UTC."""
    rows = [{'date': b.date, 'Open': b.open, 'High': b.high,
             'Low': b.low, 'Close': b.close, 'Volume': b.volume}
            for b in bars]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    return df.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def fetch_bars_15min(symbol: str, years: int = 2) -> pd.DataFrame:
    """Fetch historique 15min via chunks de 180 jours (limitation IBKR).

    Fetche `years` annees de barres 15min en plusieurs requetes de 180 jours.
    Chaque chunk est mis en cache dans data/raw_15min/ pour les runs suivants.
    """
    import time
    from datetime import timezone, timedelta

    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)

    n_chunks = max(1, round(max(years, 1) * 365 / 180))
    all_dfs = []
    end_dt = datetime.now(timezone.utc)

    for i in range(n_chunks):
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"
        try:
            bars = _ib().reqHistoricalData(
                contract, endDateTime=end_str, durationStr='180 D',
                barSizeSetting='15 mins', whatToShow='TRADES',
                useRTH=True, formatDate=1,
            )
            if bars:
                all_dfs.append(_bars_to_df(bars))
        except Exception as e:
            print(f"     ⚠️ 15min chunk {i+1}/{n_chunks} erreur: {e}")
        end_dt = end_dt - timedelta(days=179)
        if i < n_chunks - 1:
            time.sleep(1)  # pacing IBKR

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def place_order(symbol: str, side: str, notional: float):
    """Place un ordre marché en valeur notionnelle ($). Supporte les fractions de titres."""
    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)

    [ticker] = _ib().reqTickers(contract)
    price = _safe_price(ticker)
    if price <= 0:
        raise ValueError(f"Prix indisponible pour {symbol}")

    qty    = int(notional / price)  # IBKR API n'accepte pas les fractions d'actions
    action = 'BUY' if side == 'buy' else 'SELL'
    trade  = _ib().placeOrder(contract, MarketOrder(action, qty, tif='DAY'))
    _ib().sleep(2)
    return trade


def close_position(symbol: str):
    """Ferme entièrement la position sur un symbole."""
    for pos in _ib().positions():
        if pos.contract.symbol == symbol:
            contract = Stock(symbol, 'SMART', 'USD')
            _ib().qualifyContracts(contract)
            trade = _ib().placeOrder(contract, MarketOrder('SELL', abs(float(pos.position)), tif='DAY'))
            _ib().sleep(2)
            return trade
    print(f'  ⚠️ Position {symbol} introuvable pour clôture')


def fetch_index_bars(symbol: str, exchange: str = 'CBOE', currency: str = 'USD',
                     days: int = 365, freq: str = '1h') -> pd.DataFrame:
    """Fetch historical bars pour un indice (ex. VIX) via IBKR."""
    from ib_insync import Index as IbIndex
    contract = IbIndex(symbol, exchange, currency)
    _ib().qualifyContracts(contract)

    duration = f'{max(1, round(days / 365))} Y' if days > 365 else f'{days} D'

    bars = _ib().reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=_bar_size(freq),
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return pd.DataFrame()

    rows = [{'date': b.date, 'Open': b.open, 'High': b.high,
             'Low': b.low, 'Close': b.close, 'Volume': b.volume}
            for b in bars]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    return df.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()



def fetch_premarket_data(symbol: str):
    """Retourne (prix_actuel, close_precedent, volume_premarket).

    volume_premarket = somme des volumes barres 5min hors-RTH depuis minuit jusqu'a 9h30 ET.
    """
    import pytz as _pytz
    from datetime import datetime as _dt

    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)
    [ticker] = _ib().reqTickers(contract)
    current    = float(ticker.last or ticker.bid or 0)
    prev_close = float(ticker.close or 0)

    # Volume pre-market : barres 5min hors-RTH du jour courant
    ET = _pytz.timezone('America/New_York')
    market_open = _dt.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)

    bars = _ib().reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1,
    )

    pm_vol = 0
    for bar in bars:
        bar_time = pd.to_datetime(bar.date)
        if bar_time.tzinfo is None:
            bar_time = ET.localize(bar_time)
        else:
            bar_time = bar_time.tz_convert(ET)
        if bar_time < market_open:
            pm_vol += int(bar.volume)

    return current, prev_close, pm_vol

def fetch_hv_ibkr(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch Historical Volatility IBKR (estimateur Parkinson daily, plus efficace que Close-to-Close)."""
    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)
    duration = f'{max(1, round(days/365))} Y' if days > 365 else f'{days} D'
    bars = _ib().reqHistoricalData(
        contract, endDateTime='', durationStr=duration,
        barSizeSetting='1 day', whatToShow='HISTORICAL_VOLATILITY',
        useRTH=True, formatDate=1,
    )
    if not bars:
        return pd.DataFrame()
    rows = [{'date': b.date, 'hv': b.close} for b in bars]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    return df.set_index('date')[['hv']].dropna()


def fetch_option_volume(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch volume journalier des options IBKR (toutes options confondues, daily)."""
    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)
    duration = f'{max(1, round(days/365))} Y' if days > 365 else f'{days} D'
    bars = _ib().reqHistoricalData(
        contract, endDateTime='', durationStr=duration,
        barSizeSetting='1 day', whatToShow='OPTION_VOLUME',
        useRTH=True, formatDate=1,
    )
    if not bars:
        return pd.DataFrame()
    rows = [{'date': b.date, 'opt_vol': b.volume} for b in bars]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    return df.set_index('date')[['opt_vol']].dropna()


def fetch_premarket_price(symbol: str):
    """Retourne (prix_actuel, close_precedent) pour calculer le gap pre-market.

    En pre-market : ticker.last = dernier trade pre-market, ticker.close = close RTH précédent.
    Retourne (0.0, 0.0) si les données sont indisponibles.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    _ib().qualifyContracts(contract)
    [ticker] = _ib().reqTickers(contract)
    current    = float(ticker.last or ticker.bid or 0)
    prev_close = float(ticker.close or 0)
    return current, prev_close
