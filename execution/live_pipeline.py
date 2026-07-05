"""
Live Trading Pipeline â€” IBKR Paper Trading
=============================================
SÃ©quence Ã  chaque exÃ©cution :
  1. VÃ©rifie que le marchÃ© est ouvert
  2. TÃ©lÃ©charge les barres rÃ©centes depuis IBKR
  3. Relance alpha_factors â†’ ml_features â†’ prediction (modÃ¨les prÃ©-entraÃ®nÃ©s)
  4. Lit les derniers signaux + applique le filtre rÃ©gime
  5. Place / ferme les ordres via l'API IBKR (ib_insync)

Lancement : python -m execution.live_pipeline
Planification : toutes les heures pendant les heures de marchÃ© (9h30-16h00 ET)
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

import csv
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from strategy_definition import STRATEGY
from data_pipeline.data_storage import PARQUET_DIR, save_data, load_data
from execution.broker import (
    get_account, get_positions, is_market_open,
    fetch_bars, place_order, close_position, IBKR_AVAILABLE,
    fetch_premarket_price, fetch_premarket_data, fetch_index_bars,
)
from execution.macro_calendar import check_macro_risk

SYMBOLS = STRATEGY['universe']
ET      = pytz.timezone('America/New_York')

# Stop Loss / Take Profit par ticker — identique à backtesting/engine.py
_TICKER_RISK = {
    'AAPL':  (0.02, 0.03),
    'MSFT':  (0.02, 0.04),
    'GOOGL': (0.02, 0.05),
    'COST':  (0.02, 0.03),
}
_DEFAULT_RISK = (0.02, 0.03)


def _load_kelly_weights():
    """Charge les poids Kelly ou retourne equal-weight si fichier absent."""
    from data_pipeline.data_storage import PARQUET_DIR as _PD
    kelly_path = os.path.join(_PD, "kelly_weights.parquet")
    eq = 1.0 / len(SYMBOLS)
    if not os.path.exists(kelly_path):
        return {sym: eq for sym in SYMBOLS}
    kw_df  = pd.read_parquet(kelly_path).set_index('symbol')
    raw    = {sym: float(kw_df.loc[sym, 'kelly_weight']) if sym in kw_df.index else eq
              for sym in SYMBOLS}
    total  = sum(raw.values())
    return {sym: raw[sym] / total for sym in SYMBOLS}
MIN_NOTIONAL  = 500.0  # ~0.05% du compte paper 1M€ — à recalibrer en réel
KELLY_WEIGHTS = None   # chargÃ© une fois dans run_live_pipeline

PEAK_PATH         = os.path.join(BASE_DIR, 'data', 'peak_value.json')
CACHE_PATH        = os.path.join(BASE_DIR, 'data', 'signal_cache.json')
IC_ALERT_PATH     = os.path.join(BASE_DIR, 'data', 'ic_alert.json')
LIVE_STATE_PATH   = os.path.join(BASE_DIR, 'data', 'live_state.json')
LOCK_PATH         = os.path.join(BASE_DIR, 'data', 'pipeline.lock')
CACHE_MAX_AGE_MIN = 15
LOCK_MAX_AGE_SEC  = 300
def _load_peak():
    """Charge le high-water mark et le risk_scale courant persistés entre les runs."""
    if os.path.exists(PEAK_PATH):
        with open(PEAK_PATH, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('peak', None), data.get('risk_scale', 1.0)
    return None, 1.0


def _save_peak(peak, dd, risk_scale=1.0):
    os.makedirs(os.path.dirname(PEAK_PATH), exist_ok=True)
    with open(PEAK_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'peak':       round(peak, 2),
            'dd_pct':     round(dd * 100, 3),
            'risk_scale': round(risk_scale, 4),
            'updated':    datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
        }, f)


def _acquire_lock():
    """CrÃ©e pipeline.lock. Retourne False si un run est dÃ©jÃ  actif (< 5 min)."""
    if os.path.exists(LOCK_PATH):
        age = time.time() - os.path.getmtime(LOCK_PATH)
        if age < LOCK_MAX_AGE_SEC:
            return False
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, 'w') as f:
        f.write(str(os.getpid()))
    return True


def _release_lock():
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def _load_signal_cache():
    """Charge le cache prÃ©-calculÃ© si < CACHE_MAX_AGE_MIN minutes, sinon None."""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, encoding='utf-8') as f:
        cache = json.load(f)
    ts    = ET.localize(datetime.strptime(cache['timestamp'], '%Y-%m-%d %H:%M:%S'))
    age_m = (datetime.now(ET) - ts).total_seconds() / 60
    if age_m > CACHE_MAX_AGE_MIN:
        print(f"  âš ï¸ Cache expirÃ© ({age_m:.0f} min) â€” recalcul complet.")
        return None
    print(f"  âš¡ Cache valide ({age_m:.1f} min) â€” fetch+ML ignorÃ©s, ordre en ~5s")
    return cache['signals']


def _portfolio_risk_scale(portfolio_value, peak, current_scale=1.0):
    """
    Protection immédiate (baisse instantanée) + récupération progressive (+0.25 max/run).

    Cibles DD :  DD < -3% → 0.75 | DD < -5% → 0.50 | DD < -7% → 0.25 | DD ≤ -10% → 0.0
    Récupération : la scale ne remonte que d’un palier (0.25) par run, même si le DD
    a fortement rebondi — évite le retour brutal à pleine taille après une grosse perte.
    """
    STEP = 0.25
    if peak is None or peak <= 0:
        return 1.0, 0.0
    dd = (portfolio_value - peak) / peak

    if dd <= -0.10:
        target = 0.0
    elif dd < -0.07:
        target = 0.25
    elif dd < -0.05:
        target = 0.50
    elif dd < -0.03:
        target = 0.75
    else:
        target = 1.0

    if target < current_scale:
        new_scale = target                           # baisse instantanée
    elif target > current_scale:
        new_scale = min(target, current_scale + STEP)  # montée limitée à +0.25/run
    else:
        new_scale = current_scale

    return round(new_scale, 4), dd

LOG_PATH = os.path.join(BASE_DIR, 'data', 'execution_log.csv')
_LOG_FIELDS = [
    'timestamp', 'symbol', 'action',
    'signal', 'pred_return', 'signal_size', 'vol_factor', 'effective_size',
    'regime_ok', 'notional', 'portfolio_value',
]

IC_PATH   = os.path.join(BASE_DIR, 'data', 'ic_tracker.csv')
_IC_FIELDS = ['timestamp_utc', 'symbol', 'predicted_return']

LOG_RETENTION_DAYS = 90


def _rotate_logs():
    """Tronque execution_log.csv et ic_tracker.csv à LOG_RETENTION_DAYS jours."""
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=LOG_RETENTION_DAYS)

    for path, ts_col, tz in [
        (LOG_PATH, 'timestamp',     'America/New_York'),
        (IC_PATH,  'timestamp_utc', 'UTC'),
    ]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if ts_col not in df.columns or len(df) == 0:
                continue
            df[ts_col] = pd.to_datetime(df[ts_col], utc=(tz == 'UTC'), errors='coerce')
            if tz != 'UTC':
                df[ts_col] = df[ts_col].dt.tz_localize(
                    tz, ambiguous='NaT', nonexistent='NaT'
                ).dt.tz_convert('UTC')
            before = len(df)
            df = df[df[ts_col] >= cutoff]
            removed = before - len(df)
            if removed > 0:
                df.to_csv(path, index=False)
                print(f"  🗂 Rotation {os.path.basename(path)} : -{removed} lignes ({len(df)} conservées)")
        except Exception as e:
            print(f"  ⚠️  Rotation {os.path.basename(path)} échouée ({e})")


def _log(row: dict):
    """Ajoute une ligne au CSV de log avec retry si fichier verrouille (OneDrive)."""
    import time as _time
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    for attempt in range(6):
        try:
            with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, '') for k in _LOG_FIELDS})
            return
        except PermissionError:
            if attempt < 5:
                _time.sleep(2)
            else:
                print(f"  WARNING log non ecrit apres 6 tentatives : {row.get('action','?')} {row.get('symbol','?')}")


def _save_prediction(symbol: str, pred_ret: float):
    """Enregistre chaque prÃ©diction pour le calcul d'IC live."""
    ts = datetime.now(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%S')
    write_header = not os.path.exists(IC_PATH)
    with open(IC_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_IC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({'timestamp_utc': ts, 'symbol': symbol, 'predicted_return': pred_ret})


def _sms(msg: str):
    """Envoie un SMS via l'API Free Mobile."""
    user = os.getenv('FREEMOBILE_USER')
    pwd  = os.getenv('FREEMOBILE_PASS')
    if not user or not pwd:
        print(f"  âš ï¸ SMS ignorÃ© â€” FREEMOBILE_USER/PASS manquants : {msg[:60]}")
        return
    try:
        r = requests.get(
            'https://smsapi.free-mobile.fr/sendmsg',
            params={'user': user, 'pass': pwd, 'msg': msg},
            timeout=5,
        )
        if r.status_code == 200:
            print(f"  ðŸ“± SMS envoyÃ© (HTTP 200) : {msg[:60]}")
        else:
            print(f"  âš ï¸ SMS non envoyÃ© (HTTP {r.status_code}) : {msg[:60]}")
    except Exception as e:
        print(f"  âš ï¸ SMS erreur rÃ©seau : {e} â€” {msg[:60]}")




# ─── Filtre gap pre-market ───────────────────────────────────────────────────

def _get_premarket_gaps() -> dict:
    """Récupère gaps + volumes pre-market via IBKR.

    Retourne {symbol: {'gap': float, 'pm_vol_ratio': float}}
    pm_vol_ratio = volume_premarket / avg_daily_volume (20 derniers jours).
    """
    result = {}
    neutral = {'gap': 0.0, 'pm_vol_ratio': 0.0}
    try:
        for symbol in SYMBOLS:
            try:
                current, prev_close, pm_vol = fetch_premarket_data(symbol)
                gap = (current - prev_close) / prev_close if current > 0 and prev_close > 0 else 0.0

                pm_vol_ratio = 0.0
                bt_path = os.path.join(PARQUET_DIR, f"{symbol}_1h.parquet")
                if os.path.exists(bt_path) and pm_vol > 0:
                    df = pd.read_parquet(bt_path)
                    daily_vol = df['Volume'].resample('1D').sum().dropna()
                    avg_daily = float(daily_vol.tail(20).mean()) if len(daily_vol) >= 5 else 0
                    pm_vol_ratio = pm_vol / avg_daily if avg_daily > 0 else 0.0

                result[symbol] = {'gap': gap, 'pm_vol_ratio': round(pm_vol_ratio, 4), 'current': current}
            except Exception as e:
                print(f"     ⚠️ Data pre-market {symbol} indisponible ({e})")
                result[symbol] = dict(neutral)
    except Exception as e:
        print(f"  ⚠️ Gaps pre-market indisponibles ({e}) — filtre neutralisé")
        return {s: dict(neutral) for s in SYMBOLS}
    return result


def _gap_scale(signal: int, gap: float, pm_vol_ratio: float = 0.0) -> float:
    """Ajuste la taille selon gap pre-market ET volume pre-market.

    gap < -2%  + vol élevé  (>15% quotidien) : x0.50  gap baissier fort confirmé
    gap < -2%  + vol faible (<3%  quotidien) : x0.80  gap sans volume = probable bruit
    gap < -1%  + vol élevé                  : x0.75  gap modéré confirmé
    gap < -1%  + vol faible                 : x1.00  gap modéré sans volume = ignoré
    sinon                                   : x1.00
    """
    if signal != 1 or gap >= -0.01:
        return 1.0
    high_vol = pm_vol_ratio > 0.15
    if gap < -0.02:
        return 0.50 if high_vol else 0.80
    return 0.75 if high_vol else 1.00


# â”€â”€â”€ RÃ©gime filter (identique Ã  backtesting/engine.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# MA window du filtre downtrend par ticker -- identique a backtesting/engine.py
_TICKER_MA_WINDOW = {
    'AAPL':  30,
    'MSFT':  50,
    'GOOGL': 50,
    'COST':  20,
}
_DEFAULT_MA_WINDOW = 50


def _regime_ok(df, symbol=""):
    ma_window = _TICKER_MA_WINDOW.get(symbol, _DEFAULT_MA_WINDOW)
    ret     = df['Close'].pct_change()
    vol_20  = ret.rolling(20).std().iloc[-1]
    vol_100 = ret.rolling(100).std().iloc[-1]
    ma_n    = df['Close'].rolling(ma_window).mean().iloc[-1]
    close   = df['Close'].iloc[-1]

    high_vol  = vol_20 > 1.5 * vol_100 if vol_100 > 0 else False
    downtrend = close < ma_n

    return not (high_vol or downtrend)
def _vol_scale(df):
    """Facteur de rÃ©duction 0.5 si vol_10 > vol_100 (hors filtre rÃ©gime), sinon 1.0."""
    ret     = df['Close'].pct_change()
    vol_10  = ret.rolling(10).std().iloc[-1]
    vol_100 = ret.rolling(100).std().iloc[-1]
    if vol_100 == 0:
        return 1.0
    ratio = vol_10 / vol_100
    return 0.5 if 1.0 < ratio < 1.5 else 1.0


# â”€â”€â”€ DonnÃ©es live â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fetch_all_data():
    """Met Ã  jour les barres Alpaca en mode incrÃ©mental (conserve l'historique run_all.py)."""
    print("  ðŸ“¡ Mise Ã  jour donnÃ©es Alpaca...")

    for symbol in SYMBOLS + ['SPY']:
        name = f"{symbol}_1h" if symbol in SYMBOLS else "SPY_context_1h"
        df_new = pd.DataFrame()
        for attempt in range(3):
            try:
                df_new = fetch_bars(symbol, days=5, freq=STRATEGY['frequency'])
                break
            except Exception as e:
                print(f"     ⚠️ {symbol} tentative {attempt+1}/3 — {e}")
                if attempt < 2:
                    import time; time.sleep(8)
        if df_new.empty:
            print(f"     ❌ {symbol} — echec apres 3 tentatives")
            return False
        existing = load_data(name)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df_new])
            df = df[~df.index.duplicated(keep='last')].sort_index()
        else:
            df = df_new
        save_data(df, name)
        print(f"     ✅ {symbol} — {len(df)} barres ({len(df_new)} nouvelles)")

    # VIX via Yahoo daily + ffill horaire -- identique a run_all.py (1 valeur VIX par jour)
    try:
        import yfinance as yf
        vix_daily = yf.Ticker('^VIX').history(period='60d', interval='1d', auto_adjust=True)
        if not vix_daily.empty:
            existing_vix = load_data('VIX_context_1h')
            if existing_vix is not None and not existing_vix.empty:
                vix_resampled = vix_daily[["Close"]].rename(columns={"Close": "Close"})
                vix_resampled = vix_daily.reindex(existing_vix.index, method='ffill').dropna(subset=['Close'])
                vix_merged = pd.concat([existing_vix, vix_resampled])
                vix_merged = vix_merged[~vix_merged.index.duplicated(keep='last')].sort_index()
                save_data(vix_merged, 'VIX_context_1h')
                print(f"     VIX (Yahoo daily) -- {len(vix_merged)} barres")
    except Exception as e:
        print(f"     VIX Yahoo indisponible ({e})")

    # 15min IBKR — mise a jour incrementale (features intraday)
    try:
        from execution.broker import fetch_bars_15min
        from features.intraday import RAW_15MIN_DIR
        import os as _os
        print("  Mise a jour 15min IBKR...")
        for symbol in SYMBOLS:
            cache_path = _os.path.join(RAW_15MIN_DIR, f"{symbol}_15min.parquet")
            df_new = fetch_bars_15min(symbol, years=1)
            if not df_new.empty:
                if _os.path.exists(cache_path):
                    df_ex = pd.read_parquet(cache_path)
                    df = pd.concat([df_ex, df_new])
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                else:
                    df = df_new
                df.to_parquet(cache_path, compression='snappy')
                print(f"     {symbol} 15min OK ({len(df_new)} nouvelles barres)")
    except Exception as e:
        print(f"  15min indisponible ({e}) — features intraday sur donnees precedentes")

    # IV 1h IBKR — mise a jour incrementale (features VRP/iv_change)
    try:
        from features.ibkr_derivatives import fetch_and_cache_iv
        print("  Mise a jour IV IBKR...")
        for symbol in SYMBOLS:
            df = fetch_and_cache_iv(symbol)
            if not df.empty:
                print(f"     {symbol} IV OK ({len(df)} barres)")
    except Exception as e:
        print(f"  IV indisponible ({e}) — features IV sur donnees precedentes")

    return True


def _run_module(mod):
    """Lance un module Python comme subprocess (mÃªme env que live_pipeline)."""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(
        [sys.executable, '-m', mod],
        capture_output=True, text=True,
        encoding='utf-8', errors='replace',
        env=env
    )
    if result.returncode != 0:
        print(f"     âŒ Erreur dans {mod} :\n{result.stderr[-400:]}")
        return False
    return True


# â”€â”€â”€ ExÃ©cution des ordres â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _execute(symbol, signal, signal_size, vol_factor, pred_ret, regime_ok, positions, portfolio_value, risk_scale=1.0, gap_factor=1.0, is_retained=False):
    in_position = symbol in positions
    weight      = KELLY_WEIGHTS.get(symbol, 1.0 / len(SYMBOLS)) if KELLY_WEIGHTS else 1.0 / len(SYMBOLS)
    effective   = signal_size * vol_factor * risk_scale * gap_factor
    ts          = datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')

    base_row = dict(
        timestamp=ts, symbol=symbol, signal=signal,
        pred_return=round(pred_ret, 6), signal_size=round(signal_size, 4),
        vol_factor=vol_factor, effective_size=round(effective, 4),
        regime_ok=regime_ok, portfolio_value=round(portfolio_value, 2),
    )

    # SL/TP — vérifie AVANT la logique signal pour sortir immédiatement si seuil atteint
    if in_position:
        pos_data       = positions[symbol]
        pnl_pct        = pos_data['unrealized_plpc'] / 100.0
        sl_pct, tp_pct = _TICKER_RISK.get(symbol, _DEFAULT_RISK)
        if pnl_pct <= -sl_pct:
            close_position(symbol)
            mv = pos_data['market_value']
            print(f'     [SL] STOP LOSS  -- {pnl_pct:+.2%} depuis entree ({mv:.0f} $)')
            _log({**base_row, 'action': 'SL_CLOSE', 'notional': round(mv, 2)})
            _sms(f"[V3] STOP LOSS {symbol} — {pnl_pct:+.2%} | ptf={portfolio_value:.0f}$")
            return
        if pnl_pct >= tp_pct:
            close_position(symbol)
            mv = pos_data['market_value']
            print(f"     \U0001f3af TAKE PROFIT — {pnl_pct:+.2%} depuis entrée ({mv:.0f} $)")
            _log({**base_row, 'action': 'TP_CLOSE', 'notional': round(mv, 2)})
            _sms(f"[V3] TAKE PROFIT {symbol} — {pnl_pct:+.2%} | ptf={portfolio_value:.0f}$")
            return

    if signal == 1 and regime_ok and is_retained and not in_position:
        print(f"     Signal retenu (N_HOLD) — pas d'entree, attente signal frais")
        _log({**base_row, 'action': 'SKIP_RETAINED', 'notional': 0})
        return

    if signal == 1 and regime_ok and not in_position:
        macro_risk, macro_reason = check_macro_risk(SYMBOLS)
        if macro_risk:
            print(f"     ⛔ MACRO BLOCK — {macro_reason}")
            _log({**base_row, 'action': 'SKIP_MACRO', 'notional': 0})
            return
        notional = portfolio_value * weight * effective
        if notional >= MIN_NOTIONAL:
            trade = place_order(symbol, 'buy', notional)
            time.sleep(5)
            status = trade.orderStatus.status if trade is not None else 'Unknown'
            if status in ('Filled', 'Submitted', 'PreSubmitted', 'PendingSubmit'):
                print(f"     [OK] ACHAT  -- {notional:.0f} $ notional ({effective:.0%} taille)")
                _log({**base_row, 'action': 'BUY', 'notional': round(notional, 2)})
                _sms(f"[V3] ACHAT {symbol} -- {notional:.0f}$ ({effective:.0%}) | pred={pred_ret:+.3%} | ptf={portfolio_value:.0f}$")
            else:
                print(f"     [FAIL] Ordre annule ({status}) -- marche ferme ou erreur IBKR")
                _log({**base_row, 'action': f'ORDER_FAILED:{status}', 'notional': round(notional, 2)})
        else:
            print(f"     âš ï¸ Notionnel trop faible ({notional:.2f} $)")
            _log({**base_row, 'action': 'SKIP_NOTIONAL', 'notional': round(notional, 2)})

    elif (signal != 1 or not regime_ok) and in_position:
        close_position(symbol)
        mv = positions[symbol]['market_value']
        print(f'     [CLOSE] Position fermee ({mv:.0f} $)')
        _log({**base_row, 'action': 'CLOSE', 'notional': round(mv, 2)})
        _sms(f'[V3] CLOTURE {symbol} -- {mv:.0f}$ | ptf={portfolio_value:.0f}$')

    elif signal == 1 and regime_ok and in_position:
        mv = positions[symbol]['market_value']
        print(f"     âœ… Position maintenue ({mv:.0f} $)")
        _log({**base_row, 'action': 'HOLD', 'notional': round(mv, 2)})

    else:
        reason = "rÃ©gime bloquÃ©" if not regime_ok else "signal neutre"
        print(f"     âšª Aucune action ({reason})")
        _log({**base_row, 'action': 'FLAT', 'notional': 0})


# â”€â”€â”€ ClÃ´ture forcÃ©e EOD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _force_close_all_eod(positions, portfolio_value):
    """ClÃ´ture systÃ©matique toutes les positions 30 min avant la fermeture du marchÃ©."""
    if not positions:
        return
    print("  ðŸ”” EOD â€” ClÃ´ture forcÃ©e de toutes les positions...")
    for symbol, pos in positions.items():
        close_position(symbol)
        mv = pos['market_value']
        print(f"     âœ… {symbol} clÃ´turÃ© ({mv:.0f} $)")
        _log({
            'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol, 'action': 'EOD_CLOSE',
            'notional': round(mv, 2),
            'portfolio_value': round(portfolio_value, 2),
        })


# â”€â”€â”€ PrÃ©-calcul 15h20 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _precompute_signals():
    """Fetch + features + ML â†’ cache JSON. Aucun ordre placÃ©. LancÃ© Ã  15h20."""
    now_et = datetime.now(ET)
    print(f"\n{'='*58}")
    print(f"  PRÃ‰-CALCUL â€” {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*58}")

    if not _fetch_all_data():
        print("  âŒ Ã‰chec donnÃ©es â€” prÃ©-calcul annulÃ©.")
        return

    print("  ðŸ”§ Feature engineering + prÃ©diction...")
    for mod in ['features.alpha_factors', 'features.ml_features', 'models.prediction']:
        if not _run_module(mod):
            print(f"  âŒ ArrÃªt sur {mod}")
            return
        print(f"     âœ… {mod}")

    cache = {"timestamp": now_et.strftime('%Y-%m-%d %H:%M:%S'), "signals": {}}
    for symbol in SYMBOLS:
        sig_path = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
        bt_path  = os.path.join(PARQUET_DIR, f"{symbol}_1h.parquet")
        if not os.path.exists(sig_path):
            print(f"     âš ï¸ {symbol} â€” signaux introuvables, ignorÃ©")
            continue

        signals     = pd.read_parquet(sig_path)
        latest      = signals.iloc[-1]
        signal      = int(latest['Signal'])
        pred_ret    = float(latest.get('Predicted_Return', latest.get('Prob_Up', 0.0)))
        signal_size = float(latest.get('Signal_Size', float(signal)))

        is_retained = False
        N_HOLD_BARS = 2
        if signal == 0 and len(signals) > N_HOLD_BARS:
            recent      = signals.iloc[-(N_HOLD_BARS + 1):-1]
            last_active = recent[recent['Signal'] == 1]
            if not last_active.empty:
                signal      = 1
                signal_size = float(last_active.iloc[-1].get('Signal_Size', 1.0))
                is_retained = True

        regime_ok  = False
        vol_factor = 1.0
        if os.path.exists(bt_path):
            df_raw     = pd.read_parquet(bt_path)
            regime_ok  = _regime_ok(df_raw, symbol)
            vol_factor = _vol_scale(df_raw)

        cache["signals"][symbol] = {
            "signal": signal, "signal_size": signal_size,
            "pred_ret": pred_ret, "regime_ok": regime_ok, "vol_factor": vol_factor,
            "is_retained": is_retained,
        }
        sig_str = f"ðŸŸ¢ Long {signal_size:.0%}" if signal == 1 else "ðŸ”´ Neutre"
        print(f"     {symbol}: {sig_str}  pred={pred_ret:+.3%}  rÃ©gime={'âœ…' if regime_ok else 'â›”'}")

    # Filtre gap pre-market — enrichit chaque signal avec gap_factor
    print("  📡 Gaps pre-market IBKR...")
    gaps = _get_premarket_gaps()
    for sym in SYMBOLS:
        if sym not in cache["signals"]:
            continue
        d            = gaps.get(sym, {'gap': 0.0, 'pm_vol_ratio': 0.0})
        gap          = d['gap']
        pm_vol_ratio = d['pm_vol_ratio']
        gs           = _gap_scale(cache["signals"][sym]["signal"], gap, pm_vol_ratio)
        cache["signals"][sym]["gap"]          = round(gap, 4)
        cache["signals"][sym]["pm_vol_ratio"] = pm_vol_ratio
        cache["signals"][sym]["gap_factor"]   = gs
        gap_str = f"{gap:+.2%}" if gap != 0.0 else "N/A"
        vol_str = f"vol={pm_vol_ratio:.1%}" if pm_vol_ratio > 0 else "vol=N/A"
        note    = f" → ×{gs:.0%} ⚠️ gap contre signal" if gs < 1.0 else ""
        print(f"     {sym}: gap={gap_str}  {vol_str}{note}")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    print(f"\n  âœ… Cache sauvegardÃ© â†’ {len(cache['signals'])} signaux prÃªts pour 15h30")
    print(f"{'='*58}\n")


# â”€â”€â”€ Pipeline principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _check_model_freshness():
    """Vérifie l'âge des modèles entraînés. Alerte SMS si > 30 jours."""
    MODEL_MAX_DAYS = 30
    ages = []
    for sym in SYMBOLS:
        pkl = os.path.join(PARQUET_DIR, f"{sym}_best_model.pkl")
        if os.path.exists(pkl):
            age_days = (time.time() - os.path.getmtime(pkl)) / 86400
            ages.append((sym, age_days))

    if not ages:
        print("  ⚠️  Modèles introuvables — run_all.py n'a pas encore tourné ?")
        return

    max_age   = max(d for _, d in ages)
    ref_sym, ref_age = max(ages, key=lambda x: x[1])
    ref_date  = datetime.fromtimestamp(
        os.path.getmtime(os.path.join(PARQUET_DIR, f"{ref_sym}_best_model.pkl")), ET
    ).strftime('%d %b %Y')

    if max_age > MODEL_MAX_DAYS:
        msg = f"[V3] ⚠️ Modèles périmés ({max_age:.0f}j, dernier run: {ref_date}) — relancer run_all.py"
        print(f"  ⚠️  Modèles : {max_age:.0f} jours ({ref_date}) — relancer run_all.py recommandé")
        _sms(msg)
    else:
        print(f"  ✅ Modèles : {max_age:.0f} jours ({ref_date})")


def run_live_pipeline(force=False):
    if not _acquire_lock():
        print("  âš ï¸ Pipeline dÃ©jÃ  en cours d'exÃ©cution â€” abandon pour Ã©viter doublon.")
        return
    try:
        _run_live_pipeline_inner(force)
    finally:
        _release_lock()


def _run_live_pipeline_inner(force=False):
    now_et = datetime.now(ET)
    print(f"\n{'='*58}")
    print(f"  LIVE PIPELINE â€” {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    if force:
        print(f"  âš ï¸  MODE FORCE â€” ordres simulÃ©s (marchÃ© peut Ãªtre fermÃ©)")
    print(f"{'='*58}")

    global KELLY_WEIGHTS
    KELLY_WEIGHTS = _load_kelly_weights()
    eq = 1.0 / len(SYMBOLS)
    kw_str = "  ".join(f"{s}:{KELLY_WEIGHTS[s]/eq:+.0%}" for s in SYMBOLS)
    print(f"  âš–ï¸  Kelly weights : {kw_str}")

    _check_model_freshness()

    if not IBKR_AVAILABLE:
        print("  âŒ ib_insync non installÃ© â€” pip install ib_insync")
        return

    # 1. VÃ©rification marchÃ© ouvert
    if not force and not is_market_open():
        print("  â›” MarchÃ© fermÃ© â€” aucune action.")
        return

    _rotate_logs()

    # 2. Compte
    account = get_account()

    # ClÃ´ture forcÃ©e si moins de 30 min avant la fermeture (16h00 ET = 22h00 Paris)
    if not force:
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        minutes_to_close = (market_close - now_et).total_seconds() / 60
        if 0 < minutes_to_close < 30:
            positions = get_positions()
            _force_close_all_eod(positions, account['portfolio_value'])
            acc_eod = get_account()
            _sms(f"[V3] EOD -- toutes positions cloturees | Portefeuille : {acc_eod['portfolio_value']:.2f} $")
            with open(LIVE_STATE_PATH, 'w', encoding='utf-8') as _f:
                json.dump({
                    'portfolio_value': acc_eod['portfolio_value'],
                    'cash': acc_eod['cash'],
                    'equity': acc_eod.get('equity', 0),
                    'risk_scale': 1.0,
                    'dd': 0.0,
                    'peak': acc_eod['portfolio_value'],
                    'positions': {},
                    'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
                }, _f, indent=2)
            print('  [EOD] Pipeline arrete - moins de 30 min avant cloture.')
            return
    print(f"  Compte : {account['portfolio_value']:.0f} $  |  Cash : {account['cash']:.0f} $")

    # Portfolio stop-loss progressif basÃ© sur le drawdown depuis le pic
    portfolio_value = account['portfolio_value']
    peak, current_scale = _load_peak()
    if peak is None or portfolio_value > peak:
        peak = portfolio_value
        current_scale = 1.0  # nouveau pic → reset scale
    risk_scale, dd = _portfolio_risk_scale(portfolio_value, peak, current_scale)
    _save_peak(peak, dd, risk_scale)


    dd_str = f"{dd:+.2%}" if dd < 0 else "â€”"
    def _label(s):
        if s >= 1.0:   return "✅ Normal"
        if s >= 0.75:  return "\U0001f7e1 \xd70.75 (r\xe9cup.)"
        if s >= 0.50:  return "\U0001f7e0 \xd70.50 (r\xe9cup.)"
        if s > 0.0:    return "\U0001f534 \xd70.25 (r\xe9cup.)"
        return "\U0001f6a8 -10% DD"
    print(f"  \U0001f4c9 Portefeuille : DD={dd_str}  Pic={peak:.0f}$  →  Risk scale={risk_scale:.0%}  {_label(risk_scale)}")

    if risk_scale == 0.0:
        positions = get_positions()
        _force_close_all_eod(positions, portfolio_value)
        _sms(f"[V3] ðŸš¨ DD > 10% â€” toutes positions clÃ´turÃ©es | ptf={portfolio_value:.0f}$")
        print("  ðŸš¨ Drawdown > 10% â€” pipeline arrÃªtÃ©, tout est fermÃ©.")
        return

    # 3. Cache prÃ©-calculÃ© (15h20) ou pipeline complet
    cached_signals = _load_signal_cache()

    if cached_signals is None:
        if not _fetch_all_data():
            print("  âŒ Ã‰chec tÃ©lÃ©chargement donnÃ©es â€” pipeline annulÃ©.")
            return
        print("  ðŸ”§ Feature engineering + prÃ©diction...")
        for mod in ['features.alpha_factors', 'features.ml_features', 'models.prediction']:
            if not _run_module(mod):
                print(f"  âŒ ArrÃªt sur {mod}")
                return
            print(f"     âœ… {mod}")


    # Filtre gap pre-market (chemin sans cache — si cache: gap_factor lu depuis le cache)
    premarket_gaps = _get_premarket_gaps() if cached_signals is None else {}
    # 4. Lecture signaux + rÃ©gime + exÃ©cution
    positions = get_positions()

    print(f"\n  {'Ticker':<6}  {'Signal':<14}  {'Prob':>5}  {'RÃ©gime':<10}  Action")
    print("  " + "-"*54)

    for symbol in SYMBOLS:
        if cached_signals and symbol in cached_signals:
            c           = cached_signals[symbol]
            signal      = c['signal']
            pred_ret    = c['pred_ret']
            signal_size = c['signal_size']
            regime_ok   = c['regime_ok']
            vol_factor  = c['vol_factor']
            gap_factor  = c.get('gap_factor', 1.0)
            is_retained = c.get('is_retained', False)
        else:
            sig_path = os.path.join(PARQUET_DIR, f"{symbol}_Signals.parquet")
            bt_path  = os.path.join(PARQUET_DIR, f"{symbol}_1h.parquet")

            if not os.path.exists(sig_path):
                print(f"  {symbol:<6}  âŒ Signaux introuvables")
                continue

            signals     = pd.read_parquet(sig_path)
            latest      = signals.iloc[-1]
            signal      = int(latest['Signal'])
            pred_ret    = float(latest.get('Predicted_Return', latest.get('Prob_Up', 0.0)))
            signal_size = float(latest.get('Signal_Size', float(signal)))

            is_retained = False
            N_HOLD_BARS = 2
            if signal == 0 and len(signals) > N_HOLD_BARS:
                recent      = signals.iloc[-(N_HOLD_BARS + 1):-1]
                last_active = recent[recent['Signal'] == 1]
                if not last_active.empty:
                    signal      = 1
                    signal_size = float(last_active.iloc[-1].get('Signal_Size', 1.0))
                    is_retained = True

            regime_ok  = False
            vol_factor = 1.0
            if os.path.exists(bt_path):
                df_raw = pd.read_parquet(bt_path)
                # Injecter le prix pre-market courant pour que le regime reflète
                # la realite du marche avant l'ouverture RTH
                pm_d = premarket_gaps.get(symbol, {'gap': 0.0, 'pm_vol_ratio': 0.0, 'current': 0.0})
                pm_price = pm_d.get('current', 0.0)
                if pm_price > 0:
                    df_raw = df_raw.copy()
                    df_raw.iloc[-1, df_raw.columns.get_loc('Close')] = pm_price
                regime_ok  = _regime_ok(df_raw, symbol)
                vol_factor = _vol_scale(df_raw)

            pm_d       = premarket_gaps.get(symbol, {'gap': 0.0, 'pm_vol_ratio': 0.0})
            gap_factor = _gap_scale(signal, pm_d['gap'], pm_d['pm_vol_ratio'])

        _save_prediction(symbol, pred_ret)

        size_str   = f"{signal_size:.0%}" if signal == 1 else "â€”"
        sig_str    = f"ðŸŸ¢ Long {size_str}" if signal == 1 else "ðŸ”´ Neutre"
        regime_str = "âœ… Normal" if regime_ok else "â›” BloquÃ©"
        print(f"  {symbol:<6}  {sig_str:<16}  {pred_ret:+.3%}  {regime_str:<10}", end="  ")

        _execute(symbol, signal, signal_size, vol_factor, pred_ret, regime_ok, positions, portfolio_value, risk_scale, gap_factor, is_retained)

    _check_live_ic()

    # Sauvegarde etat live apres execution des ordres (positions a jour pour le dashboard)
    acc_final = get_account()
    positions_snap = get_positions()
    with open(LIVE_STATE_PATH, 'w', encoding='utf-8') as _f:
        json.dump({
            'portfolio_value': acc_final['portfolio_value'],
            'cash': acc_final['cash'],
            'equity': acc_final.get('equity', 0),
            'risk_scale': risk_scale,
            'dd': dd,
            'peak': peak,
            'positions': {s: {k: float(v) for k, v in p.items()} for s, p in positions_snap.items()},
            'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'),
        }, _f, indent=2)

    print(f"\n{'='*58}\n")
    _log({
        'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': 'PIPELINE', 'action': 'END',
        'portfolio_value': round(acc_final['portfolio_value'], 2),
    })


def _check_live_ic():
    """Vérifie si l'IC live est négatif sur IC_BAD_STREAK barres consécutives.
    Envoie un SMS d'alerte au plus une fois par jour.
    """
    IC_WINDOW    = 20   # barres par fenêtre rolling IC
    IC_MIN_OBS   = 5    # minimum par fenêtre pour calculer
    IC_BAD_STREAK = 10  # barres rolling IC < 0 consécutives pour déclencher

    # Cooldown : une seule alerte par jour
    today = datetime.now(ET).strftime('%Y-%m-%d')
    if os.path.exists(IC_ALERT_PATH):
        with open(IC_ALERT_PATH, encoding='utf-8') as f:
            last = json.load(f).get('last_alert_date', '')
        if last == today:
            return

    if not os.path.exists(IC_PATH):
        return

    try:
        from scipy.stats import spearmanr

        preds = pd.read_csv(IC_PATH, parse_dates=['timestamp_utc'])
        preds['timestamp_utc'] = pd.to_datetime(preds['timestamp_utc'], utc=True)

        degraded = []
        for symbol in SYMBOLS:
            sym_p = preds[preds['symbol'] == symbol].copy().sort_values('timestamp_utc')
            if len(sym_p) < IC_WINDOW:
                continue

            bt_path = os.path.join(PARQUET_DIR, f"{symbol}_1h.parquet")
            if not os.path.exists(bt_path):
                continue

            prices = pd.read_parquet(bt_path)
            prices.index = pd.to_datetime(prices.index, utc=True)
            prices['ret'] = prices['Close'].pct_change()

            actual = []
            for ts in sym_p['timestamp_utc']:
                nxt = prices[prices.index > ts]
                actual.append(float(nxt['ret'].iloc[0]) if len(nxt) > 0 else np.nan)
            sym_p['actual'] = actual
            sym_p = sym_p.dropna(subset=['actual'])

            if len(sym_p) < IC_WINDOW:
                continue

            rolling = []
            for i in range(len(sym_p)):
                chunk = sym_p.iloc[max(0, i - IC_WINDOW + 1): i + 1]
                if len(chunk) >= IC_MIN_OBS:
                    r, _ = spearmanr(chunk['predicted_return'], chunk['actual'])
                    rolling.append(r)

            if len(rolling) < IC_BAD_STREAK:
                continue

            last_streak = rolling[-IC_BAD_STREAK:]
            if all(r < 0 for r in last_streak):
                degraded.append(f"{symbol}(IC={last_streak[-1]:+.3f})")

        if degraded:
            msg = (f"[V3] ⚠️ IC live dégradé {IC_BAD_STREAK} barres : "
                   f"{', '.join(degraded)} — relancer run_all.py")
            print(f"  ⚠️  IC dégradé : {', '.join(degraded)}")
            _sms(msg)
            os.makedirs(os.path.dirname(IC_ALERT_PATH), exist_ok=True)
            with open(IC_ALERT_PATH, 'w', encoding='utf-8') as f:
                json.dump({'last_alert_date': today, 'symbols': degraded}, f)

    except Exception as e:
        print(f"  ⚠️  Vérification IC live échouée ({e})")


def _check_gateway():
    """Lance IB Gateway si nécessaire, attend la connexion, envoie SMS de statut."""
    from execution.broker import _ib, IBKR_HOST, IBKR_PORT

    def _try_connect():
        try:
            ib = _ib()
            return ib.isConnected()
        except Exception:
            return False

    if _try_connect():
        msg = "[V3] ✅ IB Gateway connecté — pipeline prêt pour 15h20"
        print(f"  {msg}")
        _sms(msg)
        return

    print("  ⚡ IB Gateway non connecté — lancement via IBC...")
    subprocess.Popen(
        [r"C:\IBC\StartGateway.bat"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        shell=True,
    )

    for i in range(12):
        time.sleep(10)
        print(f"  ⏳ Attente connexion IB Gateway... ({(i+1)*10}s)")
        if _try_connect():
            msg = "[V3] ✅ IB Gateway démarré et connecté — pipeline prêt pour 15h20"
            print(f"  {msg}")
            _sms(msg)
            return

    msg = "[V3] ⚠️ IB Gateway non connecté après 2 min — vérifiez manuellement"
    print(f"  ❌ {msg}")
    _sms(msg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force',         action='store_true', help='Bypass market hours check')
    parser.add_argument('--precompute',    action='store_true', help='Pré-calcul 15h20 : fetch+ML sans ordres')
    parser.add_argument('--check-gateway', action='store_true', help='Vérifie IB Gateway et envoie SMS si KO')
    args = parser.parse_args()
    if args.check_gateway:
        _check_gateway()
    elif args.precompute:
        _precompute_signals()
    else:
        run_live_pipeline(force=args.force)

