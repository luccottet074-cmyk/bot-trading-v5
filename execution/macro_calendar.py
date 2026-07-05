"""
Détecte les jours à risque macro élevé pour bloquer les nouveaux BUY.
Utilise l'API Finnhub (finnhub-python) pour les événements US high-impact.
NFP est calculé algorithmiquement en fallback si pas de clé Finnhub.
Cache JSON quotidien pour éviter les appels répétés.
"""
import datetime
import json
import os
import pytz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET = pytz.timezone('America/New_York')
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'macro_cache.json')

# ---------------------------------------------------------------------------
# FOMC hardcodé — annonce TOUJOURS à 14:00 ET, sans exception
# Source officielle à vérifier chaque janvier : federalreserve.gov/monetarypolicy/fomccalendars.htm
# Finnhub gratuit ne donne pas accès au calendrier économique (403) → fallback obligatoire
# ---------------------------------------------------------------------------
FOMC_DATES_2026 = {
    datetime.date(2026, 1, 28),   # Jan 27-28  ✅ federalreserve.gov
    datetime.date(2026, 3, 18),   # Mar 17-18  ✅
    datetime.date(2026, 4, 29),   # Avr 28-29  ✅
    datetime.date(2026, 6, 17),   # Jun 16-17  ✅
    datetime.date(2026, 7, 29),   # Jul 28-29  ✅
    datetime.date(2026, 9, 16),   # Sep 15-16  ✅ (avec projections)
    datetime.date(2026, 10, 28),  # Oct 27-28  ✅
    datetime.date(2026, 12, 9),   # Déc 8-9    ✅ (avec projections)
}
FOMC_HOUR_ET = 14  # 14:00 ET — immuable

# CPI — toujours 08:30 ET — source : bls.gov/schedule/news_release/cpi.htm
CPI_DATES_2026 = {
    datetime.date(2026, 1, 13),
    datetime.date(2026, 2, 13),
    datetime.date(2026, 3, 11),
    datetime.date(2026, 4, 10),
    datetime.date(2026, 5, 12),
    datetime.date(2026, 6, 10),   # passé
    datetime.date(2026, 7, 14),
    datetime.date(2026, 8, 12),
    datetime.date(2026, 9, 11),
    datetime.date(2026, 10, 14),
    datetime.date(2026, 11, 10),
    datetime.date(2026, 12, 10),
}

# PPI — toujours 08:30 ET — source : bls.gov (calendrier vérifié mois par mois)
PPI_DATES_2026 = {
    datetime.date(2026, 1, 14),   # PPI Nov 2025 (décalé)
    datetime.date(2026, 1, 30),   # PPI Déc 2025 (double janvier)
    datetime.date(2026, 2, 27),
    datetime.date(2026, 3, 18),   # même jour que FOMC → déjà bloqué
    datetime.date(2026, 4, 14),
    datetime.date(2026, 5, 13),
    datetime.date(2026, 6, 11),   # passé
    datetime.date(2026, 7, 15),
    datetime.date(2026, 8, 13),
    datetime.date(2026, 9, 10),   # avant CPI ce mois-ci
    datetime.date(2026, 10, 15),
    datetime.date(2026, 11, 13),
    datetime.date(2026, 12, 15),
}
CPI_PPI_HOUR_ET = 8  # 08:30 ET — immuable

# PCE (Personal Income & Outlays) + GDP Advance — toujours cotés le même jour — 08:30 ET
# Source : bea.gov — à mettre à jour chaque janvier
# Note : Jan 22 est à 10:00 ET (exception) mais la fenêtre [-1h,+2h] depuis 08:30 la couvre
PCE_DATES_2026 = {
    datetime.date(2026, 1, 22),
    datetime.date(2026, 2, 20),
    datetime.date(2026, 3, 13),
    datetime.date(2026, 4, 9),
    datetime.date(2026, 4, 30),
    datetime.date(2026, 5, 28),
    datetime.date(2026, 6, 25),   # passé
    datetime.date(2026, 7, 30),
    datetime.date(2026, 8, 26),
    datetime.date(2026, 9, 30),
    datetime.date(2026, 10, 29),
    datetime.date(2026, 11, 25),
    datetime.date(2026, 12, 23),
}

# Mots-clés des événements vraiment systémiques pour les tech stocks
# Finnhub "high" inclut aussi des événements mineurs (Home Sales, Michigan...) — on filtre
SYSTEMIC_KEYWORDS = [
    'nonfarm payroll',       # NFP
    'unemployment rate',     # taux chômage (même jour NFP)
    'inflation rate',        # CPI
    'core inflation',        # Core CPI
    'consumer price',        # CPI variante
    'ppi',                   # PPI
    'producer price',        # PPI variante
    'fomc',                  # décision Fed
    'fed rate',              # taux Fed
    'interest rate decision',
    'gdp',                   # GDP (trimestriel)
    'gross domestic',
    'initial jobless',       # demandes chômage hebdo (moins fort mais significatif)
]


def _is_systemic(event_name: str) -> bool:
    name_lower = event_name.lower()
    return any(kw in name_lower for kw in SYSTEMIC_KEYWORDS)


def _first_friday(year: int, month: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    return d + datetime.timedelta(days=(4 - d.weekday()) % 7)


def _is_nfp_day(today: datetime.date) -> bool:
    return today == _first_friday(today.year, today.month)


def _check_830_events(now_et: datetime.datetime) -> tuple[bool, str]:
    """Détecte CPI / PPI / PCE+GDP via les listes hardcodées. Annonce à 08:30 ET.
    Bloque dans la fenêtre [-1h, +2h] = 07:30 → 10:30 ET."""
    today = now_et.date()
    if today in CPI_DATES_2026:
        label = "CPI"
    elif today in PPI_DATES_2026:
        label = "PPI"
    elif today in PCE_DATES_2026:
        label = "PCE/GDP"
    else:
        return False, ""
    release_et = now_et.replace(hour=CPI_PPI_HOUR_ET, minute=30, second=0, microsecond=0)
    hours_diff = (release_et - now_et).total_seconds() / 3600
    if -1 <= hours_diff <= 2:
        return True, f"{label} release (08:30 ET) — pas de nouveau BUY"
    return False, ""


def _check_fomc_hardcoded(now_et: datetime.datetime) -> tuple[bool, str]:
    """Détecte le FOMC via la liste hardcodée. Annonce toujours à 14:00 ET.
    Bloque dans la fenêtre [-1h, +2h] = 13:00 → 16:00 ET."""
    today = now_et.date()
    if today not in FOMC_DATES_2026:
        return False, ""
    fomc_et = now_et.replace(hour=FOMC_HOUR_ET, minute=0, second=0, microsecond=0)
    hours_diff = (fomc_et - now_et).total_seconds() / 3600
    if -1 <= hours_diff <= 2:
        return True, f"FOMC decision ({fomc_et.strftime('%H:%M')} ET) — pas de nouveau BUY"
    return False, ""


def _fetch_events(today: datetime.date) -> list:
    """
    Retourne la liste des événements US high-impact du jour via Finnhub.
    Résultat mis en cache dans data/macro_cache.json (1 appel par jour max).
    """
    # Lecture cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding='utf-8') as f:
                cache = json.load(f)
            if cache.get('date') == today.isoformat():
                return cache.get('events', [])
        except Exception:
            pass

    token = os.getenv('FINNHUB_TOKEN', '')
    if not token:
        return []

    try:
        import finnhub
        client = finnhub.Client(api_key=token)
        date_str = today.isoformat()
        data = client.calendar_economic(_from=date_str, to=date_str)
        events = [
            e for e in data.get('economicCalendar', [])
            if e.get('country') == 'US' and e.get('impact') == 'high'
        ]
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'date': today.isoformat(), 'events': events}, f, indent=2)
        return events
    except Exception:
        return []


def check_macro_risk(symbols: list) -> tuple[bool, str]:
    """
    Retourne (is_risky: bool, reason: str).
    is_risky=True → bloquer les nouveaux BUY pour tous les tickers.
    """
    now_et = datetime.datetime.now(ET)
    today  = now_et.date()
    hour   = now_et.hour

    # CPI / PPI / PCE+GDP hardcodés — 08:30 ET, fenêtre [-1h, +2h]
    is_risky, reason = _check_830_events(now_et)
    if is_risky:
        return True, reason

    # FOMC hardcodé — 14:00 ET, fenêtre [-1h, +2h]
    is_risky, reason = _check_fomc_hardcoded(now_et)
    if is_risky:
        return True, reason

    # Finnhub — seulement les événements vraiment systémiques
    UTC = pytz.utc
    events = _fetch_events(today)
    for e in events:
        try:
            event_name = e.get('event', '')
            if not _is_systemic(event_name):
                continue
            event_utc  = UTC.localize(datetime.datetime.strptime(e['time'], '%Y-%m-%d %H:%M:%S'))
            event_et   = event_utc.astimezone(ET)
            hours_diff = (event_et - now_et).total_seconds() / 3600
            # Bloquer dans la fenêtre [-1h, +2h] autour de l'annonce
            if -1 <= hours_diff <= 2:
                return True, f"{event_name} ({event_et.strftime('%H:%M')} ET) — pas de nouveau BUY"
        except Exception:
            continue

    # NFP algorithmique — 1er vendredi du mois, 08:30 ET, fenêtre [-1h, +2h]
    if _is_nfp_day(today):
        nfp_et = now_et.replace(hour=8, minute=30, second=0, microsecond=0)
        hours_diff = (nfp_et - now_et).total_seconds() / 3600
        if -1 <= hours_diff <= 2:
            return True, f"NFP ({today}) — annonce 8h30 ET, pas de nouveau BUY"

    return False, ""
