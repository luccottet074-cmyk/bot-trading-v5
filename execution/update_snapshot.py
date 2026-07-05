"""
Connecte à IBKR en clientId=10 (lecture seule, distinct du pipeline clientId=1),
récupère l'état du compte et des positions, puis met à jour live_state.json.
Appelé par le dashboard pour un refresh à la demande.
"""
import os
import sys
import json
from datetime import datetime

import pytz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))

LIVE_STATE_PATH = os.path.join(BASE_DIR, 'data', 'live_state.json')


def _safe_price(ticker, fallback):
    for p in (ticker.last, ticker.close, ticker.bid, ticker.ask):
        if p and p > 0:
            return float(p)
    return float(fallback)


def fetch_and_save():
    import random
    from ib_insync import IB

    ib = IB()
    client_id = random.randint(20, 99)
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=client_id, timeout=12, readonly=True)

    try:
        # NetLiquidation = temps réel côté serveur IBKR (pas besoin d'abonnement)
        all_vals = [(v.tag, v.currency, v.value) for v in ib.accountValues()
                    if v.tag in ('NetLiquidation', 'TotalCashValue', 'GrossPositionValue')]
        base_ccy = 'USD'
        for tag, ccy, val in all_vals:
            if tag == 'NetLiquidation' and float(val) > 0:
                base_ccy = ccy
                break
        vals = {v.tag: v.value for v in ib.accountValues() if v.currency == base_ccy}
        account = {
            'portfolio_value': float(vals.get('NetLiquidation', 0)),
            'cash':            float(vals.get('TotalCashValue', 0)),
            'equity':          float(vals.get('GrossPositionValue', 0)),
        }

        # Positions via portfolio() — unrealizedPNL calculé par IBKR (différé ~20s sans abonnement)
        positions = {}
        for item in ib.portfolio():
            if item.contract.secType != 'STK':
                continue
            sym       = item.contract.symbol
            qty       = float(item.position)
            avg_price = float(item.averageCost)
            cur_price = float(item.marketPrice) if item.marketPrice and item.marketPrice > 0 else avg_price
            positions[sym] = {
                'qty':             qty,
                'avg_price':       avg_price,
                'current_price':   cur_price,
                'market_value':    float(item.marketValue),
                'unrealized_pl':   float(item.unrealizedPNL),
                'unrealized_plpc': (cur_price / avg_price - 1) * 100 if avg_price else 0,
            }
    finally:
        ib.disconnect()

    ts = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M')
    state = {
        'portfolio_value': account['portfolio_value'],
        'cash':            account['cash'],
        'equity':          account['equity'],
        'positions':       positions,
        'timestamp':       ts,
    }

    with open(LIVE_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print(f"✅ live_state.json mis à jour — {ts} ET  "
          f"ptf=${account['portfolio_value']:,.0f}  positions={list(positions.keys())}")
    return True


if __name__ == '__main__':
    try:
        fetch_and_save()
    except Exception as e:
        print(f"❌ Échec : {e}")
        sys.exit(1)
