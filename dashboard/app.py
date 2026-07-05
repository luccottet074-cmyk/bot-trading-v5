import json
import os
import sys
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

from strategy_definition import STRATEGY
from data_pipeline.data_storage import PARQUET_DIR
from backtesting.engine import INITIAL_CAPITAL

LOG_PATH        = os.path.join(BASE_DIR, 'data', 'execution_log.csv')
IC_PATH         = os.path.join(BASE_DIR, 'data', 'ic_tracker.csv')
PEAK_PATH       = os.path.join(BASE_DIR, 'data', 'peak_value.json')
KELLY_PATH      = os.path.join(PARQUET_DIR, 'kelly_weights.parquet')

SYMBOLS = STRATEGY["universe"]
FREQ    = STRATEGY.get("frequency", "1h")
PERIODS_PER_YEAR = {"1m": 252*6.5*60, "5m": 252*6.5*12, "15m": 252*6.5*4, "1h": 252*6.5, "1d": 252}
PPY = PERIODS_PER_YEAR.get(FREQ, 252 * 6.5)

DARK   = "#0e1117"
CARD   = "#1a1d2e"
GRID   = "#1e2130"
BLUE   = "#58a6ff"
GRAY   = "#8b949e"
GREEN  = "#3fb950"
ORANGE = "#f0883e"
RED    = "#f85149"

st.set_page_config(page_title="Trading Bot V3 — IBKR Dashboard", page_icon="🤖", layout="wide")

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}
    div[data-testid="metric-container"] {{
        background: {CARD}; border-radius: 8px; padding: 12px 16px;
        border: 1px solid {GRID};
    }}
    div[data-testid="stTabs"] button {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_risk_state():
    if not os.path.exists(PEAK_PATH):
        return None
    try:
        with open(PEAK_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_kelly():
    if not os.path.exists(KELLY_PATH):
        return None
    try:
        return pd.read_parquet(KELLY_PATH)
    except Exception:
        return None


@st.cache_data(ttl=300)
def compute_live_ic():
    from scipy.stats import spearmanr

    if not os.path.exists(IC_PATH):
        return None

    preds = pd.read_csv(IC_PATH, parse_dates=['timestamp_utc'])
    preds['timestamp_utc'] = pd.to_datetime(preds['timestamp_utc'], utc=True)
    if len(preds) < 5:
        return None

    ticker_results = {}
    for symbol in preds['symbol'].unique():
        sym_preds = preds[preds['symbol'] == symbol].copy().sort_values('timestamp_utc')
        price_path = os.path.join(PARQUET_DIR, f"{symbol}_1h.parquet")
        if not os.path.exists(price_path):
            continue

        prices = pd.read_parquet(price_path)
        prices.index = pd.to_datetime(prices.index, utc=True)
        prices['ret'] = prices['Close'].pct_change()

        actual_rets = []
        for ts in sym_preds['timestamp_utc']:
            future = prices[prices.index > ts]
            actual_rets.append(future['ret'].iloc[0] if len(future) > 0 else np.nan)

        sym_preds['actual_return'] = actual_rets
        sym_preds = sym_preds.dropna()
        if len(sym_preds) < 5:
            ticker_results[symbol] = {'ic': None, 'n': len(sym_preds), 'rolling': None}
            continue

        ic, pval = spearmanr(sym_preds['predicted_return'], sym_preds['actual_return'])

        window     = 20
        rolling_ic = []
        for i in range(len(sym_preds)):
            start = max(0, i - window + 1)
            chunk = sym_preds.iloc[start:i+1]
            if len(chunk) >= 5:
                r, _ = spearmanr(chunk['predicted_return'], chunk['actual_return'])
                rolling_ic.append({'ts': chunk['timestamp_utc'].iloc[-1], 'ic': r})

        ticker_results[symbol] = {
            'ic': float(ic), 'pval': float(pval), 'n': len(sym_preds),
            'rolling': pd.DataFrame(rolling_ic) if rolling_ic else None,
        }

    return ticker_results if ticker_results else None


LIVE_STATE_PATH = os.path.join(BASE_DIR, 'data', 'live_state.json')


def _update_ibkr_snapshot() -> bool:
    """Lance update_snapshot.py dans un subprocess pour rafraîchir live_state.json."""
    import subprocess
    script = os.path.join(BASE_DIR, 'execution', 'update_snapshot.py')
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR, capture_output=True, timeout=20,
    )
    return result.returncode == 0


@st.cache_data(ttl=60)
def load_live():
    try:
        if not os.path.exists(LIVE_STATE_PATH):
            return None
        with open(LIVE_STATE_PATH, encoding='utf-8-sig') as f:
            state = json.load(f)
        log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
        return {
            'account': {
                'portfolio_value': state['portfolio_value'],
                'cash':            state['cash'],
                'equity':          state.get('equity', 0),
            },
            'positions': state.get('positions', {}),
            'log':       log,
            'timestamp': state.get('timestamp', '—'),
        }
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_all():
    out = {}
    for sym in SYMBOLS:
        out[sym] = {
            "bt":  _parquet(f"{sym}_Backtest"),
            "met": _parquet(f"{sym}_Metrics"),
            "wf":  _parquet(f"{sym}_walk_forward"),
            "sig": _parquet(f"{sym}_Signals"),
        }
    out["panel"] = _parquet("Panel_Backtest")
    return out


def _parquet(name):
    p = os.path.join(PARQUET_DIR, f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def _plot_cfg(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=28, b=0),
        plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color="#c9d1d9", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, showgrid=True),
    )
    return fig


def panel_metrics(panel):
    ret   = panel["Panel_Return_Net"].dropna()
    final = panel["Portfolio_Value"].iloc[-1]
    bh    = panel["Buy_Hold_Value"].iloc[-1]
    r_pct = (final / INITIAL_CAPITAL - 1) * 100
    b_pct = (bh    / INITIAL_CAPITAL - 1) * 100
    sharpe  = (ret.mean() / ret.std()) * np.sqrt(PPY) if ret.std() > 0 else 0.0
    down    = ret[ret < 0]
    sortino = (ret.mean() / down.std()) * np.sqrt(PPY) if len(down) > 0 and down.std() > 0 else 0.0
    pv      = panel["Portfolio_Value"]
    max_dd  = ((pv - pv.cummax()) / pv.cummax()).min() * 100
    return r_pct, b_pct, sharpe, sortino, max_dd


def equity_fig(panel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=panel.index, y=panel["Portfolio_Value"],
                             name="ML Strategy", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=panel.index, y=panel["Buy_Hold_Value"],
                             name="Buy & Hold", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.update_yaxes(title_text="Capital ($)")
    return _plot_cfg(fig, height=300)


def ticker_fig(bt):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        subplot_titles=("Equity Curve", "Position"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Portfolio_Value"],
                             name="ML", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Buy_Hold_Value"],
                             name="B&H", line=dict(color=GRAY, width=1.5, dash="dot")), row=1, col=1)
    colors = [BLUE if p > 0 else GRID for p in bt["Position"]]
    fig.add_trace(go.Bar(x=bt.index, y=bt["Position"],
                         name="Position", marker_color=colors, opacity=0.8), row=2, col=1)
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=28, b=0),
        plot_bgcolor=DARK, paper_bgcolor=DARK, font=dict(color="#c9d1d9", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        yaxis2=dict(gridcolor=GRID, title="Pos.", range=[-0.1, 1.2]),
    )
    return fig


def wf_table(wf):
    score_col = "ic" if "ic" in wf.columns else "accuracy"
    df = wf[["fold", "debut", "fin", "model", score_col, "baseline", "delta", "beats"]].copy()
    if score_col == "ic":
        df[score_col] = df[score_col].map("{:+.3f}".format)
        df["baseline"] = df["baseline"].map("{:+.3f}".format)
        df["delta"]    = df["delta"].map("{:+.3f}".format)
        score_label = "IC"
    else:
        df[score_col] = (df[score_col] * 100).map("{:.1f}%".format)
        df["baseline"] = (df["baseline"] * 100).map("{:.1f}%".format)
        df["delta"]    = (df["delta"]    * 100).map("{:+.1f}%".format)
        score_label = "Accuracy"
    df["beats"] = df["beats"].map({True: "✅", False: "❌"})
    df.columns  = ["Fold", "Début", "Fin", "Modèle", score_label, "Naïf", "Delta", ""]
    return df


def _risk_tier(dd):
    """Couleur et label basés sur le DD (pour la barre visuelle des paliers)."""
    if dd <= -0.10: return 0.00, RED,    "🛑 FLAT — DD ≤ -10%"
    if dd <= -0.07: return 0.25, RED,    "🔴 ×0.25 — DD ≤ -7%"
    if dd <= -0.05: return 0.50, ORANGE, "🟠 ×0.50 — DD ≤ -5%"
    if dd <= -0.03: return 0.75, ORANGE, "🟡 ×0.75 — DD ≤ -3%"
    return 1.00, GREEN, "🟢 ×1.00 — Normal"


def _scale_label(scale):
    """Label et couleur basés sur le risk_scale réel du bot (récupération progressive)."""
    if scale <= 0.00: return RED,    "🛑 FLAT"
    if scale <= 0.25: return RED,    f"🔴 ×{scale:.2f}"
    if scale <= 0.50: return ORANGE, f"🟠 ×{scale:.2f}"
    if scale <= 0.75: return ORANGE, f"🟡 ×{scale:.2f}"
    return GREEN, "🟢 ×1.00 — Normal"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LAYOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

AUTO_REFRESH_SEC = 300  # 5 min


def _is_market_hours() -> bool:
    now = datetime.now(pytz.timezone('America/New_York'))
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= mins < 16 * 60


if _is_market_hours():
    st.markdown(
        f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SEC}">',
        unsafe_allow_html=True,
    )

col_h, col_btn = st.columns([8, 1])
with col_h:
    st.title("🤖 Trading Bot V3 — IBKR — ML4T Dashboard")
    st.caption(f"Stratégie : **{STRATEGY['name']}** | Univers : **{', '.join(SYMBOLS)}** "
               f"| Fréquence : **{FREQ}** | Lookback : **{STRATEGY['lookback_period']}**"
               f" | Ensemble : RFR + Ridge + XGB | Signal retention : N=2")
with col_btn:
    st.markdown("##")
    if st.button("🔄 Refresh", use_container_width=True):
        with st.spinner("IBKR…"):
            _update_ibkr_snapshot()
        st.cache_data.clear()
        st.rerun()
    if _is_market_hours():
        st.caption(f"⏱ auto {AUTO_REFRESH_SEC // 60}min")

data = load_all()

# â"€â"€ Live Alpaca â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("---")
st.markdown("### Live — IBKR Paper [V3]")

live = load_live()
if live is None:
    st.info("Connexion IBKR indisponible — IB Gateway non connecté")
else:
    acc = live['account']
    pos = live['positions']

    unreal_pl = sum(p['unrealized_pl'] for p in pos.values()) if pos else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio",       f"${acc['portfolio_value']:,.0f}")
    c2.metric("Cash",            f"${acc['cash']:,.0f}")
    c3.metric("Positions ouv.",  str(len(pos)))
    c4.metric("Unrealized P&L",  f"${unreal_pl:+,.2f}",
              delta_color="normal" if unreal_pl >= 0 else "inverse")

    st.caption(f"Snapshot IBKR : {live.get('timestamp', '—')} ET")

    if pos:
        st.markdown("**Positions ouvertes**")
        rows = []
        for sym, p in pos.items():
            rows.append({
                'Ticker':    sym,
                'Qté':       f"{p['qty']:.4f}",
                'Prix moy.': f"${p['avg_price']:.2f}",
                'Prix act.': f"${p['current_price']:.2f}",
                'Valeur':    f"${p['market_value']:,.2f}",
                'P&L $':     f"${p['unrealized_pl']:+.2f}",
                'P&L %':     f"{p['unrealized_plpc']:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Aucune position ouverte actuellement.")

    log = live['log']
    if not log.empty:
        st.markdown("**Dernières décisions du bot (20)**")
        display_log = log[log['symbol'] != 'PIPELINE'].tail(20).iloc[::-1].copy()
        display_log['timestamp'] = (
            pd.to_datetime(display_log['timestamp'])
            .dt.tz_localize('America/New_York')
            .dt.tz_convert('Europe/Paris')
            .dt.strftime('%Y-%m-%d %H:%M:%S')
        )
        action_icons = {'BUY': '🟢 BUY', 'CLOSE': '🔴 CLOSE', 'HOLD': '🔵 HOLD',
                        'FLAT': '⚪ FLAT', 'SKIP_NOTIONAL': '⚠️ SKIP',
                        'SL_CLOSE': '🛑 STOP LOSS', 'TP_CLOSE': '🎯 TAKE PROFIT'}
        display_log['action'] = display_log['action'].map(lambda x: action_icons.get(x, x))
        display_log['pred_return'] = pd.to_numeric(display_log['pred_return'], errors='coerce').map(
            lambda x: f"{x:+.3%}" if pd.notna(x) else '')
        display_log['effective_size'] = pd.to_numeric(display_log['effective_size'], errors='coerce').map(
            lambda x: f"{x:.0%}" if pd.notna(x) and x > 0 else '—')
        display_log['notional'] = pd.to_numeric(display_log['notional'], errors='coerce').map(
            lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else '—')
        st.dataframe(
            display_log[['timestamp', 'symbol', 'action', 'pred_return',
                          'effective_size', 'regime_ok', 'notional']].rename(columns={
                'timestamp': 'Heure', 'symbol': 'Ticker', 'action': 'Action',
                'pred_return': 'Pred. ret.', 'effective_size': 'Taille',
                'regime_ok': 'Régime', 'notional': 'Montant',
            }),
            use_container_width=True, hide_index=True,
        )

# â"€â"€ Portfolio Risk (DD progressif) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("---")
st.markdown("### Risque Portefeuille — Stop-loss progressif")

risk = load_risk_state()
kelly_df = load_kelly()

col_risk, col_kelly = st.columns([1, 1])

with col_risk:
    if risk is None:
        st.info("Pas encore de données de risque — le fichier `peak_value.json` sera créé au premier run live.")
    else:
        dd_pct   = risk.get('dd_pct', 0.0)
        peak_val = risk.get('peak', 0.0)
        updated  = risk.get('updated', '—')
        dd_frac  = dd_pct / 100.0

        # Scale réel du bot (récupération progressive) — priorité sur le calcul DD seul
        actual_scale = risk.get('risk_scale', None)
        if actual_scale is not None:
            scale = actual_scale
            tier_color, tier_label = _scale_label(actual_scale)
            dd_scale, _, _ = _risk_tier(dd_frac)
            recovering = actual_scale < dd_scale  # bot plus prudent que le DD seul
        else:
            scale, tier_color, tier_label = _risk_tier(dd_frac)
            recovering = False

        r1, r2, r3 = st.columns(3)
        r1.metric("Peak value",      f"${peak_val:,.0f}")
        r2.metric("Drawdown actuel", f"{dd_pct:+.2f}%",
                  delta_color="inverse" if dd_pct < 0 else "off")
        r3.metric("Facteur risque",  f"×{scale:.2f}")

        label_str = f"**Palier actif :** {tier_label}"
        if recovering:
            label_str += "  *(récupération progressive)*"
        st.markdown(label_str)
        st.caption(f"Mis à jour : {updated} (heure ET)")

        # Barre visuelle des paliers
        paliers = [
            ("Normal",  0.0,  -3.0,  GREEN),
            ("×0.75",  -3.0,  -5.0,  "#d2a800"),
            ("×0.50",  -5.0,  -7.0,  ORANGE),
            ("×0.25",  -7.0, -10.0,  RED),
            ("FLAT",  -10.0, -15.0,  "#8b0000"),
        ]
        fig_dd = go.Figure()
        for label, start, end, color in paliers:
            active = (dd_pct <= start) and (dd_pct > end)
            fig_dd.add_trace(go.Bar(
                x=[label], y=[abs(end - start)],
                base=[abs(start)],
                marker_color=color,
                opacity=1.0 if active else 0.3,
                name=label,
                showlegend=False,
                text=[f"<b>{label}</b>" if active else label],
                textposition="inside",
            ))
        fig_dd.add_vline(x=dd_pct, line_dash='dash', line_color='white', line_width=2,
                         annotation_text=f"DD={dd_pct:.1f}%", annotation_position="top right")
        fig_dd.update_layout(
            height=160, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor=DARK, paper_bgcolor=DARK,
            font=dict(color="#c9d1d9", size=11),
            barmode='stack',
            xaxis=dict(showgrid=False),
            yaxis=dict(title="DD (%)", gridcolor=GRID, tickformat=".0f"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with col_kelly:
    st.markdown("**Kelly Weights (IC-proportionnels)**")
    if kelly_df is None:
        st.info("kelly_weights.parquet introuvable — lancer run_all.py d'abord.")
    else:
        eq_weight = 1.0 / len(kelly_df)
        rows_k = []
        for _, row in kelly_df.iterrows():
            diff_pct = (row['kelly_weight'] / eq_weight - 1) * 100
            rows_k.append({
                'Ticker':      row['symbol'],
                'IC':          f"{row['ic']:+.4f}",
                'Poids Kelly': f"{row['kelly_weight']:.3f}×",
                'vs Equal-W':  f"{diff_pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows_k), use_container_width=True, hide_index=True)

        # Graphique bar comparatif
        fig_k = go.Figure()
        colors_k = [BLUE, ORANGE, GREEN, '#d2a8ff']
        fig_k.add_trace(go.Bar(
            x=kelly_df['symbol'],
            y=kelly_df['kelly_weight'],
            marker_color=colors_k[:len(kelly_df)],
            name="Kelly",
        ))
        fig_k.add_hline(y=eq_weight, line_dash='dot', line_color=GRAY, line_width=1.5,
                        annotation_text="Equal-weight", annotation_position="right")
        fig_k.update_yaxes(title_text="Poids")
        st.plotly_chart(_plot_cfg(fig_k, height=200), use_container_width=True)

# â"€â"€ IC Live â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("---")
st.markdown("### IC Live — Santé du modèle")

ic_data = compute_live_ic()
MIN_OBS = 20

if ic_data is None:
    st.info(f"Pas encore assez de données — l'IC sera calculable après ~{MIN_OBS} observations par ticker (~{MIN_OBS} heures de marché).")
else:
    cols = st.columns(len(ic_data))
    for col, (sym, res) in zip(cols, ic_data.items()):
        if res['ic'] is None:
            col.metric(sym, "—", f"{res['n']} obs. (min {MIN_OBS})")
        else:
            ic_val = res['ic']
            status = "✅" if ic_val > 0.05 else "⚠️" if ic_val > 0 else "❌"
            col.metric(f"{status} {sym}", f"IC {ic_val:+.3f}",
                       f"n={res['n']} | p={res['pval']:.2f}")

    rolling_frames = {s: r['rolling'] for s, r in ic_data.items() if r.get('rolling') is not None}
    if rolling_frames:
        fig = go.Figure()
        colors_map = {'AAPL': BLUE, 'MSFT': ORANGE, 'GOOGL': GREEN, 'COST': '#d2a8ff'}
        for sym, df_r in rolling_frames.items():
            fig.add_trace(go.Scatter(
                x=df_r['ts'], y=df_r['ic'], name=sym,
                line=dict(color=colors_map.get(sym, GRAY), width=2),
            ))
        fig.add_hline(y=0, line_dash='dash', line_color=GRAY, line_width=1)
        fig.add_hline(y=0.05, line_dash='dot', line_color=GREEN, line_width=1,
                      annotation_text="seuil positif", annotation_position="right")
        fig.update_yaxes(title_text="IC (rolling 20)")
        st.plotly_chart(_plot_cfg(fig, height=260), use_container_width=True)
        st.caption("IC > 0 : modèle prédictif  |  IC > 0.05 : signal solide  |  IC < 0 : modèle à re-entraîner")

# â"€â"€ Panel KPIs â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("---")
st.markdown("### Portefeuille Panel")

panel = data.get("panel")
if panel is not None:
    _, _, sharpe, sortino, max_dd = panel_metrics(panel)
    ind_metrics = [data[sym]["met"] for sym in SYMBOLS if data[sym]["met"] is not None]
    if ind_metrics:
        avg_ml_pct = sum(m["Return_%"].iloc[0]  for m in ind_metrics) / len(ind_metrics)
        avg_bh_pct = sum(m["BuyHold_%"].iloc[0] for m in ind_metrics) / len(ind_metrics)
    else:
        avg_ml_pct, avg_bh_pct = 0.0, 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Panel ML",  f"{avg_ml_pct:+.1f}%")
    c2.metric("B&H Panel", f"{avg_bh_pct:+.1f}%", delta=f"{avg_ml_pct - avg_bh_pct:+.1f}% vs B&H", delta_color="off")
    c3.metric("Sharpe",    f"{sharpe:.2f}")
    c4.metric("Sortino",   f"{sortino:.2f}")
    c5.metric("Max DD",    f"{max_dd:.1f}%")
    st.plotly_chart(equity_fig(panel), use_container_width=True)
else:
    st.warning("Panel_Backtest.parquet introuvable — lancez run_all.py")

# â"€â"€ Per-ticker tabs â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("---")
st.markdown("### Par Ticker")

tabs = st.tabs([f"  {s}  " for s in SYMBOLS])

for i, sym in enumerate(SYMBOLS):
    with tabs[i]:
        bt  = data[sym]["bt"]
        met = data[sym]["met"]
        wf  = data[sym]["wf"]
        sig = data[sym]["sig"]

        if met is not None:
            row = met.iloc[0]
            ic_str = f"{row['IC']:+.3f} (p={row['IC_pvalue']:.2f})" if row["IC"] is not None else "—"
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("ML Return",  f"{row['Return_%']:+.1f}%")
            m2.metric("B&H Return", f"{row['BuyHold_%']:+.1f}%")
            m3.metric("Sharpe",     f"{row['Sharpe']:.2f}")
            m4.metric("Sortino",    f"{row['Sortino']:.2f}")
            m5.metric("Max DD",     f"{row['Max_Drawdown_%']:.1f}%")
            m6.metric("Hit Ratio",  f"{row['Hit_Ratio_%']:.0f}%")
            m7.metric("IC",         ic_str)

        if bt is not None:
            st.plotly_chart(ticker_fig(bt), use_container_width=True)

        col_wf, col_sig = st.columns([3, 2])

        with col_wf:
            if wf is not None:
                n_wins  = int(wf["beats"].sum())
                verdict = "✅ ROBUSTE" if n_wins >= 4 else "⚠️ INSTABLE" if n_wins >= 3 else "âŒ FRAGILE"
                st.markdown(f"**Walk-Forward** — {verdict} ({n_wins}/5)")
                st.dataframe(wf_table(wf), use_container_width=True, hide_index=True)

        with col_sig:
            if sig is not None:
                st.markdown("**Derniers signaux (10)**")
                score_col = "Predicted_Return" if "Predicted_Return" in sig.columns else "Prob_Up"
                display = sig[["Signal", score_col]].copy()
                if bt is not None and "Regime" in bt.columns:
                    display = display.join(bt[["Regime"]], how="left")
                    display["Regime"] = display["Regime"].map({1: "✅ Normal", 0: "⛔ Bloqué"})
                display["Signal"] = display["Signal"].map({1: "🟢 Long", -1: "🔴 Short", 0: "⚪ Neutre"})
                display[score_col] = (display[score_col] * 100).map("{:+.3f}%".format)
                st.dataframe(display.tail(10).iloc[::-1], use_container_width=True)

st.markdown("---")
st.caption(
    f"Données : `{PARQUET_DIR}`  |  "
    "Ensemble : RFR+Ridge+XGB (IC-pondéré)  |  "
    "Signal retention : N=2  |  "
    "SL/TP : AAPL/COST 2%/3% Â· MSFT 2%/4% Â· GOOGL 2%/5%  |  "
    "Frais+slippage : 0.15%"
)

