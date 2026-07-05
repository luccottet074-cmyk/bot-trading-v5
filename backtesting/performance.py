import os
import sys
import pandas as pd
import numpy as np

# Ajouter la racine au sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategy_definition import STRATEGY
from data_pipeline.data_storage import DATA_DIR, PARQUET_DIR, load_data, save_data
from backtesting.risk import calculate_max_drawdown
from backtesting.engine import INITIAL_CAPITAL

# Nombre de variantes stratégiques testées sur ces données (seuils, fenêtres MA,
# sizing, N_HOLD_BARS, combinaisons de features…). Ajuster honnêtement à chaque
# cycle d'optimisation — sous-estimer gonfle artificiellement la DSR.
N_TRIALS = 20


def deflated_sharpe_ratio(returns, n_trials=N_TRIALS, periods_per_year=252 * 6.5):
    """
    Deflated Sharpe Ratio — Bailey & López de Prado (2014).
    Corrige le SR pour : multiple testing, non-normalité et longueur d'échantillon.

    Formule :
      DSR = Φ( (SR̂ − SR₀) · √(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) )

      SR̂  = Sharpe par barre (non annualisé)
      T   = nb d'observations
      SR₀ = E[max SR sous H₀] avec n_trials essais — seuil de déflation
      γ₃  = skewness des rendements
      γ₄  = kurtosis excess des rendements (Fisher)

    Retourne (dsr_prob, sr_deflated) :
      dsr_prob    : P[vrai SR > 0] après correction pour le multiple testing
      sr_deflated : SR₀ annualisé (benchmark de déflation affiché)
    """
    import math
    from scipy.stats import norm as _norm

    ret = np.asarray(returns.dropna() if hasattr(returns, 'dropna') else returns)
    T = len(ret)
    if T < 5:
        return (np.nan, np.nan)

    mu, sigma = ret.mean(), ret.std(ddof=1)
    if sigma == 0:
        return (np.nan, np.nan)

    sr_hat = mu / sigma  # SR par barre (non annualisé)

    gamma3 = float(pd.Series(ret).skew())      # skewness
    gamma4 = float(pd.Series(ret).kurtosis())  # kurtosis excess (Fisher)

    # E[max de n_trials normales std] — approx. constante d'Euler-Mascheroni
    euler_gamma = 0.5772156649
    z1 = _norm.ppf(1 - 1 / max(n_trials, 2))
    z2 = _norm.ppf(1 - 1 / (max(n_trials, 2) * math.e))
    e_max = (1 - euler_gamma) * z1 + euler_gamma * z2

    # SR₀ non annualisé : seuil ajusté pour la longueur d'échantillon
    sr0 = e_max / np.sqrt(T - 1)

    # Dénominateur : correction pour non-normalité de la distribution
    denom_sq = 1 - gamma3 * sr_hat + ((gamma4 - 1) / 4) * sr_hat ** 2
    if denom_sq <= 0:
        denom_sq = 1e-9

    z_stat = (sr_hat - sr0) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    dsr_prob = float(_norm.cdf(z_stat))

    sr_deflated = sr0 * np.sqrt(periods_per_year)  # SR₀ annualisé pour affichage
    return (dsr_prob, sr_deflated)


def evaluate_performance(symbol):

    df = load_data(f"{symbol}_Backtest")
    if df is None:
        print(f"⚠️ Fichier de backtest introuvable pour {symbol}.")
        return

    # ==========================================
    # 1. Rendements de base
    # ==========================================
    initial_cap = INITIAL_CAPITAL
    final_cap   = df['Portfolio_Value'].iloc[-1]
    total_return_pct = (final_cap / initial_cap - 1) * 100

    bh_cap = df['Buy_Hold_Value'].iloc[-1]
    bh_return_pct = (bh_cap / initial_cap - 1) * 100

    # Annualisation selon la fréquence de trading
    FREQ = STRATEGY.get('frequency', '1m')
    ANNUALIZATION_MAP = {
        '1m':  252 * 6.5 * 60,       # ~98 280 périodes/an
        '5m':  252 * 6.5 * 12,       # ~19 656 périodes/an
        '15m': 252 * 6.5 * 4,        # ~6 552 périodes/an
        '1h':  252 * 6.5,            # ~1 638 périodes/an
        '1d':  252,
    }
    PERIODS_PER_YEAR = ANNUALIZATION_MAP.get(FREQ, 252 * 6.5 * 60)

    returns = df['Strategy_Return_Net'].dropna()

    # ==========================================
    # 2. Sharpe Ratio (rendement ajusté risque symétrique)
    # ==========================================
    mean_return = returns.mean()
    std_return  = returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(PERIODS_PER_YEAR) if std_return > 0 else 0.0

    # ==========================================
    # 3. Sortino Ratio (pénalise uniquement la volatilité négative)
    # ==========================================
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (mean_return / downside_std) * np.sqrt(PERIODS_PER_YEAR) if downside_std > 0 else 0.0

    # ==========================================
    # 4. Max Drawdown
    # ==========================================
    max_drawdown_pct = calculate_max_drawdown(df['Portfolio_Value']) * 100

    # ==========================================
    # 5. Hit Ratio (% de périodes gagnantes en position)
    # ==========================================
    trades_only = df[df['Position'] > 0]['Strategy_Return_Net']
    if len(trades_only) > 0:
        winning_trades = len(trades_only[trades_only > 0])
        hit_ratio = (winning_trades / len(trades_only)) * 100
    else:
        hit_ratio = 0.0

    # ==========================================
    # 6. Turnover (fréquence de rotation du portefeuille)
    # Mesure à quel point le robot "trade beaucoup" (coût caché)
    # Turnover = nb de changements de position / nb total de périodes
    # ==========================================
    total_trades = int(np.abs(df['Trade']).sum())
    turnover_pct = (total_trades / len(df)) * 100

    # ==========================================
    # 7. Information Coefficient (IC)
    # Corrélation de Spearman entre le signal prédit et le rendement réel suivant.
    # IC proche de 0 = signal nul. IC > 0.05 est considéré bon en pratique.
    # ==========================================
    try:
        from scipy.stats import spearmanr
        if 'Signal' in df.columns and 'Market_Return' in df.columns:
            # On compare le signal d'aujourd'hui avec le rendement du lendemain
            ic_corr, ic_pvalue = spearmanr(
                df['Signal'].iloc[:-1],
                df['Market_Return'].shift(-1).dropna()
            )
            ic_value  = round(ic_corr, 4)
            ic_pvalue = round(ic_pvalue, 4)
        else:
            ic_value  = None
            ic_pvalue = None
    except ImportError:
        ic_value  = None
        ic_pvalue = None

    # Diagnostic du filtre de régime : compare E[r|bloqué] vs E[r|tradé]
    regime_diag = ""
    if 'Regime' in df.columns and 'Signal' in df.columns:
        signal_lag   = df['Signal'].shift(1).fillna(0)
        blocked_mask = (signal_lag == 1) & (df['Regime'] == 0)
        traded_mask  = df['Position'] > 0
        if blocked_mask.sum() > 0 and traded_mask.sum() > 0:
            e_blocked = df.loc[blocked_mask, 'Market_Return'].mean() * 100
            e_traded  = df.loc[traded_mask,  'Market_Return'].mean() * 100
            verdict   = "✅ filtre utile" if e_blocked <= e_traded else "⚠️ filtre coupe bons trades"
            regime_diag = f"  |  Bloqué:{e_blocked:+.3f}% Tradé:{e_traded:+.3f}% {verdict}"

    n_long_bars  = len(trades_only)
    n_roundtrips = total_trades // 2
    ic_str = f"  IC={ic_value:+.3f}(p={ic_pvalue:.2f})" if ic_value is not None else ""
    print(f"  {symbol:<6}  ML:{total_return_pct:>+6.1f}%  B&H:{bh_return_pct:>+6.1f}%  "
          f"Sharpe:{sharpe_ratio:>5.2f}  DD:{max_drawdown_pct:>+5.1f}%  "
          f"Hit:{hit_ratio:.0f}%  Trades:{n_roundtrips}({n_long_bars}h){ic_str}{regime_diag}")

    # ==========================================
    # 9. Sauvegarde des métriques
    # ==========================================
    metrics = pd.DataFrame({
        'Symbol':         [symbol],
        'Return_%':       [round(total_return_pct, 4)],
        'BuyHold_%':      [round(bh_return_pct, 4)],
        'Sharpe':         [round(sharpe_ratio, 4)],
        'Sortino':        [round(sortino_ratio, 4)],
        'Max_Drawdown_%': [round(max_drawdown_pct, 4)],
        'Hit_Ratio_%':    [round(hit_ratio, 4)],
        'Turnover_%':     [round(turnover_pct, 4)],
        'Trades':         [total_trades],
        'IC':             [ic_value],
        'IC_pvalue':      [ic_pvalue],
    })

    save_data(metrics, f"{symbol}_Metrics")


def _save_live_history():
    """Appende un résumé live hebdomadaire dans logs/live_history.csv (jamais écrasé)."""
    import csv
    from datetime import date

    log_path = os.path.join(BASE_DIR, "data", "execution_log.csv")
    if not os.path.exists(log_path):
        return

    df = pd.read_csv(log_path, parse_dates=['timestamp'])
    pipeline = df[df['symbol'] == 'PIPELINE'].copy()
    if pipeline.empty:
        return

    current_value = pipeline.iloc[-1]['portfolio_value']

    # Détection du démarrage production : dernier saut de portfolio_value > 50%
    pv = pipeline['portfolio_value'].values
    live_start_idx = 0
    for i in range(1, len(pv)):
        if pv[i] > pv[i - 1] * 1.5:
            live_start_idx = i
    live_start_value = pv[live_start_idx]
    total_return_pct = (current_value / live_start_value - 1) * 100 if live_start_value > 0 else 0.0

    # Semaine = 7 derniers jours calendaires
    cutoff = pipeline.iloc[-1]['timestamp'] - pd.Timedelta(days=7)
    week_pipeline = pipeline[pipeline['timestamp'] >= cutoff]
    week_start_value = week_pipeline.iloc[0]['portfolio_value'] if not week_pipeline.empty else current_value
    week_return_pct = (current_value / week_start_value - 1) * 100 if week_start_value > 0 else 0.0

    # Comptage actions sur la semaine
    week_all = df[df['timestamp'] >= cutoff]
    n_buys          = int((week_all['action'] == 'BUY').sum())
    n_closes        = int(week_all['action'].isin(['CLOSE', 'EOD_CLOSE']).sum())
    n_sl            = int((week_all['action'] == 'SL_CLOSE').sum())
    n_order_failed  = int(week_all['action'].str.startswith('ORDER_FAILED', na=False).sum())
    n_skip_retained = int((week_all['action'] == 'SKIP_RETAINED').sum())
    n_skip_macro    = int((week_all['action'] == 'SKIP_MACRO').sum())

    row = {
        'date':                   date.today().isoformat(),
        'portfolio_value':        round(current_value, 2),
        'week_return_pct':        round(week_return_pct, 2),
        'total_live_return_pct':  round(total_return_pct, 2),
        'n_buys':                 n_buys,
        'n_closes':               n_closes,
        'n_sl':                   n_sl,
        'n_order_failed':         n_order_failed,
        'n_skip_retained':        n_skip_retained,
        'n_skip_macro':           n_skip_macro,
    }

    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    history_path = os.path.join(logs_dir, "live_history.csv")

    fieldnames = ['date', 'portfolio_value', 'week_return_pct', 'total_live_return_pct',
                  'n_buys', 'n_closes', 'n_sl', 'n_order_failed', 'n_skip_retained', 'n_skip_macro']

    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        if row['date'] in existing['date'].values:
            print(f"  📈 Live history déjà enregistré pour {row['date']} — ignoré")
            return
        # Migration : réécrire si le header ne correspond pas aux fieldnames actuels
        if list(existing.columns) != fieldnames:
            for col in fieldnames:
                if col not in existing.columns:
                    existing[col] = 0
            existing = existing[fieldnames]
            existing.to_csv(history_path, index=False, encoding='utf-8')

    write_header = not os.path.exists(history_path)
    with open(history_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  📈 Live history → logs/live_history.csv  "
          f"Semaine:{week_return_pct:+.2f}%  Total:{total_return_pct:+.2f}%  "
          f"Trades:{n_buys}B/{n_closes}C  SL:{n_sl}  Fails:{n_order_failed}")


def _save_model_history(symbols, avg_ml_pct, avg_bh_pct, sharpe, sortino, max_dd,
                        dsr_prob=None, sr_deflated=None):
    """Appende les métriques du run courant dans logs/model_history.csv (jamais écrasé)."""
    import csv
    from datetime import date

    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    history_path = os.path.join(logs_dir, "model_history.csv")

    row = {
        "date":              date.today().isoformat(),
        "panel_ml_pct":      round(avg_ml_pct, 2),
        "panel_bh_pct":      round(avg_bh_pct, 2),
        "panel_sharpe":      round(sharpe, 2),
        "panel_sortino":     round(sortino, 2),
        "panel_max_dd":      round(max_dd, 2),
        "panel_dsr":         round(dsr_prob, 4) if dsr_prob is not None and not np.isnan(dsr_prob) else "",
        "panel_sr_deflated": round(sr_deflated, 4) if sr_deflated is not None and not np.isnan(sr_deflated) else "",
    }

    for symbol in symbols:
        m_df = load_data(f"{symbol}_Metrics")
        if m_df is not None and len(m_df) > 0:
            row[f"{symbol}_ml_pct"] = round(float(m_df["Return_%"].iloc[0]), 2)
            row[f"{symbol}_bh_pct"] = round(float(m_df["BuyHold_%"].iloc[0]), 2)
            row[f"{symbol}_sharpe"] = round(float(m_df["Sharpe"].iloc[0]), 2)
            ic_val = m_df["IC"].iloc[0]
            row[f"{symbol}_ic_oos"] = round(float(ic_val), 4) if ic_val is not None else ""

        wf_df = load_data(f"{symbol}_walk_forward")
        if wf_df is not None and len(wf_df) > 0:
            row[f"{symbol}_wf_ic"]    = round(float(wf_df["ic"].mean()), 4)
            row[f"{symbol}_wf_folds"] = int(wf_df["beats"].sum())

    fieldnames = ["date", "panel_ml_pct", "panel_bh_pct", "panel_sharpe", "panel_sortino",
                  "panel_max_dd", "panel_dsr", "panel_sr_deflated"]
    for s in symbols:
        fieldnames += [f"{s}_ml_pct", f"{s}_bh_pct", f"{s}_sharpe", f"{s}_ic_oos", f"{s}_wf_ic", f"{s}_wf_folds"]

    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        if row["date"] in existing["date"].values:
            print(f"  📊 Model history déjà enregistré pour {row['date']} — ignoré")
            return
        # Migration auto si des colonnes manquent (ex: ajout panel_dsr/panel_sr_deflated)
        if list(existing.columns) != fieldnames:
            for col in fieldnames:
                if col not in existing.columns:
                    existing[col] = ""
            existing = existing[fieldnames]
            existing.to_csv(history_path, index=False, encoding='utf-8')

    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"  📊 Historique sauvegardé → logs/model_history.csv")


def evaluate_panel_performance(symbols):
    """
    Évalue chaque ticker individuellement, puis affiche un résumé comparatif
    et les métriques du portefeuille panel combiné (equity curve agrégée).
    """
    all_metrics = []
    print(f"\n{'Ticker':<8}  {'ML':>7}  {'B&H':>7}  {'Sharpe':>7}  {'DD':>6}  {'Hit':>4}  {'Trades':>12}  IC")
    print("  " + "-"*72)
    for symbol in symbols:
        evaluate_performance(symbol)
        m_df = load_data(f"{symbol}_Metrics")
        if m_df is not None:
            all_metrics.append(m_df)

    # --- Métriques du portefeuille panel combiné ---
    panel_path = os.path.join(PARQUET_DIR, "Panel_Backtest.parquet")
    if not os.path.exists(panel_path):
        print("\n⚠️ Panel_Backtest.parquet introuvable — lancez backtesting/engine.py d'abord.")
        return

    panel = pd.read_parquet(panel_path)

    FREQ = STRATEGY.get('frequency', '1h')
    ANNUALIZATION_MAP = {
        '1m':  252 * 6.5 * 60,
        '5m':  252 * 6.5 * 12,
        '15m': 252 * 6.5 * 4,
        '1h':  252 * 6.5,
        '1d':  252,
    }
    periods_per_year = ANNUALIZATION_MAP.get(FREQ, 252 * 6.5)

    ret = panel['Panel_Return_Net'].dropna()
    sharpe  = (ret.mean() / ret.std()) * np.sqrt(periods_per_year) if ret.std() > 0 else 0.0
    down    = ret[ret < 0]
    sortino = (ret.mean() / down.std()) * np.sqrt(periods_per_year) if len(down) > 0 else 0.0
    max_dd  = calculate_max_drawdown(panel['Portfolio_Value']) * 100

    bh_val  = panel['Buy_Hold_Value'].iloc[-1]

    if all_metrics:
        avg_ml_pct = sum(m['Return_%'].iloc[0] for m in all_metrics) / len(all_metrics)
        avg_bh_pct = sum(m['BuyHold_%'].iloc[0] for m in all_metrics) / len(all_metrics)
    else:
        avg_ml_pct = (panel['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
        avg_bh_pct = (bh_val / INITIAL_CAPITAL - 1) * 100

    dsr_prob, sr_deflated = deflated_sharpe_ratio(ret, n_trials=N_TRIALS, periods_per_year=periods_per_year)

    print("\n" + "="*62)
    print(f"{'PORTEFEUILLE PANEL (allocation égale)':^62}")
    print("="*62)
    print(f"  ML : {avg_ml_pct:+.1f}%   B&H : {avg_bh_pct:+.1f}%")
    print(f"  Sharpe : {sharpe:.2f}   Sortino : {sortino:.2f}   Max DD : {max_dd:.1f}%")
    if not np.isnan(dsr_prob):
        print(f"  DSR : P[SR>0]={dsr_prob:.1%}  (seuil={sr_deflated:.2f}, {N_TRIALS} essais)")
    print("="*62 + "\n")

    _save_live_history()
    _save_model_history(symbols, avg_ml_pct, avg_bh_pct, sharpe, sortino, max_dd, dsr_prob, sr_deflated)


if __name__ == "__main__":
    evaluate_panel_performance(STRATEGY["universe"])
