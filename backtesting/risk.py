import numpy as np
import pandas as pd


def apply_stop_loss(df, stop_loss_pct=0.02, take_profit_pct=None):
    """
    Applique Stop Loss (SL) et Take Profit (TP) sur les positions actives.

    La taille de position est prise en compte : une position 0.5 capée à ±SL
    produit un retour brut de ±SL × 0.5, ce qui correspond à un mouvement
    de marché de ±SL.

    Paramètres :
        stop_loss_pct   : perte max autorisée sur le marché (défaut 2 %)
        take_profit_pct : gain cible sur le marché (défaut 1.5 × SL = 3 %)
    """
    if take_profit_pct is None:
        take_profit_pct = stop_loss_pct * 1.5

    data = df.copy()
    data['Stop_Loss_Triggered']   = False
    data['Take_Profit_Triggered'] = False

    in_pos = data['Position'] > 0

    sl_hit = in_pos & (data['Market_Return'] < -stop_loss_pct)
    tp_hit = in_pos & (data['Market_Return'] >  take_profit_pct) & ~sl_hit

    # Retour brut capé = ±seuil × taille de position
    data.loc[sl_hit, 'Strategy_Return_Gross'] = -stop_loss_pct   * data.loc[sl_hit, 'Position']
    data.loc[tp_hit, 'Strategy_Return_Gross'] =  take_profit_pct * data.loc[tp_hit, 'Position']

    data.loc[sl_hit, 'Stop_Loss_Triggered']   = True
    data.loc[tp_hit, 'Take_Profit_Triggered'] = True

    return data


def calculate_max_drawdown(portfolio_values):
    rolling_max = portfolio_values.cummax()
    drawdown    = (portfolio_values - rolling_max) / rolling_max
    return drawdown.min()
