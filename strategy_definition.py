# ==========================================
# 1. STRATEGY DEFINITION (Idea Generation)
# ==========================================
"""
Ce module définit la logique économique et les paramètres globaux de la stratégie.
Conformément au ML4T Workflow, tout commence par une hypothèse testable.

Hypothèse économique (Mean Reversion Intraday) : 
Sur de courtes périodes (1 minute à 1 heure), les prix des actifs très liquides 
tendent à surréagir aux micro-événements (bruit de marché). S'ils s'écartent trop 
de leur moyenne locale, ils finissent par y revenir ("mean reversion").

Nous allons capturer ce "bruit" via des features techniques (RSI, Volatilité) 
pour prédire le rendement futur sur les 10 à 60 prochaines minutes.
"""

STRATEGY = {
    "name": "mean_reversion_intraday",
    "target_horizon": "1h_forward_return",  # Ce qu'on cherche à prédire : prochaine barre 1h (aligné sur la fréquence de trading)
    "signal_type": "continuous_return",     # Prédiction d'un retour continu (régression)
    "universe": ["AAPL", "MSFT", "GOOGL", "COST"],  # Univers de trading
    "data_source": "ibkr",                 # Source de données principale (ibkr, yahoo, polygon...)
    "frequency": "1h",                     # Fréquence d'échantillonnage de la donnée
    "lookback_period": "1825d",            # 5 ans Alpaca IEX — ~9000 barres 1h, ~200+ trades/ticker
    "target_threshold": 0.001             # Seuil min de rendement pour Hausse/Baisse (0.1%). En dessous → Neutre (0)
}

if __name__ == "__main__":
    print(f"✅ Stratégie chargée : {STRATEGY['name']}")
    print(f"✅ Univers : {STRATEGY['universe']}")
    print(f"✅ Fréquence : {STRATEGY['frequency']}")
