import subprocess
import sys
import os
import shutil
from datetime import datetime


def _check_not_business_day():
    """Bloque l'exécution un jour ouvré (lundi-vendredi).

    run_all ne doit être lancé que le week-end : ça évite (1) le bruit de
    ré-entraînement (RFR/XGB non déterministes) sans nouvelle information
    utile en pleine semaine, et (2) le risque de données incomplètes/non
    finalisées côté Alpaca/IBKR si lancé un jour de bourse.
    Contournable avec --force pour du debug ponctuel.
    """
    if "--force" in sys.argv:
        print("⚠️  --force détecté : vérification jour ouvré ignorée.")
        return
    jours = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
    today = datetime.now()
    if today.weekday() < 5:  # 0=lundi ... 4=vendredi
        print(f"❌ run_all.py bloqué : on est {jours[today.weekday()]}, un jour ouvré.")
        print("   run_all ne doit être lancé que le week-end (samedi de préférence).")
        print("   Pour forcer l'exécution malgré tout : python run_all.py --force")
        sys.exit(1)


def clear_cache():
    """
    Nettoie les fichiers de données et le cache Python avant chaque run.
    - Vide data/parquet/ et data/csv/ (données du run précédent)
    - Supprime les dossiers __pycache__ (évite les conflits de module)
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Vide les dossiers de données (supprime les fichiers, pas les dossiers)
    #    On supprime les fichiers individuellement car Windows/OneDrive verrouille les dossiers
    data_dirs_to_clear = [
        os.path.join(root_dir, "data", "parquet"),
        os.path.join(root_dir, "data", "csv"),
    ]
    for d in data_dirs_to_clear:
        if os.path.exists(d):
            for f in os.listdir(d):
                file_path = os.path.join(d, f)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except PermissionError:
                    pass  # Fichier verrouillé, on ignore
        else:
            os.makedirs(d)

    # 2. Suppression des __pycache__ (évite que Python charge l'ancien module compilé)
    #    ignore_errors=True car VS Code verrouille parfois ces dossiers sur Windows
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                shutil.rmtree(os.path.join(dirpath, dirname), ignore_errors=True)

    print("🧹 Cache nettoyé (data/parquet, data/csv, __pycache__)")


def main():
    print("=" * 50)
    print("🚀 DÉMARRAGE DU PIPELINE ML4T (ARCHITECTURE HEDGE FUND)")
    print("=" * 50)

    # Garde-fou : pas de run_all en pleine semaine
    _check_not_business_day()

    # Nettoyage du cache avant chaque run (évite les fichiers orphelins et conflits de module)
    clear_cache()

    # Liste de tous nos modules dans l'ordre chronologique strict
    # Exécutés en tant que modules (-m) pour garantir que les imports fonctionnent depuis la racine
    modules = [
        "data_pipeline.pipeline",
        "features.alpha_factors",   # ← Orchestrateur central des Alpha Factors
        "features.ml_features",
        "features.factor_evaluation",
        "models.training",
        "models.shap_analysis",
        "models.walk_forward",
        "models.prediction",
        "features.ic_kelly",
        "backtesting.engine",
        "backtesting.performance"
    ]

    # S'assurer qu'on est bien à la racine du projet
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    for mod in modules:
        print(f"\n▶ {mod}")

        result = subprocess.run([sys.executable, "-m", mod])

        # Si un script plante on arrête tout par sécurité.
        if result.returncode != 0:
            print(f"\n❌ Erreur critique détectée dans {mod}. Arrêt immédiat du pipeline.")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS ! TOUS LES RAPPORTS SONT PRÊTS.")
    print("=" * 50)


if __name__ == "__main__":
    main()
