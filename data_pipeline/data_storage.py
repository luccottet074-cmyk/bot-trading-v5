import os
import pandas as pd

# Calcul dynamique du dossier racine (1 niveau au-dessus car on est dans /data_pipeline)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")   # features, modèles, signaux — effacé à chaque run
RAW_DIR     = os.path.join(DATA_DIR, "raw")        # données brutes Alpaca — conservé entre runs
CSV_DIR     = os.path.join(DATA_DIR, "csv")

os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


def save_data(df, filename):
    """
    Sauvegarde un DataFrame :
    - En Parquet dans data/parquet/ (lecture machine rapide)
    - En CSV dans data/csv/ en fallback si pyarrow est absent
    """
    parquet_path = os.path.join(PARQUET_DIR, f"{filename}.parquet")
    csv_path     = os.path.join(CSV_DIR,     f"{filename}.csv")
    try:
        df.to_parquet(parquet_path, compression='snappy')
        return parquet_path
    except ImportError:
        df.to_csv(csv_path)
        return csv_path


def load_data(filename):
    """
    Charge un DataFrame :
    - Cherche d'abord dans data/parquet/
    - Fallback sur data/csv/ si le parquet est absent
    """
    parquet_path = os.path.join(PARQUET_DIR, f"{filename}.parquet")
    csv_path     = os.path.join(CSV_DIR,     f"{filename}.csv")

    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    return None


def file_exists(filename):
    """Vérifie si un fichier existe dans l'un des deux dossiers."""
    parquet_path = os.path.join(PARQUET_DIR, f"{filename}.parquet")
    csv_path     = os.path.join(CSV_DIR,     f"{filename}.csv")
    return os.path.exists(parquet_path) or os.path.exists(csv_path)
