import numpy as np
import pandas as pd

# ============================================================
# DENOISING FACTORS (Chapitre 5 - ML4T)
# ============================================================
# Les prix financiers sont du "signal + bruit". Les techniques de débruitage
# permettent d'isoler le signal sous-jacent (tendance vraie) du bruit aléatoire.
#
# Deux approches complémentaires :
#   - Kalman Filter   : Filtre temporel probabiliste (suit le signal en temps réel)
#   - Wavelet Denoise : Décomposition fréquentielle (élimine le bruit haute fréquence)
#
# IMPORTANT : Ces features agissent sur df['Close'].shift(1) pour rester Bias-Free.
# ============================================================


# ============================================================
# 1. KALMAN FILTER (implémentation numpy pur — sans pykalman)
# ============================================================
def _kalman_filter_1d(series, process_noise=1e-4, measurement_noise=1e-2):
    """
    Filtre de Kalman 1D implémenté from scratch en numpy.
    
    Modèle : marche aléatoire (le "vrai prix" se déplace lentement).
    - Q (process_noise)     : Confiance dans le modèle. Faible → lisse davantage.
    - R (measurement_noise) : Confiance dans la mesure. Élevé → lisse davantage.
    
    IMPORTANT : Travaille uniquement sur les valeurs non-NaN pour éviter la propagation.
    Les NaN (ex: première ligne du .shift(1)) sont préservés dans le résultat final.
    """
    # Prépare un résultat vide indexé sur la série originale
    result = pd.Series(np.nan, index=series.index, dtype=float)

    # On ne travaille que sur les lignes valides (non-NaN)
    valid = series.dropna()
    if len(valid) == 0:
        return result

    values = valid.to_numpy(dtype=float)
    n = len(values)

    x = np.zeros(n)
    p = np.zeros(n)
    x[0] = values[0]
    p[0] = 1.0

    for i in range(1, n):
        x_pred = x[i - 1]
        p_pred = p[i - 1] + process_noise
        K = p_pred / (p_pred + measurement_noise)
        x[i] = x_pred + K * (values[i] - x_pred)
        p[i] = (1 - K) * p_pred

    result.loc[valid.index] = x
    return result


def add_kalman_features(df, process_noise=1e-4, measurement_noise=1e-2):
    """
    Génère 2 features issues du Kalman Filter appliqué au prix :
    - `kalman_smooth`   : Le "vrai prix" estimé (signal filtré)
    - `kalman_residual` : Close - kalman_smooth (= le "bruit" isolé)
      Un résidu élevé signale une déviation anormale → mean-reversion imminente ?
    """
    c = df['Close'].shift(1)
    kalman = _kalman_filter_1d(c, process_noise, measurement_noise)

    df['kalman_smooth']   = kalman
    df['kalman_residual'] = c - kalman   # Écart entre prix réel et prix "filtré"
    return df


# ============================================================
# 2. WAVELET DENOISING
# ============================================================
def _wavelet_denoise_numpy(series, keep_ratio=0.1):
    """
    Débruitage par transformée de Fourier rapide (FFT) — fallback sans pywt.
    Travaille uniquement sur les valeurs non-NaN pour éviter la propagation.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) == 0:
        return result

    values = valid.to_numpy(dtype=float)
    n = len(values)
    fft_vals = np.fft.rfft(values)
    cutoff = max(1, int(len(fft_vals) * keep_ratio))
    fft_filtered = fft_vals.copy()
    fft_filtered[cutoff:] = 0
    denoised = np.fft.irfft(fft_filtered, n=n)

    result.loc[valid.index] = denoised[:len(valid)]
    return result


def _wavelet_denoise_pywt(series, wavelet='db4', level=1):
    """
    Débruitage par Discrete Wavelet Transform via pywt.
    Méthode soft-thresholding de Donoho & Johnstone (standard académique).
    Travaille uniquement sur les valeurs non-NaN pour éviter la propagation.
    """
    import pywt
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) == 0:
        return result

    values = valid.to_numpy(dtype=float).copy()  # .copy() requis : pywt refuse les tableaux read-only
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Seuillage universel (Donoho)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    denoised = pywt.waverec(coeffs_thresh, wavelet)
    result.loc[valid.index] = denoised[:len(valid)]  # tronque si pywt ajoute 1 sample
    return result


def add_wavelet_features(df, keep_ratio=0.1):
    """
    Génère 2 features issues du débruitage wavelet appliqué au prix :
    - `wavelet_smooth`   : Le prix débruité (composante basse fréquence = tendance)
    - `wavelet_residual` : Close - wavelet_smooth (= bruit haute fréquence isolé)
    """
    c = df['Close'].shift(1)

    try:
        smooth = _wavelet_denoise_pywt(c)
        method = "pywt (Haar DWT)"
    except ImportError:
        smooth = _wavelet_denoise_numpy(c, keep_ratio)
        method = "FFT (fallback numpy)"

    df['wavelet_smooth']   = smooth.values
    df['wavelet_residual'] = c.values - smooth.values
    return df


# ============================================================
# 3. EMA DIFFÉRENTIELLE (Denoising léger, toujours utile)
# ============================================================
def add_ema_diff(df, fast=8, slow=21):
    """
    Différence entre EMA rapide et EMA lente.
    Filtre le bruit en comparant deux niveaux de lissage.
    - ema_diff > 0 → Tendance haussière (fast > slow)
    - ema_diff < 0 → Tendance baissière (fast < slow)
    """
    c = df['Close'].shift(1)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    df['ema_diff']       = ema_fast - ema_slow
    df['ema_diff_pct']   = (ema_fast - ema_slow) / (ema_slow + 1e-9)  # Normalisé
    return df


def compute_all_denoising(df):
    """Applique toutes les techniques de débruitage d'un seul appel."""
    df = add_kalman_features(df)
    df = add_wavelet_features(df)
    df = add_ema_diff(df)
    return df
