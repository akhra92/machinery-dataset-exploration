"""Per-clip signal statistics (time and frequency domain), quality flags, and how well a statistic separates groups."""
import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

# Scalar statistics compared between normal and faulty clips (averaged over channels)
STATS = {
    "rms": "RMS (signal energy)",
    "peak": "peak |x|",
    "crest_factor": "crest factor (peak / RMS); impacts raise it",
    "kurtosis": "kurtosis (0 = Gaussian); impacts raise it",
    "skewness": "skewness",
    "zero_crossings_per_s": "zero crossings per second",
    "spectral_centroid_hz": "spectral centroid (Hz)",
    "spectral_bandwidth_hz": "spectral bandwidth (Hz)",
    "spectral_rolloff_hz": "frequency below which 85% of the energy lies (Hz)",
    "dominant_freq_hz": "strongest frequency (Hz)",
    "spectral_flatness": "spectral flatness (1 = noise, 0 = tonal)",
}


def welch_psd(x: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density averaged over channels: (frequencies, psd)."""
    f, p = sps.welch(x, fs=sr, nperseg=min(4096, x.shape[-1]), axis=-1)
    return f, np.atleast_2d(p).mean(axis=0)


def signal_stats(x: np.ndarray, sr: float) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Statistics of one (channels, samples) clip, and its PSD (reused for the spectrum plots)."""
    x = np.atleast_2d(x).astype(np.float64)
    finite = bool(np.isfinite(x).all())
    if not finite:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=1))
    peak = np.abs(centered).max(axis=1)
    raw_peak = np.abs(x).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 0, peak / rms, 0.0)
        dc = np.where(rms > 0, np.abs(mean[:, 0]) / rms, 0.0)
    # clipping / saturation: consecutive samples stuck at the extreme value (a flat top). A clean waveform
    # touches its peak at isolated samples only.
    at = np.abs(x) >= raw_peak[:, None] * (1 - 1e-6)
    at_peak = np.where(raw_peak > 0, (at[:, 1:] & at[:, :-1]).mean(axis=1), 0.0)
    crossings = (np.diff(np.signbit(centered), axis=1)).sum(axis=1) / (x.shape[1] / sr)

    f, p = welch_psd(centered, sr)
    p = p + 1e-30
    total = p.sum()
    centroid = float((f * p).sum() / total)
    stats = {
        "duration_s": x.shape[1] / sr,
        "sample_rate": float(sr),
        "n_channels": x.shape[0],
        "rms": float(rms.mean()),
        "peak": float(peak.mean()),
        "crest_factor": float(crest.mean()),
        "kurtosis": float(np.nan_to_num(kurtosis(centered, axis=1)).mean()),
        "skewness": float(np.nan_to_num(skew(centered, axis=1)).mean()),
        "zero_crossings_per_s": float(crossings.mean()),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": float(np.sqrt(((f - centroid) ** 2 * p).sum() / total)),
        "spectral_rolloff_hz": float(f[min(np.searchsorted(np.cumsum(p), 0.85 * total), f.size - 1)]),
        "dominant_freq_hz": float(f[1:][np.argmax(p[1:])]) if f.size > 1 else 0.0,
        "spectral_flatness": float(np.exp(np.log(p).mean()) / p.mean()),
        # quality flags
        "has_nan_or_inf": not finite,
        "is_silent": bool(rms.max() < 1e-8),
        "dc_offset_ratio": float(dc.mean()),
        "clipped_fraction": float(at_peak.mean()),
    }
    return stats, (f, p)


def separation_auc(values, labels) -> float:
    """How well one statistic separates two groups on its own, ignoring direction:
    0.5 = not at all, 1.0 = perfectly (a threshold on this number alone would split them)."""
    values, labels = np.asarray(values, dtype=float), np.asarray(labels, dtype=int)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return float("nan")
    auc = roc_auc_score(labels, values)
    return float(max(auc, 1.0 - auc))
