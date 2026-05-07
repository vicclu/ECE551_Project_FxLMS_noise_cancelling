import numpy as np
from pathlib import Path

def find_wav_file(filename: str) -> Path:
    """Find a WAV file inside the local sounds/ folder."""
    sounds_dir = Path("sounds")
    wav_path = sounds_dir / filename

    if wav_path.exists():
        return wav_path

    raise FileNotFoundError(f"Could not find WAV file: {wav_path}")

def past_vector(signal, n, L):
    vec = np.zeros(L)
    for k in range(L):
        if n - k >= 0:
            vec[k] = signal[n - k]
    return vec


def moving_average(x, window):
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def fir_filter(x, h):
    return np.convolve(x, h, mode="full")[:len(x)]


def normalize(x, eps=1e-12):
    return x / (np.sqrt(np.mean(x**2)) + eps)


def scale_noise_to_snr(clean, noise, snr_db_value, eps=1e-12):
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2) + eps
    target_noise_power = clean_power / (10 ** (snr_db_value / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return scale * noise, scale


def snr_db(clean, test, eps=1e-12):
    error = test - clean
    return 10 * np.log10((np.sum(clean**2) + eps) / (np.sum(error**2) + eps))


def speech_distortion(clean, test, eps=1e-12):
    return np.sum((clean - test) ** 2) / (np.sum(clean**2) + eps)