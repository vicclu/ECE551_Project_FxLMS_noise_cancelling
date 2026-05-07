import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from helpers import normalize, fir_filter


def load_wav(path, target_fs=8000, max_seconds=30):
    fs_original, x = wavfile.read(path)

    #Normalize to float from -1 to 1
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / np.iinfo(x.dtype).max
    else:
        x = x.astype(float)

    #If stereo -> mono
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    #Remove DC offset
    x = x - np.mean(x)

    if target_fs is not None and fs_original != target_fs:
        gcd = np.gcd(fs_original, target_fs)
        up = target_fs // gcd
        down = fs_original // gcd
        #Uses polyphase from ch 3
        x = resample_poly(x, up, down) #Upsamples -> Filters -> DOwnsamples
        fs = target_fs
    else:
        fs = fs_original

    max_samples = int(max_seconds * fs)
    x = x[:max_samples]

    x = normalize(x)

    return fs, x

def save_wav_float(path, fs, x, normalize=True, reference_peak=None):
    """
    Save float signal as int16 WAV.

    normalize=True:
        Normalizes this signal, or uses reference_peak if provided.

    normalize=False:
        Preserves relative amplitude and clips to [-1, 1].
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x, dtype=float)

    if normalize:
        if reference_peak is None:
            reference_peak = np.max(np.abs(x)) + 1e-12

        x = x / (reference_peak + 1e-12)
    else:
        x = np.clip(x, -1.0, 1.0)

    x = np.clip(x, -1.0, 1.0)
    x_int16 = (0.95 * x * 32767).astype(np.int16)

    wavfile.write(path, fs, x_int16)



def make_engine_like_noise(fs, T, seed=20):
    rng = np.random.default_rng(seed)
    N = int(fs * T)
    t = np.arange(N) / fs

    white = rng.standard_normal(N)

    h_col = np.array([0.05, 0.10, 0.16, 0.20, 0.16, 0.10, 0.05])
    colored = fir_filter(white, h_col)

    tonal = (
        1.0 * np.sin(2 * np.pi * 60 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t + 0.4)
        + 0.25 * np.sin(2 * np.pi * 180 * t + 1.2)
    )

    noise = colored + 0.8 * tonal
    return normalize(noise)

def frequency_tone(fs, T, freq=440, amplitude=1.0, phase=0.0):
    """
    Generate a pure sinusoidal tone.

    fs:
        Sampling frequency in Hz.

    T:
        Duration in seconds.

    freq:
        Tone frequency in Hz.

    amplitude:
        Signal amplitude.

    phase:
        Initial phase in radians.
    """
    N = int(fs * T)
    t = np.arange(N) / fs

    tone = amplitude * np.sin(2 * np.pi * freq * t + phase)

    return tone