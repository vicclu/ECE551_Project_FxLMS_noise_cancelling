import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test_config import (
    FS_TARGET,
    MAX_SECONDS,
    WEIGHT_LEN,
    MU,
    INPUT_SNR_DB,
)

from helpers import (
    fir_filter,
    scale_noise_to_snr,
)

from signal_generation import (
    make_engine_like_noise,
)

from adaptive_filters import fxnlms


@dataclass
class FxNLMSCase:
    fs: int
    N: int
    T: float
    t: np.ndarray

    clean_tone: np.ndarray
    base_noise: np.ndarray

    ear_noise: np.ndarray
    environment_noise_ref: np.ndarray

    playback_path: np.ndarray
    secondary_path: np.ndarray
    secondary_path_hat: np.ndarray

    w_filter_len: int
    
def algorithm_label(normalize):
    if normalize:
        return "FxNLMS"
    return "FxLMS"

def make_fxnlms_case(
    seed=20,
    input_seed=1,
    noise_seed=2,
    noise_std=0.2,
    clean=False,
    h_ear_noise=None,
    h_headphone_reference_noise=None,
    secondary_path=None,
    secondary_path_hat=None,
):
    fs = FS_TARGET
    T = MAX_SECONDS
    N = int(fs * T)
    t = np.arange(N) / fs

    # -------------------------------------------------
    # Input signal: white noise
    # -------------------------------------------------

    rng_input = np.random.default_rng(input_seed)
    rng_noise = np.random.default_rng(noise_seed)

    input_signal = rng_input.standard_normal(N)

    # RMS normalization for fair testing
    input_signal = input_signal / (np.sqrt(np.mean(input_signal**2)) + 1e-12)

    # -------------------------------------------------
    # Plant/output noise
    # -------------------------------------------------

    if clean:
        output_noise = np.zeros(N)
    else:
        output_noise = noise_std * rng_noise.standard_normal(N)

    # -------------------------------------------------
    # Acoustic / FxLMS paths
    # -------------------------------------------------

    if h_ear_noise is None:
        h_ear_noise = np.array([0.00, 0.00, 0.32, 0.18, 0.08, 0.035, -0.015])

    if h_headphone_reference_noise is None:
        h_headphone_reference_noise = np.array([0.90, 0.18, -0.07, 0.03, -0.015])

    if secondary_path is None:
        secondary_path = np.array([0.0, 0.85, 0.25, -0.10])

    if secondary_path_hat is None:
        secondary_path_hat = secondary_path.copy()

    playback_path = np.array([0.95, 0.10, -0.05])

    # -------------------------------------------------
    # Environmental noise setup
    # -------------------------------------------------

    base_noise = make_engine_like_noise(fs, T, seed=seed)
    base_noise = base_noise[:N]

    ear_noise_raw = fir_filter(base_noise, h_ear_noise)
    reference_noise_raw = fir_filter(base_noise, h_headphone_reference_noise)

    ear_noise, ear_scale = scale_noise_to_snr(
        clean=input_signal,
        noise=ear_noise_raw,
        snr_db_value=INPUT_SNR_DB,
    )

    environment_noise_ref = ear_scale * reference_noise_raw

    # Add optional output/sensor noise to the ear disturbance
    ear_noise = ear_noise + output_noise

    return FxNLMSCase(
        fs=fs,
        N=N,
        T=T,
        t=t,
        base_noise=base_noise,
        ear_noise=ear_noise,
        environment_noise_ref=environment_noise_ref,
        playback_path=playback_path,
        secondary_path=secondary_path,
        secondary_path_hat=secondary_path_hat,
        w_filter_len=WEIGHT_LEN,
    )


def run_case(
    case,
    mu = MU,
    w_filter_len=None,
    secondary_path_hat=None,
    normalize=True,
    reference_gain=1.0,
):
    if w_filter_len is None:
        w_filter_len = case.w_filter_len

    if secondary_path_hat is None:
        secondary_path_hat = case.secondary_path_hat

    return fxnlms(
        input=case.clean_tone,
        p_hat=case.playback_path,
        enviroment_noise_ref=reference_gain * case.environment_noise_ref,
        secondary_path=case.secondary_path,
        secondary_path_hat=secondary_path_hat,
        w_filter_len=w_filter_len,
        ear_noise=reference_gain * case.ear_noise,
        mu=mu,
        N=case.N,
        clean=False,
        normalize=normalize,
    )

def tail_metrics(result, tail_fraction=0.8, divergence_threshold=1e6):
    error = result["error_FxNLMS"]
    ear_noise = result["ear_noise"]
    w_hist = result["w_update_hist_FxNLMS"]

    #Checks for nan in the error or weight
    finite_error = np.all(np.isfinite(error))
    finite_weights = np.all(np.isfinite(w_hist))

    #Check if the numbers are are under threshold
    max_abs_error = np.nanmax(np.abs(error))
    max_abs_weight = np.nanmax(np.abs(w_hist))

    diverged = (
        (not finite_error)
        or (not finite_weights)
        or max_abs_error > divergence_threshold
        or max_abs_weight > divergence_threshold
    )


    tail_start = int(tail_fraction * len(error))
    tail_error = error[tail_start:]
    tail_noise = ear_noise[tail_start:]

    #If diverged, we make the test = error or nan
    if diverged:
        tail_mse = np.nan
        noise_reduction_db = np.nan
    else:
        tail_mse = np.mean(tail_error**2)
        input_noise_power = np.mean(tail_noise**2)

        noise_reduction_db = 10 * np.log10(
            (input_noise_power + 1e-12) / (tail_mse + 1e-12)
        )

    input_noise_power = np.mean(tail_noise**2)

    return {
        "tail_mse": float(tail_mse),
        "input_noise_power": float(input_noise_power),
        "noise_reduction_db": float(noise_reduction_db),
        "max_abs_error": float(max_abs_error),
        "max_abs_weight": float(max_abs_weight),
        "diverged": bool(diverged),
        "stable": int(not diverged),
    }

def save_csv(rows, path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def plot_sweep_by_algorithm(
    rows,
    x_key,
    y_key,
    save_path,
    title,
    xlabel,
    ylabel,
    algorithm_key="algorithm",
    log_x=False,
    log_y=False,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = sorted(set(row[algorithm_key] for row in rows))

    plt.figure(figsize=(8, 5))

    for algorithm in algorithms:
        alg_rows = [row for row in rows if row[algorithm_key] == algorithm]

        x = np.array([row[x_key] for row in alg_rows], dtype=float)
        y = np.array([row[y_key] for row in alg_rows], dtype=float)

        # Replace non-finite values so matplotlib does not crash
        y = np.where(np.isfinite(y), y, np.nan)

        order = np.argsort(x)

        plt.plot(
            x[order],
            y[order],
            marker="o",
            linewidth=1.5,
            label=algorithm,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    if log_x:
        plt.xscale("log")

    if log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")