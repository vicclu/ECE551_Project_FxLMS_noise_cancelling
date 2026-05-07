import matplotlib.pyplot as plt
from pathlib import Path

from helpers import moving_average
import numpy as np


def plot_anc_residual_power(
    ear_noise,
    anc_error_used,
    smooth_window=500,
    save_path=None,
):
    plt.figure(figsize=(11, 5))

    plt.plot(
        moving_average(ear_noise**2, smooth_window),
        label="Ear noise before ANC",
        linewidth=2,
    )

    plt.plot(
        moving_average(anc_error_used**2, smooth_window),
        label="Residual noise after FxLMS ANC",
        linewidth=2,
    )

    plt.title("Headphone Noise Reduction Without Cancelling Speech")
    plt.xlabel("Iteration")
    plt.ylabel("Smoothed power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()




def plot_snr_bar(
    speech_input_snr,
    speech_enhanced_snr,
    ear_input_snr,
    ear_output_snr,
    save_path=None,
):
    labels = [
        "Speech mic input",
        "After speech enhancement",
        "Ear before ANC",
        "Ear after ANC",
    ]

    values = [
        speech_input_snr,
        speech_enhanced_snr,
        ear_input_snr,
        ear_output_snr,
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.title("SNR Comparison")
    plt.ylabel("SNR [dB]")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    
    
# -------------------------------------------------
# Plot helpers
# -------------------------------------------------

def plot_weight_bias(clean, noisy, figures_dir, weight_index=0, plot_start=0):
    """
    Compares learned NLMS weights between:

        clean/noise-only case:
            speech = 0

        noisy/speech+noise case:
            speech = real speech

    The reference microphone has no speech leakage in either case.
    """

    clean_w_hist = clean["w_update_hist_signal_noise"]
    noisy_w_hist = noisy["w_update_hist_signal_noise"]

    N = clean_w_hist.shape[0]
    iters = np.arange(plot_start, N)

    delta_w = clean_w_hist - noisy_w_hist

    k = weight_index

    plt.figure(figsize=(10, 5))

    plt.plot(
        iters,
        clean_w_hist[plot_start:, k],
        label=f"noise-only learned w[{k}]",
        linewidth=2,
    )

    plt.plot(
        iters,
        noisy_w_hist[plot_start:, k],
        label=f"speech+noise learned w[{k}]",
        linewidth=2,
    )

    plt.plot(
        iters,
        delta_w[plot_start:, k],
        ":",
        label=f"empirical bias = noise-only w[{k}] - speech+noise w[{k}]",
        linewidth=2,
    )

    plt.title(f"Stage 1 NLMS Weight Bias for w[{k}]")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / f"weight_bias_w{k}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")


def plot_all_weight_biases(clean, noisy, figures_dir, plot_start=0):
    clean_w_hist = clean["w_update_hist_signal_noise"]
    noisy_w_hist = noisy["w_update_hist_signal_noise"]

    N, W = clean_w_hist.shape
    iters = np.arange(plot_start, N)

    delta_w = clean_w_hist - noisy_w_hist

    plt.figure(figsize=(10, 5))

    for k in range(W):
        plt.plot(
            iters,
            delta_w[plot_start:, k],
            linewidth=1,
            label=f"w[{k}]",
        )

    plt.title("Stage 1 Empirical Bias for All NLMS Weights")
    plt.xlabel("Iteration")
    plt.ylabel("noise-only weight - speech+noise weight")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()

    save_path = figures_dir / "all_weight_biases.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")


def plot_noise_only_cancellation(t, clean, figures_dir, fs, start_sec=3, duration_sec=2):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        clean["noisy_speech_mic"][start:end],
        label="primary noise input",
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        clean["estimated_speech_noise"][start:end],
        label="estimated noise",
        linewidth=1,
        alpha=0.6,

    )

    plt.plot(
        t[start:end],
        clean["enhanced_speech"][start:end],
        label="residual after cancellation",
        linewidth=1,
    )

    plt.title("Stage 1 Noise-Only Case")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "noise_only_cancellation.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")

##Experiment 1 plots
def plot_speech_case_output(t, speech, noisy, figures_dir, fs, start_sec=3, duration_sec=2):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        noisy["input_speech"][start:end],
        label="clean speech",
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        noisy["noisy_speech_mic"][start:end],
        label="speech + primary noise",
        linewidth=1,
        alpha=0.6,

    )

    plt.plot(
        t[start:end],
        noisy["enhanced_speech"][start:end],
        label="enhanced speech",
        linewidth=1,
    
    )

    plt.title("Stage 1 Speech+Noise Case")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "speech_case_output.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")


    
def plot_noise_diff(
    t,
    noisy,
    figures_dir,
    fs,
    primary_noise,
    start_sec=3,
    duration_sec=2,
):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    # If your adaptive filter explicitly returns the estimated noise, use it.
    # Otherwise, estimate it from:
    # noisy_speech_mic = enhanced_speech + estimated_noise
    if "estimated_speech_noise" in noisy:
        estimated_noise = noisy["estimated_speech_noise"]
    else:
        estimated_noise = noisy["noisy_speech_mic"] - noisy["enhanced_speech"]

    noise_error = primary_noise - estimated_noise

    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        primary_noise[start:end],
        label="true primary noise",
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        -estimated_noise[start:end],
        label="(-) estimated primary noise",
        linewidth=1,
        alpha=0.7,
    )

    plt.plot(
        t[start:end],
        noise_error[start:end],
        label="noise estimation error",
        linewidth=1,
    )

    plt.title("noise_diff")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "noise_diff.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")


##Experiment 2 plots

def plot_noise_comparison(
    t,
    noise_input_1,
    name_1,
    noise_input_2,
    name_2,
    noise_input_3,
    name_3,
    figures_dir,
    fs,
    start_sec=3,
    duration_sec=2,
):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        noise_input_1[start:end],
        label=name_1,
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        noise_input_2[start:end],
        label=name_2,
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        noise_input_3[start:end],
        label=name_3,
        linewidth=1,
        alpha=0.7,

    )

    plt.title("Noise comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "noise_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")
    
def plot_noise_antinoise_compare(
    t,
    input,
    figures_dir,
    fs,
    start_sec=3,
    duration_sec=2,
):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)
    noise_error = input["ear_noise"][start:end] + input["anti_noise_at_ear"][start:end]


    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        input["ear_noise"][start:end],
        label="ear_noise",
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        input["anti_noise_at_ear"][start:end],
        label="anti noise at ear",
        linewidth=1,
        alpha=0.5,

    )
    
    plt.plot(
        t[start:end],
        noise_error,
        label="differnce",
        linewidth=1,
    )

    plt.title("Noise comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "noise_antinoise_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")
    
def plot_speech_case_output_2(t, input, figures_dir, fs, start_sec=3, duration_sec=2):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    plt.figure(figsize=(10, 5))

    plt.plot(
        t[start:end],
        input["input"][start:end],
        label="clean speech",
        linewidth=1,
    )

    plt.plot(
        t[start:end],
        input["media_to_ear"][start:end] + input["ear_noise"][start:end] ,
        label="speech + ear noise",
        linewidth=1,
        alpha=0.6,

    )

    plt.plot(
        t[start:end],
        input["ear_mic_signal"][start:end],
        label="enhanced speech",
        linewidth=1,
        alpha=0.6,

    )

    plt.title("Speech+Noise Case")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = figures_dir / "speech_case_output.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")

def plot_weight_trajectory(
    t,
    result,
    figures_dir,
    fs,
    start_sec=3,
    duration_sec=2,
    name = None
):
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)

    w_hist = result["w_update_hist_FxNLMS"]

    # Expected shape: (N, number_of_weights)
    if w_hist.ndim != 2:
        raise ValueError(
            f"Expected w_update_hist_FxNLMS to be 2D with shape (N, L), "
            f"but got shape {w_hist.shape}"
        )

    plt.figure(figsize=(10, 5))

    num_weights = w_hist.shape[1]

    for k in range(num_weights):
        plt.plot(
            t[start:end],
            w_hist[start:end, k],
            label=f"w[{k}]",
            linewidth=1,
        )

    plt.title(f"FxLMS Weight Trajectories {name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Weight value")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    save_path = figures_dir / f"fxlms_{name}weight_trajectories.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")
