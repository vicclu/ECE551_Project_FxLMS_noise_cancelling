"""
Experiment 1: LMS/NLMS speech-noise cancellation

Goal:
    Simulate a microphone close to the mouth that receives speech + noise,
    plus a second reference microphone that mostly receives correlated noise.
    The adaptive filter predicts the noise and subtracts it to produce a
    cleaner speech signal.

Output folder:
    results/experiment_1/
"""

import numpy as np
from pathlib import Path

from config import (
    FS_TARGET,
    MAX_SECONDS,
    WEIGHT_LEN_SPEECH,
    MU_ENHANCE,
    INPUT_SNR_DB,
    ALPHA_SPEECH_LEAKAGE,
    WAV_FILENAME,
    WAV_NOISE,
    NOISE_INPUT,
)

from helpers import (
    fir_filter,
    scale_noise_to_snr,
    snr_db,
    find_wav_file,
)

from signal_generation import (
    load_wav,
    save_wav_float,
    make_engine_like_noise,
)

from adaptive_filters import signal_noise_nlms
from plotting import (
    plot_speech_case_output,
    plot_noise_diff,    
)

def save_experiment_1_metrics(
    metrics_path: Path,
    speech_path: Path,
    fs: int,
    duration_sec: float,
    speech_input_snr: float,
    speech_enhanced_snr: float,
    speech_enhancement_improvement: float,
) -> None:
    """Save a small metrics report for Experiment 1 only."""
    metrics_path.write_text(
        "Experiment 1: LMS/NLMS speech-noise cancellation\n"
        "=================================================\n\n"
        f"Speech file:                  {speech_path}\n"
        f"Sampling frequency:           {fs} Hz\n"
        f"Duration:                     {duration_sec:.3f} s\n\n"
        f"Input speech SNR:             {speech_input_snr:.3f} dB\n"
        f"Enhanced speech SNR:          {speech_enhanced_snr:.3f} dB\n"
        f"Speech SNR improvement:       {speech_enhancement_improvement:.3f} dB\n",
        encoding="utf-8",
    )


def main_experiment_1():
    # -------------------------------------------------
    # Create output folders
    # -------------------------------------------------
    results_dir = Path("results")
    output_dir = results_dir / "experiment_1"
    figures_dir = output_dir / "figures"
    generated_sounds_dir = output_dir / "generated_sounds"
    test_outputs_dir = output_dir / "test_outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_sounds_dir.mkdir(parents=True, exist_ok=True)
    test_outputs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load real speech WAV file
    # -------------------------------------------------
    speech_path = find_wav_file(WAV_FILENAME)

    fs, speech = load_wav(
        path=speech_path,
        target_fs=FS_TARGET,
        max_seconds=MAX_SECONDS,
    )

    N = len(speech)
    T = N / fs
    t = np.arange(N) / fs

    print("\nExperiment 1: LMS/NLMS speech-noise cancellation")
    print("------------------------------------------------")
    print(f"Speech file:         {speech_path}")
    print(f"Sampling frequency:  {fs} Hz")
    print(f"Number of samples:   {N}")
    print(f"Duration:            {T:.3f} s")
    print(f"Output directory:    {output_dir}")

    # -------------------------------------------------
    # Load or generate base noise
    # -------------------------------------------------
    if NOISE_INPUT:
        noise_path = find_wav_file(WAV_NOISE)

        fs_noise, base_noise = load_wav(
            path=noise_path,
            target_fs=fs,
            max_seconds=MAX_SECONDS,
        )

        print(f"Noise file:          {noise_path}")
        print(f"Noise fs:            {fs_noise} Hz")
    else:
        base_noise = make_engine_like_noise(fs, T, seed=20)
        print("Noise source:        generated engine-like noise")

    base_noise = base_noise[:N]

    h_primary_noise = np.array([0.9, 0.35, -0.20, 0.10])
    h_reference_noise = np.array([0.4, 0.75, 0.25, -0.10])

    speech_primary_noise_raw = fir_filter(base_noise, h_primary_noise)
    speech_reference_noise_raw = fir_filter(base_noise, h_reference_noise)

    speech_primary_noise, noise_scale = scale_noise_to_snr(
        clean=speech,
        noise=speech_primary_noise_raw,
        snr_db_value=INPUT_SNR_DB,
    )

    speech_reference_noise = noise_scale * speech_reference_noise_raw

    # Optional speech leakage into the reference microphone.
    # ALPHA_SPEECH_LEAKAGE = 0 means the reference mic contains no speech.
    speech_reference_noise = speech_reference_noise + ALPHA_SPEECH_LEAKAGE * speech

    # -------------------------------------------------
    # Run LMS/NLMS speech enhancement
    # -------------------------------------------------
    cleaned_speech = signal_noise_nlms(
        speech=speech,
        speech_primary_noise=speech_primary_noise,
        speech_reference_noise=speech_reference_noise,
        w_filter_len=WEIGHT_LEN_SPEECH,
        mu=MU_ENHANCE,
    )

    noisy_speech_mic = cleaned_speech["noisy_speech_mic"]
    enhanced_speech = cleaned_speech["enhanced_speech"]

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    speech_input_snr = snr_db(speech, noisy_speech_mic)
    speech_enhanced_snr = snr_db(speech, enhanced_speech)
    speech_enhancement_improvement = speech_enhanced_snr - speech_input_snr

    save_experiment_1_metrics(
        metrics_path=test_outputs_dir / "metrics.txt",
        speech_path=speech_path,
        fs=fs,
        duration_sec=T,
        speech_input_snr=speech_input_snr,
        speech_enhanced_snr=speech_enhanced_snr,
        speech_enhancement_improvement=speech_enhancement_improvement,
    )

    print(f"Input speech SNR:        {speech_input_snr:.3f} dB")
    print(f"Enhanced speech SNR:     {speech_enhanced_snr:.3f} dB")
    print(f"Improvement:             {speech_enhancement_improvement:.3f} dB")

    # -------------------------------------------------
    # Save WAV outputs
    # -------------------------------------------------
    save_wav_float(generated_sounds_dir / "01_loaded_clean_speech.wav", fs, speech)
    save_wav_float(generated_sounds_dir / "02_loaded_noise.wav", fs, base_noise)
    save_wav_float(generated_sounds_dir / "03_primary_noise.wav", fs, speech_primary_noise)
    save_wav_float(generated_sounds_dir / "04_reference_noise.wav", fs, speech_reference_noise)
    save_wav_float(generated_sounds_dir / "05_noisy_speech_mic.wav", fs, noisy_speech_mic)
    save_wav_float(generated_sounds_dir / "06_enhanced_speech_mic.wav", fs, enhanced_speech)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plot_speech_case_output(
        t=t,
        speech=speech,
        noisy=cleaned_speech,
        figures_dir=figures_dir,
        fs=fs,
        start_sec=3,
        duration_sec=2,
    )
    
    plot_noise_diff(
        t=t,
        noisy=cleaned_speech,
        figures_dir=figures_dir,
        fs=fs,
        primary_noise=speech_primary_noise,
        start_sec=3,
        duration_sec=2,
    )
    print(f"\nSaved figures to: {figures_dir}")
    print(f"Saved Experiment 1 sounds to:  {generated_sounds_dir}")

def main():
    main_experiment_1()

if __name__ == "__main__":
    main()
