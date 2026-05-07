"""
Experiment 2: FxNLMS headphone active noise cancellation

Goal:
    Simulate headphone ANC where a reference microphone measures environmental
    noise, the adaptive filter generates anti-noise, and the anti-noise passes
    through a secondary path before reaching the ear/error microphone.

Output folder:
    results/experiment_2/
"""

import numpy as np
from pathlib import Path

from config import (
    FS_TARGET,
    MAX_SECONDS,
    WEIGHT_LEN_SPEECH,
    MU_FXLMS,
    INPUT_SNR_DB,
    WAV_FILENAME,
    WAV_NOISE,
    NOISE_INPUT,
)

from helpers import (
    normalize,
    fir_filter,
    scale_noise_to_snr,
    snr_db,
    speech_distortion,
    find_wav_file
)

from signal_generation import (
    load_wav,
    save_wav_float,
    make_engine_like_noise,
)

from adaptive_filters import fxnlms
from plotting import (
    plot_noise_comparison,
    plot_noise_antinoise_compare,
    plot_speech_case_output_2,
    plot_weight_trajectory
    )

def save_experiment_2_metrics(
    metrics_path: Path,
    speech_path: Path,
    fs: int,
    duration_sec: float,
    ear_input_snr: float,
    ear_output_snr: float,
    headphone_anc_improvement: float,
    final_ear_noise_power_before: float,
    final_residual_power_after: float,
    headphone_distortion: float,
) -> None:
    """Save a small metrics report for Experiment 2 only."""
    metrics_path.write_text(
        "Experiment 2: FxNLMS headphone active noise cancellation\n"
        "=======================================================\n\n"
        f"Speech file:                  {speech_path}\n"
        f"Sampling frequency:           {fs} Hz\n"
        f"Duration:                     {duration_sec:.3f} s\n\n"
        f"Ear input SNR before ANC:     {ear_input_snr:.3f} dB\n"
        f"Ear output SNR after ANC:     {ear_output_snr:.3f} dB\n"
        f"Headphone ANC improvement:    {headphone_anc_improvement:.3f} dB\n\n"
        f"Final noise power before ANC: {final_ear_noise_power_before:.8e}\n"
        f"Final residual power after:   {final_residual_power_after:.8e}\n"
        f"Speech/playback distortion:   {headphone_distortion:.8e}\n",
        encoding="utf-8",
    )


def main_experiment_2():
    # -------------------------------------------------
    # Create output folders
    # -------------------------------------------------
    results_dir = Path("results")
    output_dir = results_dir / "experiment_2"
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

    print("\nExperiment 2: FxNLMS headphone active noise cancellation")
    print("-------------------------------------------------------")
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

    # -------------------------------------------------
    # Headphone ANC noise setup
    # -------------------------------------------------
    # k(n): environmental noise that reaches the ear/error mic
    # x(n): reference microphone noise, correlated with d(n)

    #Acoustic path: primary path -> enviroment noise -> hedphone passive isolation -> ear canal 
    h_ear_noise = np.array([0.3, -0.1, 0.03, 0.05])
    
    #Acoustic path: outside noise source -> air/reflections/headphone surface -> outside ANC reference mic
    h_headphone_reference_noise = np.array([0.30, 0.40, 0.20, -0.08])

    #Acousitc path (h_ear_noise) filter for noise to ear
    ear_noise_raw = fir_filter(base_noise, h_ear_noise)
    
    #Acousitc path (h_ear_noise) filter for noise to headphone
    ear_reference_noise_raw = fir_filter(base_noise, h_headphone_reference_noise)

    #???
    ear_noise, ear_scale = scale_noise_to_snr(
        clean=speech,
        noise=ear_noise_raw,
        snr_db_value=INPUT_SNR_DB,
    )

    enviroment_noise_ref =  ear_reference_noise_raw


    # secondary_path:      #true speaker-to-ear/error-mic path S(z)
    # secondary_path_hat:  estimated path used by FxNLMS, S_hat(z)
    # playback_path:       clean speech playback path to the ear
    #true speaker-to-ear/error-mic path H(z)
    secondary_path = np.array([0.0, 0.85, 0.25, -0.10])
    #estimated path used by FxNLMS, H_hat(z)
    secondary_path_hat = np.array([0.0, 0.85, 0.25, -0.10])
    
    #clean speech playback path to the ear
    playback_path = np.array([0.95, 0.10, -0.05])

    # -------------------------------------------------
    # Run FxNLMS headphone ANC
    # -------------------------------------------------
    # In this standalone experiment, clean speech is played directly.
    # The variable name "enhanced_speech" is kept only because fxnlms()
    # expects that argument name from your current adaptive_filters.py.

    result = fxnlms(
        input=speech,
        p_hat=playback_path,
        enviroment_noise_ref=enviroment_noise_ref,
        secondary_path=secondary_path,
        secondary_path_hat=secondary_path_hat,
        w_filter_len=WEIGHT_LEN_SPEECH,
        ear_noise=ear_noise,
        mu=MU_FXLMS,
        N=N,
    )

    noisy_media_to_ear = result["media_to_ear"] + ear_noise
    anc_error_used = result["error_FxNLMS"]
    anti_noise_at_ear = result["anti_noise_at_ear"]

    # -------------------------------------------------
    # Evaluation signals
    # -------------------------------------------------
    clean_target_at_ear = fir_filter(speech, playback_path)
    target_speech_at_ear = result["media_to_ear"]

    ear_input_snr = snr_db(
        clean_target_at_ear,
        clean_target_at_ear + ear_noise,
    )

    ear_output_snr = snr_db(
        clean_target_at_ear,
        result["ear_mic_signal"],
    )

    headphone_anc_improvement = ear_output_snr - ear_input_snr

    final_slice = slice(3 * N // 4, N)
    final_ear_noise_power_before = np.mean(ear_noise[final_slice] ** 2)
    final_residual_power_after = np.mean(anc_error_used[final_slice] ** 2)

    headphone_distortion = speech_distortion(
        clean_target_at_ear,
        result["ear_mic_signal"],
    )

    save_experiment_2_metrics(
        metrics_path=test_outputs_dir / "metrics.txt",
        speech_path=speech_path,
        fs=fs,
        duration_sec=T,
        ear_input_snr=ear_input_snr,
        ear_output_snr=ear_output_snr,
        headphone_anc_improvement=headphone_anc_improvement,
        final_ear_noise_power_before=final_ear_noise_power_before,
        final_residual_power_after=final_residual_power_after,
        headphone_distortion=headphone_distortion,
    )

    print(f"Ear input SNR before ANC:  {ear_input_snr:.3f} dB")
    print(f"Ear output SNR after ANC:  {ear_output_snr:.3f} dB")
    print(f"ANC improvement:           {headphone_anc_improvement:.3f} dB")

    # -------------------------------------------------
    # Shared peaks for fair listening comparisons
    # -------------------------------------------------

    # Compare environmental noise paths:
    # base noise vs ear noise vs reference mic noise
    noise_path_peak = max(
        np.max(np.abs(base_noise)),
        np.max(np.abs(ear_noise_raw)),
        np.max(np.abs(enviroment_noise_ref)),
    ) + 1e-12

    # Compare signals at the ear:
    # speech, anti-noise, final mic signal, ANC error
    ear_signal_peak = max(
        np.max(np.abs(result["media_to_ear"])),
        np.max(np.abs(anti_noise_at_ear)),
        np.max(np.abs(result["ear_mic_signal"])),
        np.max(np.abs(anc_error_used)),
    ) + 1e-12
    # -------------------------------------------------
    # Save WAV outputs
    # -------------------------------------------------

    # Clean speech can be normalized alone for easy listening
    save_wav_float(generated_sounds_dir / "01_loaded_clean_speech.wav",fs,speech,normalize=True)

    # Noise path comparison: same reference peak
    save_wav_float(generated_sounds_dir / "02_loaded_noise.wav",fs,base_noise,normalize=True,reference_peak=noise_path_peak)
    save_wav_float(generated_sounds_dir / "03_ear_noise_raw.wav",fs,ear_noise_raw,normalize=True,reference_peak=noise_path_peak)
    save_wav_float(generated_sounds_dir / "04_environment_reference_noise.wav",fs,enviroment_noise_ref,normalize=True,reference_peak=noise_path_peak)

    # Ear signal comparison: same reference peak
    save_wav_float(generated_sounds_dir / "05_clean_media_to_ear.wav",fs,result["media_to_ear"],normalize=True,reference_peak=ear_signal_peak)
    save_wav_float(generated_sounds_dir / "06_noisy_media_to_ear.wav",fs,noisy_media_to_ear,normalize=True,reference_peak=ear_signal_peak)
    save_wav_float(generated_sounds_dir / "07_anti_noise_at_ear.wav",fs,anti_noise_at_ear,normalize=True,reference_peak=ear_signal_peak)
    save_wav_float(generated_sounds_dir / "08_ear_mic_signal_after_anc.wav",fs,result["ear_mic_signal"],normalize=True,reference_peak=ear_signal_peak)
    save_wav_float(generated_sounds_dir / "09_anc_error_used.wav",fs,anc_error_used,normalize=True,reference_peak=ear_signal_peak)
    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    
    plot_noise_comparison(
        t = t,
        noise_input_1= base_noise,
        name_1="loaded_noise",
        noise_input_2= ear_noise_raw,
        name_2="ear noise raw",
        noise_input_3= enviroment_noise_ref,
        name_3="Enviroment rference noise",
        figures_dir=figures_dir,
        fs=fs,
        start_sec=3,
        duration_sec=2,
    )
    
    plot_noise_antinoise_compare(
        t=t,
        input = result,
        figures_dir=figures_dir,
        fs = fs,
        start_sec=3,
        duration_sec=2,
    )
    plot_speech_case_output_2(
        t=t,
        input = result,
        figures_dir=figures_dir,
        fs = fs,
        start_sec=3,
        duration_sec=2,        
    )
    
    plot_weight_trajectory(
        t=t,
        result=result,
        figures_dir=figures_dir,
        fs=fs,
        start_sec=0,
        duration_sec=1,
    )


    print(f"Saved Experiment 2 figures to: {figures_dir}")
    print(f"Saved Experiment 2 sounds to:  {generated_sounds_dir}")


def main() -> None:
    main_experiment_2()


if __name__ == "__main__":
    main()
