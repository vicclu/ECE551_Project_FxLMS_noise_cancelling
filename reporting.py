import numpy as np


def print_metrics(
    speech_input_snr,
    speech_enhanced_snr,
    speech_enhancement_improvement,
    ear_input_snr,
    ear_output_snr,
    headphone_anc_improvement,
    final_ear_noise_power_before,
    final_residual_power_after,
    full_system_distortion,
):
    print("\nCombined speech enhancement + FxLMS headphone ANC")
    print("-------------------------------------------------")
    print(f"Speech mic input SNR:              {speech_input_snr:8.3f} dB")
    print(f"After speech enhancement SNR:      {speech_enhanced_snr:8.3f} dB")
    print(f"Speech enhancement improvement:    {speech_enhancement_improvement:8.3f} dB")
    print()
    print(f"Ear input SNR before ANC:          {ear_input_snr:8.3f} dB")
    print(f"Ear output SNR after FxLMS ANC:    {ear_output_snr:8.3f} dB")
    print(f"Headphone ANC improvement:         {headphone_anc_improvement:8.3f} dB")
    print()
    print(f"Final ear noise power before ANC:  {final_ear_noise_power_before:8.6f}")
    print(f"Final residual power after ANC:    {final_residual_power_after:8.6f}")
    print(f"Full-system distortion:            {full_system_distortion:8.6f}")
    
def save_metrics(
    metrics_path,
    speech_path,
    fs,
    T,
    speech_input_snr,
    speech_enhanced_snr,
    speech_enhancement_improvement,
    ear_input_snr,
    ear_output_snr,
    headphone_anc_improvement,
    final_ear_noise_power_before,
    final_residual_power_after,
    full_system_distortion,
):
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Combined speech enhancement + FxLMS headphone ANC\n")
        f.write("-------------------------------------------------\n")
        f.write(f"Speech file: {speech_path}\n")
        f.write(f"Sampling frequency: {fs} Hz\n")
        f.write(f"Duration: {T:.3f} s\n\n")

        f.write(f"Speech mic input SNR:              {speech_input_snr:8.3f} dB\n")
        f.write(f"After speech enhancement SNR:      {speech_enhanced_snr:8.3f} dB\n")
        f.write(f"Speech enhancement improvement:    {speech_enhancement_improvement:8.3f} dB\n\n")

        f.write(f"Ear input SNR before ANC:          {ear_input_snr:8.3f} dB\n")
        f.write(f"Ear output SNR after FxLMS ANC:    {ear_output_snr:8.3f} dB\n")
        f.write(f"Headphone ANC improvement:         {headphone_anc_improvement:8.3f} dB\n\n")

        f.write(f"Final ear noise power before ANC:  {final_ear_noise_power_before:8.6f}\n")
        f.write(f"Final residual power after ANC:    {final_residual_power_after:8.6f}\n")
        f.write(f"Full-system distortion:            {full_system_distortion:8.6f}\n")
        
def run_sanity_checks(
    enhanced_speech,
    media_to_ear,
    ear_mic_signal,
    anc_error_used,
    w_enhance_hist,
    w_anc_hist,
    N,
    weight_len_speech,
):
    assert len(enhanced_speech) == N
    assert len(media_to_ear) == N
    assert len(ear_mic_signal) == N
    assert len(anc_error_used) == N

    assert w_enhance_hist.shape == (N, weight_len_speech)
    assert w_anc_hist.shape == (N, weight_len_speech)

    assert np.all(np.isfinite(enhanced_speech))
    assert np.all(np.isfinite(media_to_ear))
    assert np.all(np.isfinite(ear_mic_signal))
    assert np.all(np.isfinite(anc_error_used))

    print("\nShape and finite-value checks passed.")
    

def metrics(
    metrics_path,
    speech_path,
    fs,
    T,
    speech_input_snr,
    speech_enhanced_snr,
    speech_enhancement_improvement,
    ear_input_snr,
    ear_output_snr,
    headphone_anc_improvement,
    final_ear_noise_power_before,
    final_residual_power_after,
    full_system_distortion,
):
    print_metrics(
        speech_input_snr=speech_input_snr,
        speech_enhanced_snr=speech_enhanced_snr,
        speech_enhancement_improvement=speech_enhancement_improvement,
        ear_input_snr=ear_input_snr,
        ear_output_snr=ear_output_snr,
        headphone_anc_improvement=headphone_anc_improvement,
        final_ear_noise_power_before=final_ear_noise_power_before,
        final_residual_power_after=final_residual_power_after,
        full_system_distortion=full_system_distortion,
    )

    save_metrics(
        metrics_path=metrics_path,
        speech_path=speech_path,
        fs=fs,
        T=T,
        speech_input_snr=speech_input_snr,
        speech_enhanced_snr=speech_enhanced_snr,
        speech_enhancement_improvement=speech_enhancement_improvement,
        ear_input_snr=ear_input_snr,
        ear_output_snr=ear_output_snr,
        headphone_anc_improvement=headphone_anc_improvement,
        final_ear_noise_power_before=final_ear_noise_power_before,
        final_residual_power_after=final_residual_power_after,
        full_system_distortion=full_system_distortion,
    )

    print(f"\nSaved metrics to: {metrics_path}")
