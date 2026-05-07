import numpy as np

from helpers import past_vector, fir_filter
# -------------------------------------------------
# Speech microphone noise setup
# -------------------------------------------------
# Primary speech mic:
#   noisy_speech_mic d(n) = speech s(n) + primary noise n_s(n)
#
# Reference mic:
#   speech_reference_noise x(n) = mostly correlated noise n_r(n)
#   plus optional speech leakage.
def signal_noise_nlms(
    speech, #s
    speech_primary_noise, #ns
    speech_reference_noise, #nr
    w_filter_len, #W
    mu,    
):
    N = len(speech)
    w_update = np.zeros(w_filter_len)
    noisy_speech_mic = speech + speech_primary_noise # d = s + ns
    estimated_speech_noise = np.zeros(N) #initiate y
    enhanced_speech = np.zeros(N) # z = d-y

    w_update_hist = np.zeros((N, w_filter_len))
    
    eps = 1e-8
    
    for n in range(N):
        nr_vec = past_vector(speech_reference_noise, n, w_filter_len)

        estimated_speech_noise[n] = np.dot(w_update, nr_vec)
        enhanced_speech[n] = noisy_speech_mic[n] - estimated_speech_noise[n]

        denom = eps + np.dot(nr_vec, nr_vec)
        w_update = w_update + mu * enhanced_speech[n] * nr_vec / denom
    
        w_update_hist[n, :] = w_update
        
    return {
        "input_speech" : speech,
        "noisy_speech_mic": noisy_speech_mic,
        "estimated_speech_noise": estimated_speech_noise,
        "enhanced_speech": enhanced_speech,
        "w_update_hist_signal_noise": w_update_hist,
        "speech_primary_noise" : speech_primary_noise,
        "speech_reference_noise":speech_reference_noise
        
    }
    
def fxnlms(
    input, #m_j
    enviroment_noise_ref, #x_j
    secondary_path, #H(z) = speaker to ear path
    secondary_path_hat, #H_hat
    w_filter_len,
    ear_noise, #k_j
    mu,
    N,
    clean=False,
    normalize = True,
):

    # Choose whether environmental noise exists or not
    if clean:
        enviroment_noise_ref_used = np.zeros(N)
        ear_noise_used = np.zeros(N)
    else:
        enviroment_noise_ref_used = enviroment_noise_ref
        ear_noise_used = ear_noise
    
    #sound (start) -> headphone -> headphone cavities -> ear canal -> perceived sound (finish)
    #y_hat = H(z)_hat x
    estimation_reference = fir_filter(enviroment_noise_ref_used, secondary_path_hat) 
    media_to_ear = fir_filter(input, secondary_path)
    
    plant_len = len(secondary_path)
    
    #initiate
    w_update = np.zeros(w_filter_len)
    d_headphone = np.zeros(N)
    anti_noise_at_ear = np.zeros(N)
    ear_mic_signal = np.zeros(N)
    error = np.zeros(N)
    w_update_hist = np.zeros((N, w_filter_len))
    
    eps = 1e-8
    for n in range(N):
        x_vec = past_vector(enviroment_noise_ref_used, n, w_filter_len)

        #d = W(z)x
        d_headphone[n] = np.dot(w_update, x_vec)
        d_vec = past_vector(d_headphone, n, plant_len)
        
        #c = H(z)d
        anti_noise_at_ear[n] = np.dot(secondary_path, d_vec)

        ear_mic_signal[n] = media_to_ear[n] + ear_noise_used[n] + anti_noise_at_ear[n]

        # This prevents the FxLMS loop from cancelling the desired speech. Software processing
        error[n] = ear_mic_signal[n] - media_to_ear[n]

        y_hat_vec = past_vector(estimation_reference, n, w_filter_len)

        if normalize:
            denom = eps + np.dot(y_hat_vec, y_hat_vec)
            w_update = w_update - mu * error[n] * y_hat_vec / denom
        else:
            w_update = w_update - mu * error[n] * y_hat_vec
    
        w_update_hist[n, :] = w_update
    
    return {
        "input" : input,
        "media_to_ear": media_to_ear,
        "ear_mic_signal": ear_mic_signal,
        "ear_noise": ear_noise_used,
        "ear_noise_original": ear_noise,
        "enviroment_noise_ref": enviroment_noise_ref_used,
        "enviroment_noise_ref_original": enviroment_noise_ref,
        "anti_noise_at_ear": anti_noise_at_ear,
        "error_FxNLMS": error,
        "d_headphone": d_headphone,
        "w_update_hist_FxNLMS": w_update_hist,
        "clean": clean,
        "normalize": normalize,
    }


def nlms(
    input,
    reference_filter, #H_m(z) 
    plant_filter, #H(z) 
    w_filter_len,
    noise,
    mu,
    N,
    clean = False,
):
    plant_len = len(plant_filter)
    reference_len = len(reference_filter)

    noisy_plant = np.zeros(N)
    inverse_input = np.zeros(N)
    
    inverse_plant_output = np.zeros(N)
    reference_model_output = np.zeros(N)

    error = np.zeros(N)
    w_update = np.zeros(w_filter_len)
    w_update_hist = np.zeros((N, w_filter_len))
        
    eps = 1e-8
    
    for n in range(N):
        x_vec = past_vector(input, n, w_filter_len)
        inverse_input = np.dot(w_update, x_vec)
        inverse_input_vec = past_vector(inverse_input, w_filter_len)
        plant_output = np.dot(plant_filter, inverse_input_vec)
        
        if clean:
            noisy_plant[n] = plant_output
        else:
            noisy_plant[n] = plant_output[n] + noise[n]
        
        noisy_plant_vec = past_vector(noisy_plant, n, plant_len)
        inverse_plant_output[n] = np.dot(w_update, noisy_plant_vec)
            
        reference_input_vec = past_vector(inverse_input_vec, n, reference_len)
        reference_model_output[n] = np.dot(reference_filter, reference_input_vec)
        
        error[n] = reference_model_output[n] - inverse_plant_output[n]        
        denom = eps + np.dot(inverse_plant_output,inverse_plant_output)

        w_update = w_update - mu * error[n] * noisy_plant_vec / denom
        w_update_hist[n, :] = w_update
    
    return {
        "input": input,
        "noise": noise,

        "inverse_input": inverse_input,
        "plant_output": plant_output,
        "noisy_plant": noisy_plant,

        "inverse_plant_output": inverse_plant_output,
        "reference_model_output": reference_model_output,

        "error": error,
        "w_update_final": w_update,
        "w_update_hist": w_update_hist,

        "plant_filter": plant_filter,
        "reference_filter": reference_filter,
    }