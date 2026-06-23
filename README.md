# ECE551 Project: FxLMS Noise Cancelling

Python simulations of adaptive noise cancellation using LMS/NLMS and Filtered-x LMS/FxNLMS. The project models two related noise-cancelling problems:

1. **Speech microphone noise reduction** using an adaptive NLMS filter.
2. **Headphone active noise cancellation (ANC)** using FxNLMS, a reference microphone, and a secondary-path model.

The scripts load speech and noise WAV files, simulate acoustic paths with FIR filters, run adaptive filters, compute SNR/noise-reduction metrics, and save generated audio and figures for analysis.

---

## Project goals

This project demonstrates how adaptive filters can be used for practical noise-cancelling systems:

- estimate and subtract correlated noise from a noisy speech microphone signal;
- generate anti-noise for a headphone ANC model;
- compare LMS-style and normalized LMS-style behavior;
- evaluate noise reduction using SNR, residual error power, and filter-weight trajectories;
- produce plots and WAV files for listening and debugging.

---

## Repository structure

```text
.
├── adaptive_filters.py          # NLMS, FxNLMS, and related adaptive-filter routines
├── config.py                    # Main experiment parameters
├── experiment_1.py              # Speech microphone LMS/NLMS noise-cancellation experiment
├── experiment_2.py              # Headphone FxNLMS ANC experiment
├── helpers.py                   # FIR filtering, SNR, normalization, utility functions
├── plotting.py                  # Figure-generation helpers
├── reporting.py                 # Metric printing/saving helpers
├── signal_generation.py         # WAV loading/saving and synthetic noise generation
├── sounds/                      # Input WAV files
├── results/                     # Generated experiment outputs
├── testers/                     # FxNLMS parameter-sweep test scripts
└── ipynb/                       # Exploratory Jupyter notebooks
```

---

## Main experiments

### Experiment 1: LMS/NLMS speech-noise cancellation

`experiment_1.py` simulates a speech microphone that receives:

```text
d(n) = s(n) + n_s(n)
```

where:

- `s(n)` is the clean speech signal;
- `n_s(n)` is the primary noise at the speech microphone;
- a reference microphone receives correlated noise `n_r(n)`;
- an adaptive NLMS filter estimates the primary noise and subtracts it from the noisy microphone signal.

The experiment saves:

- clean speech;
- loaded/generated noise;
- primary noise;
- reference noise;
- noisy speech microphone signal;
- enhanced speech signal;
- SNR metrics;
- plots comparing clean, noisy, and enhanced speech.

Run it with:

```bash
python experiment_1.py
```

Outputs are written to:

```text
results/experiment_1/
├── figures/
├── generated_sounds/
└── test_outputs/
```

---

### Experiment 2: FxNLMS headphone ANC

`experiment_2.py` simulates a headphone ANC system. The model includes:

- environmental noise reaching the ear/error microphone;
- a reference microphone measuring correlated environmental noise;
- an adaptive filter generating anti-noise;
- a secondary path from headphone speaker to ear/error microphone;
- an estimated secondary path used by the FxNLMS update.

The FxNLMS algorithm filters the reference signal through the estimated secondary path before updating the adaptive weights. This is necessary because the anti-noise signal is physically modified by the headphone speaker-to-ear path before it reaches the error microphone.

Run it with:

```bash
python experiment_2.py
```

Outputs are written to:

```text
results/experiment_2/
├── figures/
├── generated_sounds/
└── test_outputs/
```

Generated audio includes clean playback, noisy ear signal, anti-noise at the ear, final ear microphone signal after ANC, and the ANC error signal.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vicclu/ECE551_Project_FxLMS_noise_cancelling.git
cd ECE551_Project_FxLMS_noise_cancelling
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

The project currently does not include a `requirements.txt`, so the dependency list above is inferred from the imports used in the Python files.

---

## Configuration

Main parameters are set in `config.py`:

```python
FS_TARGET = 96000
MAX_SECONDS = 10
WEIGHT_LEN_SPEECH = 10
MU_ENHANCE = 0.0005
MU_FXLMS = 0.08
INPUT_SNR_DB = 0
ALPHA_SPEECH_LEAKAGE = 0.0
WAV_FILENAME = "harvard.wav"
WAV_NOISE = "noise_airplane.wav"
NOISE_INPUT = True
```

Important options:

| Parameter | Meaning |
|---|---|
| `FS_TARGET` | Target sampling frequency after resampling |
| `MAX_SECONDS` | Maximum duration loaded from each WAV file |
| `WEIGHT_LEN_SPEECH` | Adaptive filter length |
| `MU_ENHANCE` | NLMS step size for speech enhancement |
| `MU_FXLMS` | FxNLMS step size for headphone ANC |
| `INPUT_SNR_DB` | Target input SNR used when scaling noise |
| `ALPHA_SPEECH_LEAKAGE` | Optional speech leakage into the reference microphone |
| `WAV_FILENAME` | Speech WAV file in `sounds/` |
| `WAV_NOISE` | Noise WAV file in `sounds/` |
| `NOISE_INPUT` | If `True`, load noise WAV; if `False`, generate synthetic engine-like noise |

The repository includes example sound files such as:

```text
sounds/harvard.wav
sounds/noise_airplane.wav
sounds/noise_cafe.wav
sounds/noise_inside-car.wav
```

---

## Running parameter sweeps

The `testers/` folder contains scripts for testing FxNLMS behavior under different settings:

```bash
python testers/test1_fxnlms_mu_sweep.py
python testers/test2_fxnlms_gain_sweep.py
python testers/test3_fxnlms_filter_length_sweep.py
python testers/test4_fxnlms_model_mismatch.py
```

These tests generate CSV files and plots for comparing stability, tail MSE, noise reduction, and weight trajectories.

---

## Output files

Typical generated outputs include:

- `.wav` files for listening comparisons;
- `.png` figures for time-domain signal comparison and filter-weight evolution;
- `metrics.txt` files containing SNR and residual-noise measurements;
- `.csv` files from parameter sweeps.

Generated files are placed under `results/` or inside the relevant `testers/` output folder.

---

## Adaptive filtering overview

### NLMS speech enhancement

The NLMS speech-enhancement stage uses a reference noise signal to estimate the noise inside the primary speech microphone signal:

```text
y(n) = w^T(n) x(n)
e(n) = d(n) - y(n)
```

where:

- `x(n)` is the reference noise vector;
- `d(n)` is the noisy microphone signal;
- `y(n)` is the estimated noise;
- `e(n)` is the enhanced speech output.

The normalized update is:

```text
w(n+1) = w(n) + μ e(n) x(n) / (ε + x^T(n)x(n))
```

### FxNLMS headphone ANC

In the headphone ANC case, the anti-noise signal passes through a secondary acoustic path before it reaches the ear. FxNLMS accounts for this by filtering the reference noise through an estimated secondary path before updating the adaptive filter.

The main idea is:

```text
reference noise → adaptive filter → headphone speaker → secondary path → anti-noise at ear
```

The adaptive filter is updated using the error microphone signal and the filtered reference signal.

---

## Known issue

In the current code, `experiment_2.py` and some tester scripts pass `p_hat=playback_path` into `fxnlms()`, but the `fxnlms()` function in `adaptive_filters.py` is defined without a `p_hat` keyword argument.

If you see an error like:

```text
TypeError: fxnlms() got an unexpected keyword argument 'p_hat'
```

update the call or the function signature so they match. For example, either:

1. remove `p_hat=playback_path` from the function call if it is not needed; or
2. add a playback-path argument to `fxnlms()` and use it consistently when computing the media/speech path to the ear.

---

## Suggested workflow

1. Edit parameters in `config.py`.
2. Run `experiment_1.py` to test speech microphone enhancement.
3. Run `experiment_2.py` to test headphone ANC.
4. Inspect generated metrics in `results/*/test_outputs/metrics.txt`.
5. Listen to generated WAV files in `results/*/generated_sounds/`.
6. Use the tester scripts to study how step size, reference gain, filter length, and secondary-path mismatch affect FxNLMS stability and noise reduction.

---

## Notes

- The project is intended as a simulation and educational DSP project, not a real-time ANC implementation.
- The acoustic paths are modeled using short FIR filters.
- The quality of cancellation depends strongly on the reference-noise correlation, step size, filter length, and secondary-path estimate.
- No license file is currently included. Add a license before distributing or reusing the project publicly.
