from pathlib import Path
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test_fxnlms_common import (
    make_fxnlms_case,
    run_case,
    tail_metrics,
    save_csv,
    plot_sweep_by_algorithm,
    algorithm_label,
)

from plotting import plot_weight_trajectory


def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "test4_fxnlms_model_mismatch"
    figures_dir = output_dir / "figures"
    weight_dir = figures_dir / "weight"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    case = make_fxnlms_case()

    model_gains = np.array([
        0.10,
        0.20,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        3.00,
        5.00,
    ])

    rows = []

    for normalize in [False, True]:
        algorithm = algorithm_label(normalize)

        for model_gain in model_gains:
            secondary_path_hat_test = model_gain * case.secondary_path

            result = run_case(
                case,
                normalize=normalize,
                secondary_path_hat=secondary_path_hat_test,
                reference_gain=1.0,
            )

            metrics = tail_metrics(result)

            rows.append({
                "algorithm": algorithm,
                "normalize": bool(normalize),
                "model_gain": float(model_gain),
                **metrics,
            })

            plot_weight_trajectory(
                t=case.t,
                result=result,
                figures_dir=weight_dir,
                fs=case.fs,
                start_sec=0,
                duration_sec=case.T,
                name=f"{algorithm}_model_gain_{model_gain:g}_",
            )

            print(
                f"{algorithm:6s}, "
                f"model_gain = {model_gain:.2f}, "
                f"tail MSE = {metrics['tail_mse']:.3e}, "
                f"reduction = {metrics['noise_reduction_db']:.2f} dB, "
                f"diverged = {metrics['diverged']}"
            )

    save_csv(rows, output_dir / "model_mismatch_sweep.csv")

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="model_gain",
        y_key="tail_mse",
        save_path=figures_dir / "model_gain_vs_tail_mse.png",
        title="FxLMS vs FxNLMS Secondary-Path Gain Mismatch",
        xlabel="Estimated secondary-path gain multiplier",
        ylabel="Tail MSE",
        log_x=True,
        log_y=True,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="model_gain",
        y_key="noise_reduction_db",
        save_path=figures_dir / "model_gain_vs_noise_reduction.png",
        title="FxLMS vs FxNLMS Secondary-Path Gain Mismatch",
        xlabel="Estimated secondary-path gain multiplier",
        ylabel="Noise reduction [dB]",
        log_x=True,
        log_y=False,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="model_gain",
        y_key="stable",
        save_path=figures_dir / "model_gain_vs_stability.png",
        title="FxLMS vs FxNLMS Stability Under Model-Gain Mismatch",
        xlabel="Estimated secondary-path gain multiplier",
        ylabel="Stable: 1 = stable, 0 = diverged",
        log_x=True,
        log_y=False,
    )


if __name__ == "__main__":
    main()