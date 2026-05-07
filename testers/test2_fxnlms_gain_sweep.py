from pathlib import Path
import numpy as np

from test_fxnlms_common import (
    make_fxnlms_case,
    run_case,
    tail_metrics,
    save_csv,
    plot_sweep_by_algorithm,
    algorithm_label
)


import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plotting import(
    plot_weight_trajectory
)


def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "test2_fxnlms_gain_sweep"
    figures_dir = output_dir / "figures"
    weight_dir = figures_dir / "weight"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)


    case = make_fxnlms_case()

    gain_values = np.array([
        0.1,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        30.0,
    ])

    rows = []

    for normalize in [False, True]:
        algorithm = algorithm_label(normalize)

        for gain in gain_values:
            result = run_case(
                case,
                normalize=normalize,
                reference_gain=gain,
            )

            metrics = tail_metrics(result)

            rows.append({
                "algorithm": algorithm,
                "normalize": normalize,
                "gain": float(gain),
                **metrics,
            })
            plot_weight_trajectory(
                t=case.t,
                result=result,
                figures_dir=weight_dir,
                fs=case.fs,
                start_sec=0,
                duration_sec=case.T,
                name=f"{algorithm}_gain_{gain:g}_",
            )

            print(
                f"{algorithm:6s}, "
                f"gain = {gain:.2f}, "
                f"tail MSE = {metrics['tail_mse']:.3e}, "
                f"reduction = {metrics['noise_reduction_db']:.2f} dB, "
                f"diverged = {metrics['diverged']}"
            )

    save_csv(rows, output_dir / "gain_sweep.csv")

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="gain",
        y_key="tail_mse",
        save_path=figures_dir / "gain_vs_tail_mse.png",
        title="FxLMS vs FxNLMS Gain Sweep",
        xlabel="Reference / noise gain",
        ylabel="Tail MSE",
        log_x=True,
        log_y=True,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="gain",
        y_key="noise_reduction_db",
        save_path=figures_dir / "gain_vs_noise_reduction.png",
        title="FxLMS vs FxNLMS gain Sweep",
        xlabel="gain size",
        ylabel="Noise reduction [dB]",
        log_x=True,
        log_y=False,
    )

if __name__ == "__main__":
    main()