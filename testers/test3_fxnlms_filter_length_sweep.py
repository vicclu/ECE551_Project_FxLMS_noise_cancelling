from pathlib import Path
import sys
import numpy as np

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
    output_dir = script_dir / "test3_fxnlms_filter_length_sweep"
    figures_dir = output_dir / "figures"
    weight_dir = figures_dir / "weight"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    case = make_fxnlms_case()

    filter_lengths = np.array([
        1,
        2,
        3,
        4,
        5,
        8,
        10,
        15,
        20,
        32,
    ])

    rows = []

    for normalize in [False, True]:
        algorithm = algorithm_label(normalize)

        for M in filter_lengths:
            result = run_case(
                case,
                w_filter_len=int(M),
                normalize=normalize,
                reference_gain=1.0,
            )

            metrics = tail_metrics(result)

            row = {
                "algorithm": algorithm,
                "normalize": bool(normalize),
                "filter_length": int(M),
                **metrics,
            }

            rows.append(row)

            plot_weight_trajectory(
                t=case.t,
                result=result,
                figures_dir=weight_dir,
                fs=case.fs,
                start_sec=0,
                duration_sec=case.T,
                name=f"{algorithm}_M_{M}_",
            )

            print(
                f"{algorithm:6s}, "
                f"M = {M:2d}, "
                f"tail MSE = {metrics['tail_mse']:.6e}, "
                f"noise reduction = {metrics['noise_reduction_db']:.2f} dB, "
                f"diverged = {metrics['diverged']}"
            )

    save_csv(rows, output_dir / "filter_length_sweep.csv")

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="filter_length",
        y_key="tail_mse",
        save_path=figures_dir / "filter_length_vs_tail_mse.png",
        title="FxLMS vs FxNLMS Controller-Length Sweep",
        xlabel="Controller length M",
        ylabel="Tail MSE",
        log_x=False,
        log_y=True,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="filter_length",
        y_key="noise_reduction_db",
        save_path=figures_dir / "filter_length_vs_noise_reduction.png",
        title="FxLMS vs FxNLMS Controller-Length Sweep",
        xlabel="Controller length M",
        ylabel="Noise reduction [dB]",
        log_x=False,
        log_y=False,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="filter_length",
        y_key="stable",
        save_path=figures_dir / "filter_length_vs_stability.png",
        title="FxLMS vs FxNLMS Stability Across Controller Length",
        xlabel="Controller length M",
        ylabel="Stable: 1 = stable, 0 = diverged",
        log_x=False,
        log_y=False,
    )


if __name__ == "__main__":
    main()