from pathlib import Path
import numpy as np

from test_config import MU_FXLMS
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
    output_dir = script_dir / "test1_fxnlms_mu_sweep"
    figures_dir = output_dir / "figures"
    weight_dir = figures_dir / "weight"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)


    case = make_fxnlms_case()

    mu_values = np.array([
        0.0005,
        0.001,
        0.003,
        0.005,
        0.01,
        0.02,
        0.05,
    ])


    rows = []

    for normalize in [True, False]:
        algorithm = algorithm_label(normalize)

        for mu in mu_values:
            result = run_case(case, mu=mu)
            metrics = tail_metrics(result)

            row = {
                "algorithm": algorithm,
                "normalize": bool(normalize),
                "mu": float(mu),
                **metrics,
            }
            
            plot_weight_trajectory(
                t=case.t,
                result=result,
                figures_dir=weight_dir,
                fs=case.fs,
                start_sec=0,
                duration_sec=case.T,
                name=f"{algorithm}_mu_{mu:g}_",
            )
    
            rows.append(row)


            print(
                f"{algorithm:6s}, "
                f"mu = {mu:.5f}, "
                f"tail MSE = {metrics['tail_mse']:.3e}, "
                f"reduction = {metrics['noise_reduction_db']:.2f} dB, "
                f"diverged = {metrics['diverged']}"
            )

    save_csv(rows, output_dir / "mu_sweep.csv")

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="mu",
        y_key="tail_mse",
        save_path=figures_dir / "mu_vs_tail_mse.png",
        title="FxLMS vs FxNLMS Step-Size Sweep",
        xlabel="Step size μ",
        ylabel="Tail MSE",
        log_x=True,
        log_y=True,
    )

    plot_sweep_by_algorithm(
        rows=rows,
        x_key="mu",
        y_key="noise_reduction_db",
        save_path=figures_dir / "mu_vs_noise_reduction.png",
        title="FxLMS vs FxNLMS Step-Size Sweep",
        xlabel="Step size μ",
        ylabel="Noise reduction [dB]",
        log_x=True,
        log_y=False,
    )

if __name__ == "__main__":
    main()