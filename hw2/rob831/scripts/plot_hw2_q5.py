"""Plot the four prescribed HW2 noisy Hopper GAE runs from TensorBoard."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LAMBDAS = ("0", "0.95", "0.99", "1")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "q5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    summaries = []
    curve_rows = []
    colors = ("#888888", "#d55e00", "#0072b2", "#009e73")
    for lam, color in zip(LAMBDAS, colors):
        candidates = sorted(args.data_dir.glob(f"q5_b2000_r0.001_lambda{lam}_Hopper-v4_*"))
        complete = []
        for run in candidates:
            events = EventAccumulator(str(run), size_guidance={"scalars": 0})
            events.Reload()
            if "Eval_AverageReturn" not in events.Tags()["scalars"]:
                continue
            points = events.Scalars("Eval_AverageReturn")
            steps = np.array([point.step for point in points])
            values = np.array([point.value for point in points])
            if np.array_equal(steps, np.arange(300)) and np.isfinite(values).all():
                complete.append((run, events, steps, values))
        if len(complete) != 1:
            raise ValueError(f"Expected exactly one complete 300-iteration run for lambda={lam}; found {len(complete)}.")
        run, events, steps, values = complete[0]
        train_steps = events.Scalars("Train_EnvstepsSoFar")[-1].value
        seconds = events.Scalars("TimeSinceStart")[-1].value
        summaries.append({
            "gae_lambda": lam,
            "source_directory": run.name,
            "iterations": len(values),
            "final_eval_return": float(values[-1]),
            "last10_mean_eval_return": float(values[-10:].mean()),
            "last50_mean_eval_return": float(values[-50:].mean()),
            "best_eval_return": float(values.max()),
            "best_iteration": int(steps[values.argmax()]),
            "train_environment_steps": int(train_steps),
            "elapsed_seconds": float(seconds),
        })
        curve_rows.extend((lam, int(step), float(value)) for step, value in zip(steps, values))
        ax.plot(steps, values, color=color, linewidth=1.1, alpha=0.88, label=rf"$\lambda={lam}$")
    ax.axhline(400, color="#777777", linewidth=1, linestyle="--", label="Assignment target: 400")
    ax.set(title="Hopper-v4 with action noise | GAE comparison",
           xlabel="Training iteration (zero-based)", ylabel="Average evaluation return",
           xlim=(0, 299))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    fig.text(0.5, -0.015, "Raw evaluation curves; one seed (1) per setting, no smoothing.",
             ha="center", fontsize=9, color="#555555")
    for extension in ("png", "pdf"):
        output = args.output_dir / f"q5_hopper.{extension}"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(output)
    plt.close(fig)
    with (args.output_dir / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("gae_lambda", "iteration", "eval_average_return"))
        writer.writerows(curve_rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
