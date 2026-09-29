"""Plot the HW2 LunarLander experiment from its TensorBoard scalars."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "q3")
    args = parser.parse_args()
    if args.run_dir is None:
        runs = sorted((root / "data").glob("q3_b10000_r0.005_LunarLanderContinuous-v2_*"))
        if len(runs) != 1:
            parser.error("Specify --run-dir when there is not exactly one matching Q3 run.")
        args.run_dir = runs[0]
    events = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    events.Reload()
    points = events.Scalars("Eval_AverageReturn")
    steps = np.array([p.step for p in points])
    returns = np.array([p.value for p in points])
    if not np.array_equal(steps, np.arange(100)) or not np.isfinite(returns).all():
        raise ValueError("Expected finite evaluation returns for all 100 iterations (0–99).")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "learning_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "eval_average_return"])
        writer.writerows(zip(steps, returns))
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    ax.plot(steps, returns, color="#4269bd", linewidth=1.6,
            label="Reward-to-go + neural network baseline")
    ax.axhline(120, color="#888888", linestyle="--", linewidth=1,
               label="Assignment target: 120")
    ax.set(title="LunarLanderContinuous-v2 | batch 10,000, learning rate 0.005",
           xlabel="Training iteration (zero-based)", ylabel="Average evaluation return",
           xlim=(0, 99))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.5, -0.015, "Raw evaluation curve; one run, no smoothing.",
             ha="center", fontsize=9, color="#555555")
    for extension in ("png", "pdf"):
        output = args.output_dir / f"q3_lunarlander.{extension}"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(output)
    plt.close(fig)
    summary = {
        "source_directory": args.run_dir.name,
        "iterations": len(points),
        "final_eval_return": float(returns[-1]),
        "last10_mean_eval_return": float(returns[-10:].mean()),
        "best_eval_return": float(returns.max()),
    }
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
