"""Export Experiment 2 scalar logs and a selected unsmoothed learning curve."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--selected", help="Full selected run-directory name; defaults to results/q2/selected.json")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "q2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.selected is None:
        args.selected = json.loads((root / "results/q2/selected.json").read_text())["run_directory"]
    rows, curves, startup_failures = [], [], []
    selected = None
    for run_dir in sorted(args.data_dir.glob("q2_*InvertedPendulum-v4_*")):
        match = re.match(r"q2_b(\d+)_r([\d.]+)_InvertedPendulum", run_dir.name)
        if match is None:
            continue
        event = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        event.Reload()
        if "Eval_AverageReturn" not in event.Tags()["scalars"]:
            startup_failures.append(run_dir.name)
            continue
        points = event.Scalars("Eval_AverageReturn")
        steps = np.array([p.step for p in points])
        values = np.array([p.value for p in points])
        if not np.array_equal(steps, np.arange(100)) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete or invalid run: {run_dir}")
        hits = steps[values >= 1000]
        rows.append({
            "run_directory": run_dir.name,
            "batch_size": int(match[1]), "learning_rate": float(match[2]),
            "iterations": len(points), "first_1000_iteration": int(hits[0]) if len(hits) else "",
            "best_eval_return": float(values.max()), "final_eval_return": float(values[-1]),
            "last10_mean_eval_return": float(values[-10:].mean()),
            "iterations_at_1000": len(hits), "selected": run_dir.name == args.selected,
        })
        curves.extend((run_dir.name, int(step), float(value)) for step, value in zip(steps, values))
        if run_dir.name == args.selected:
            selected = (steps, values, int(match[1]), float(match[2]))
    if selected is None:
        raise ValueError("Selected run not found")
    rows.sort(key=lambda r: (r["batch_size"], r["learning_rate"]))
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["run_directory", "iteration", "eval_average_return"]); writer.writerows(curves)
    (args.output_dir / "startup_failures.txt").write_text("\n".join(startup_failures) + "\n")
    steps, values, batch, rate = selected
    hit = steps[values >= 1000]
    if not len(hit):
        raise ValueError("Selected run does not reach the target")
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    ax.plot(steps, values, color="#4269bd", lw=1.6, label=f"b={batch}, lr={rate:g}, seed=1")
    ax.axhline(1000, color="#888888", ls="--", lw=1, label="Assignment optimum: 1000")
    ax.scatter([hit[0]], [1000], color="#d1495b", zorder=3)
    ax.annotate(f"First reaches 1000 at iteration {hit[0]}", (hit[0], 1000),
                xytext=(-240, -25), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": .8})
    ax.set(title="InvertedPendulum-v4 | reward-to-go, normalized advantages",
           xlabel="Training iteration (zero-based)", ylabel="Average evaluation return",
           xlim=(0, 99), ylim=(0, 1060))
    ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(.5, -.015, "Raw evaluation curve; one seed, no smoothing. Batch is a minimum rollout-step target.",
             ha="center", fontsize=9, color="#555555")
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"q2_inverted_pendulum.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # The assignment's smallest tested batch and the more stable alternative
    # are both shown, so stability is not mistaken for the literal hit criterion.
    successful = [row for row in rows if row["first_1000_iteration"] != ""]
    smallest = min(successful, key=lambda row: (row["batch_size"], -row["learning_rate"]))
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    compared = [smallest, next(row for row in rows if row["selected"])]
    seen = set()
    for row, color in zip(compared, ["#bd5b35", "#4269bd"]):
        name = row["run_directory"]
        if name in seen:
            continue
        seen.add(name)
        points = [(step, value) for run, step, value in curves if run == name]
        x, y = zip(*points)
        ax.plot(x, y, color=color, lw=1.3, alpha=.85,
                label=f"b={row['batch_size']}, lr={row['learning_rate']:g}")
    ax.axhline(1000, color="#888888", ls="--", lw=1, label="Assignment optimum: 1000")
    ax.set(title="InvertedPendulum-v4 | smallest tested successful batch and stable alternative",
           xlabel="Training iteration (zero-based)", ylabel="Average evaluation return",
           xlim=(0, 99), ylim=(0, 1060))
    ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(.5, -.015, "Raw evaluation curves; seed 1. Smallest refers only to tested settings.",
             ha="center", fontsize=9, color="#555555")
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"q2_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    batches = sorted({row["batch_size"] for row in rows})
    rates = sorted({row["learning_rate"] for row in rows})
    grid = np.full((len(batches), len(rates)), np.nan)
    cells = {}
    for row in rows:
        i, j = batches.index(row["batch_size"]), rates.index(row["learning_rate"])
        grid[i, j] = row["last10_mean_eval_return"]
        cells[i, j] = row
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(batches) * .6)), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(grid, vmin=0, vmax=1000, aspect="auto", cmap=cmap)
    for (i, j), row in cells.items():
        reached = row["first_1000_iteration"]
        label = f"{row['last10_mean_eval_return']:.0f}"
        label += f"\nfirst={reached}" if reached != "" else "\nnot reached"
        label += " *" if row["selected"] else ""
        ax.text(j, i, label, ha="center", va="center", fontsize=8,
                color="black" if grid[i,j] > 650 else "white")
    ax.set(xticks=range(len(rates)), xticklabels=[f"{r:g}" for r in rates],
           yticks=range(len(batches)), yticklabels=batches,
           xlabel="Learning rate", ylabel="Minimum batch-step target",
           title="InvertedPendulum-v4 search | seed=1, 100 iterations per run")
    fig.colorbar(im, ax=ax, label="Mean evaluation return over final 10 iterations")
    fig.text(.5, -.02, "Grey: untested. first: zero-based iteration first reaching 1000. * Selected run.",
             ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"q2_search.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(next(row for row in rows if row["selected"]))


if __name__ == "__main__":
    main()
