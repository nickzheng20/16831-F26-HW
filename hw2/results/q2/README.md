# Experiment 2 results

The selected run is `q2_b100_r0.03_InvertedPendulum-v4_29-09-2026_11-15-02`:
minimum batch target 100, learning rate 0.03, seed 1. Its average evaluation return
first reaches 1000 at zero-based iteration 65 (the 66th logged iteration), and its
final 10 returns are all 1000. All 100 raw evaluation points are plotted, without
smoothing. The selected run reaches 1000 in 15 of its 100 evaluations.

## Scope and selection

15 new configurations were run for 100 iterations each, supplementing the two
pre-existing b=500/1000, lr=0.01 runs. The new configurations are:

- b=50 and b=100, each with lr in {0.01, 0.02, 0.03, 0.04, 0.05} (10 runs).
- b=200, with lr in {0.01, 0.02, 0.03, 0.1} (4 runs).
- b=500, lr=0.05 (1 run).

The reporting preference is: among tested runs that reach 1000 before 100
iterations and have a last-10 mean return of at least 900, choose the smallest
batch, then the largest learning rate. This extra stability filter is our
selection preference; the assignment itself only explicitly requires reaching
1000 before 100 iterations and permits fluctuations. We do not claim a global
optimum or an exhaustive search. All runs use one seed, so neither ranking nor
stability is established across seeds.

The smaller b=50, lr=0.03 run also literally reaches the assignment target at
iteration 53, but falls to a final return of 106 and a last-10 mean of 444.65.
It is therefore a valid smaller *transient-hit* alternative, not a failure to
reach 1000. At b=100, lr=0.04 and 0.05 reach 1000 but have last-10 means 545.83
and 287.50. The selected b=100, lr=0.03 run gives the strongest final stability
among these smaller-batch/larger-rate candidates. See summary.csv for every
attempt, including all unsuccessful configurations.

Batch size is the minimum target of environment steps collected per iteration.
The code finishes full episodes, so the actual collected batch can be larger,
particularly when a trajectory lasts 1000 steps.

## Exact commands and runtime

`commands.sh` lists all 15 new successful launches. `run_metadata.json` in every
new data directory records the exact training command and defaults. Relative to
the assignment command, `--no_gpu` selects CPU, and OMP_NUM_THREADS,
MKL_NUM_THREADS, and OPENBLAS_NUM_THREADS are each set to 1. We preserve the
assignment environment, episode length, discount, iteration count, network size,
reward-to-go estimator, normalized advantages, and default seed=1 / evaluation
batch=400. No baseline is used for this experiment.

The local Conda environment must be activated before training, because its
activation hook supplies the MuJoCo/OSMesa system-library paths. Three initial
launches without this activation failed before collecting any evaluation data;
their raw console logs and data directories are preserved, and their names are
listed in startup_failures.txt. They are not counted as completed experiments.

## Reproduce figures

Run from hw2 with the homework Python environment:

```bash
python -m rob831.scripts.plot_hw2_q2 --data-dir data
# After packaging, use the same source logs under run_logs:
python -m rob831.scripts.plot_hw2_q2 --data-dir run_logs
```

The default selection comes from results/q2/selected.json. Override it with
`--selected FULL_RUN_DIRECTORY_NAME` if desired. The script checks that each
included run has all 100 finite evaluations at steps 0 through 99.

Outputs: q2_inverted_pendulum.pdf/png (selected raw learning curve),
q2_search.pdf/png (search matrix showing last-10 means and first target hit),
summary.csv (one row per complete run), learning_curves.csv (all raw curves).
