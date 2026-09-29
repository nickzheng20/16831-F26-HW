# HalfCheetah experiments

All 9 prescribed grid configurations and all 4 estimator comparisons completed
100 iterations, seed 1. Other learning parameters match the assignment;
`--no_gpu` and single-thread CPU execution were used. `commands.sh` lists the
training commands; per-run JSON records contain the exact command and source log.

The selected pair is batch 10000, learning rate 0.02, maximizing the last-10 mean
of 155.60 over the grid. Its final return is 161.29 and peak is 210.90. The run
reaches the neighborhood of 200, but does not maintain it at the end. No algorithm
or reward changes were made to meet the target.

The four comparisons have last-10 means:

| Estimator | Mean |
|---|---:|
| Full trajectory | -74.79 |
| Reward-to-go | 44.50 |
| Full trajectory + baseline | -92.78 |
| Reward-to-go + baseline | 155.60 |

The combined estimator performs best here. A baseline alone does not improve
this full-trajectory run. These are single-seed observations. The combined
comparison run exactly reproduces the selected search curve (same parameters
and seed), so it is not an independent replicate.

Reproduce the three figures with:

```bash
python -m rob831.scripts.plot_hw2_q4 --data-dir run_logs
```

`q4_search.pdf` shows the full grid, `q4_best.pdf` the selected grid run, and
`q4_ablation.pdf` the four comparison runs. CSV files retain every raw evaluation
point and all summary metrics. Startup failures are excluded from the submitted
runs because they contain no training scalars; their diagnostics remain local.
