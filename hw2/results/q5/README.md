# Experiment 5: noisy Hopper with GAE

All four runs completed all 300 iterations with seed 1 and the assignment's
fixed hyperparameters: Hopper-v4, episode limit 1000, discount 0.99, two hidden
layers of width 32, batch size 2000, learning rate 0.001, reward-to-go, neural
network baseline, and action noise standard deviation 0.5. The only training
setting varied was GAE lambda. CPU execution used one thread per process.

| Lambda | Final evaluation return | Last 10 iterations, mean | Last 50 iterations, mean | Maximum logged evaluation return |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160.84 | 113.93 | 88.85 | 215.65 |
| 0.95 | 341.93 | 471.31 | 386.88 | 895.30 |
| 0.99 | 547.89 | 396.79 | 393.10 | 855.18 |
| 1 | 427.76 | 359.30 | 374.86 | 541.62 |

Lambda 0 performed worst: its one-step TD advantage depends heavily on the
learned value function and can therefore be biased by inaccurate predictions.
Lambda 0.95 and 0.99 learned faster than lambda 1 after roughly iteration 150
and reached returns near the assignment's target of 400, although their late
evaluation curves fluctuated substantially. Lambda 0.95 had the highest
last-10 mean, while lambda 0.99 had the highest last-50 mean; this ranking
depends on the averaging window. Lambda 1 also approached 400 late in training
but improved more slowly. It is equivalent to the Monte Carlo reward-to-go
minus value-baseline estimator when the rollout boundaries are treated as
terminal, as in this homework. Intermediate lambda values balance reliance on
the value estimate against a longer reward horizon; these four single-seed
runs do not establish which value would be best across random seeds.

`q5_hopper.pdf` and `q5_hopper.png` contain all raw logged evaluation returns,
with no smoothing or omitted outliers. `learning_curves.csv` contains all 1200
points. `summary.csv` identifies the four exact source TensorBoard directories
and records their metrics, environment-step counts, and elapsed training times.
The plot's "average evaluation return" at each iteration averages that
iteration's evaluation episodes; the last-10/50 summaries additionally average
those logged values across iterations, rather than reweighting by episode count.

The actual training configurations and environment setup are in `commands.sh`.
To reproduce only the plot, from the `hw2` directory run:

```bash
python rob831/scripts/plot_hw2_q5.py
```

For packaged logs, pass `--data-dir run_logs`. The plotting script validates
that there is one complete run per lambda with finite returns at iterations
0 through 299. Initial launch failures were caused by omitting the existing
conda MuJoCo activation settings, and are preserved in `startup_failure/`.
They produced no training iterations and are excluded from plots and summaries.
Activating the existing conda environment resolved the issue without changing
the algorithm, dependencies, or experiment hyperparameters.
