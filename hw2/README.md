## Setup

You can run this code on your own machine or on Google Colab.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. If you completed this installation for homework 1, you do not need to repeat it.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-S25-HW/blob/main/hw2/rob831/scripts/run_hw2.ipynb)

Local Install: Might need to install the swig tool
```bash
sudo apt-get update
sudo apt-get install -y swig g++ python3-dev

pip install -r requirements.txt
```

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with "TODO: get this from hw1".

- [infrastructure/rl_trainer.py](rob831/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](rob831/infrastructure/utils.py)
- [infrastructure/pytorch_util.py](rob831/infrastructure/pytorch_util.py)
- [infrastructure/replay_buffer.py](rob831/infrastructure/replay_buffer.py)
- [policies/MLP_policy.py](rob831/policies/MLP_policy.py)

You will then need to complete the following new files for homework 2. The relevant sections are marked with "TODO".
- [agents/pg_agent.py](rob831/agents/pg_agent.py)
- [policies/MLP_policy.py](rob831/policies/MLP_policy.py)

You will also want to look through [scripts/run_hw2.py](rob831/scripts/run_hw2.py) (if running locally) or [scripts/run_hw2.ipynb](rob831/scripts/run_hw2.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

You will be running your policy gradients implementation in five experiments total, investigating the effects of design decisions like reward-to-go estimators, neural network baselines and generalized advantage estimation for variance reduction, and advantage normalization. See the assignment PDF for more details.

## Reproduce the Experiment 3 figure

From the `hw2` directory, using the homework environment:

```bash
python -m rob831.scripts.plot_hw2_q3
```

This reads the existing `data/q3_b10000_r0.005_LunarLanderContinuous-v2_*`
TensorBoard run and writes the raw evaluation learning curve (PDF and PNG),
per-iteration values, and summary statistics to `results/q3/`.
To select a run explicitly (including after moving logs into `run_logs/`), use
`--run-dir path/to/run`. The figure uses all 100 iterations without smoothing.
The original run contains scalar logs, but no saved hyperparameter manifest.


## Reproduce the completed report

Run commands from `hw2` using the homework Conda environment. Its activation
hook supplies the local MuJoCo runtime and build-library paths; invoking the
Python executable directly without activating the environment may fail.

```bash
source /home/nick12138/anaconda3/etc/profile.d/conda.sh
conda activate /home/nick12138/projects/16831-F26-HW/.conda/rob831
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

New experiments use CPU (`--no_gpu`), default seed 1, default evaluation batch
size 400, and disabled video logging. Different configurations were run
concurrently, with one computation thread per process. Exact training commands
and per-run metadata are saved under `results/q2/`, `results/q4/`, `results/q5/`
and in the new TensorBoard run directories. Existing Q1/Q3 runs were retained.
All reported curves are raw evaluation averages, without smoothing or averaging
across seeds. The Q4 grid contains all nine batch-size/learning-rate combinations;
selection uses the highest mean evaluation return over the last ten iterations.

To regenerate figures from the submission ZIP, extract it and run:

```bash
python -m rob831.scripts.plot_hw2_q1 --data-dir run_logs
python -m rob831.scripts.plot_hw2_q2 --data-dir run_logs
python -m rob831.scripts.plot_hw2_q3 --run-dir run_logs/q3_b10000_r0.005_LunarLanderContinuous-v2_27-09-2026_20-23-37
python -m rob831.scripts.plot_hw2_q4 --data-dir run_logs
python -m rob831.scripts.plot_hw2_q5 --data-dir run_logs
latexmk -pdf -interaction=nonstopmode -halt-on-error hw2_submission.tex
```

In this working directory, omit the `--data-dir` options to read `data/`, and
omit Q3's `--run-dir`. Plot scripts produce PDF/PNG figures and CSV summaries
under `results/`. Report compilation requires LaTeX, minted, and Pygments;
`.latexmkrc` enables the shell escape required by minted. Answer-box and figure
heights from the supplied template are preserved.

After all runs finish, create the code/log submission with:

```bash
python -m rob831.scripts.package_hw2
```

The packager validates complete, finite evaluation logs, preserves original run
directory names, excludes startup failures and videos, and checks ZIP integrity
and the 15 MB limit. Upload `hw2_submission.pdf` to **HW2** and `submit.zip` to
**HW2 Code** on Gradescope. Nothing is uploaded automatically.


Experiment 2 distinguishes the smallest tested batch that ever reached 1000
(`b=50, lr=0.03`, first at iteration 53) from a more stable alternative
(`b=100, lr=0.03`, all last ten evaluations equal 1000). The report plots both;
`selected.json` records the stable alternative using an additional stability
criterion, not an extra assignment requirement. The comparison figure is
`results/q2/q2_comparison.pdf`.

Experiment 4 selects `b=10000, lr=0.02` by the last-ten mean of 155.60.
The peak is 210.90, but the final evaluation is 161.29: this seed does not
maintain 200 by the end. All nine search configurations and all four comparisons are retained.

The consolidated core commands are in `core_commands.sh`. This script reruns
the 32 new training jobs; use the figure-only commands above to reproduce the
report from the saved logs without retraining.
