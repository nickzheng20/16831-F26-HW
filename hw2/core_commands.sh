#!/usr/bin/env bash
# Core commands executed for this completion, grouped by experiment.
# Run from hw2 with the homework Conda environment activated.
# Independent jobs were launched concurrently; these loops express the same runs.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Q2: 15 new configurations, 100 iterations each.
for spec in '50 0.01' '50 0.02' '50 0.03' '50 0.04' '50 0.05' \
            '100 0.01' '100 0.02' '100 0.03' '100 0.04' '100 0.05' \
            '200 0.01' '200 0.02' '200 0.03' '200 0.1' '500 0.05'; do
  read -r b r <<< "$spec"
  python -m rob831.scripts.run_hw2 --env_name InvertedPendulum-v4 \
    --ep_len 1000 --discount 0.92 -n 100 -l 2 -s 64 \
    -b "$b" -lr "$r" -rtg --no_gpu --exp_name "q2_b${b}_r${r}"
done

# Q4: all 9 prescribed search combinations, 100 iterations each.
for b in 10000 30000 50000; do
  for r in 0.005 0.01 0.02; do
    python -m rob831.scripts.run_hw2 --env_name HalfCheetah-v4 \
      --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 \
      -b "$b" -lr "$r" -rtg --nn_baseline --no_gpu \
      --exp_name "q4_search_b${b}_lr${r}_rtg_nnbaseline"
  done
done

# Q4: four estimators at the pair selected by the last-10 mean.
for suffix in '' _rtg _nnbaseline _rtg_nnbaseline; do
  flags=()
  case "$suffix" in
    _rtg) flags=(-rtg) ;;
    _nnbaseline) flags=(--nn_baseline) ;;
    _rtg_nnbaseline) flags=(-rtg --nn_baseline) ;;
  esac
  python -m rob831.scripts.run_hw2 --env_name HalfCheetah-v4 \
    --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 \
    -b 10000 -lr 0.02 "${flags[@]}" --no_gpu \
    --exp_name "q4_b10000_r0.02${suffix}"
done

# Q5: the four prescribed GAE settings, 300 iterations each.
for lam in 0 0.95 0.99 1; do
  python -m rob831.scripts.run_hw2 --env_name Hopper-v4 --ep_len 1000 \
    --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
    --reward_to_go --nn_baseline --action_noise_std 0.5 \
    --gae_lambda "$lam" --no_gpu --exp_name "q5_b2000_r0.001_lambda${lam}"
done

# Rebuild new figures, report, and submission archive.
python -m rob831.scripts.plot_hw2_q2
python -m rob831.scripts.plot_hw2_q4
python -m rob831.scripts.plot_hw2_q5
latexmk -pdf -interaction=nonstopmode -halt-on-error hw2_submission.tex
python -m rob831.scripts.package_hw2
