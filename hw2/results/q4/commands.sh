#!/usr/bin/env bash
# Run from hw2 with the homework Conda environment activated.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Default seed=1, eval_batch_size=400, video_log_freq=-1.
# The actual search was executed concurrently by run_search.py.
for b in 10000 30000 50000; do
  for r in 0.005 0.01 0.02; do
    python -m rob831.scripts.run_hw2 --env_name HalfCheetah-v4 \
      --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 \
      -b "$b" -lr "$r" -rtg --nn_baseline --no_gpu \
      --exp_name "q4_search_b${b}_lr${r}_rtg_nnbaseline"
  done
done
python -m rob831.scripts.plot_hw2_q4 --search-only

# Selected pair: highest last-10 mean among all nine complete search runs.
for suffix in "" _rtg _nnbaseline _rtg_nnbaseline; do
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
python -m rob831.scripts.plot_hw2_q4
