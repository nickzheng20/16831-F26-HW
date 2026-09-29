#!/usr/bin/env bash
# Run from the hw2 directory. Conda activation supplies local MuJoCo libraries.
set -euo pipefail
source /home/nick12138/anaconda3/etc/profile.d/conda.sh
conda activate /home/nick12138/projects/16831-F26-HW/.conda/rob831
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The four commands below were launched concurrently with stdout/stderr in
# results/q5/train_lambda<LAMBDA>.log. All unspecified arguments use defaults,
# including seed=1, eval_batch_size=400, and video_log_freq=-1.
for lam in 0 0.95 0.99 1; do
  python -u rob831/scripts/run_hw2.py \
    --env_name Hopper-v4 --ep_len 1000 \
    --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
    --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda "$lam" \
    --exp_name "q5_b2000_r0.001_lambda${lam}" --no_gpu \
    > "results/q5/train_lambda${lam}.log" 2>&1 &
done
wait
python rob831/scripts/plot_hw2_q5.py
