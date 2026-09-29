"""Run the assignment's complete HalfCheetah grid and record exact commands."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/q4'
PYTHON = str((ROOT / '../.conda/rob831/bin/python').resolve())
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

def run(batch, rate):
    name = f'q4_search_b{batch}_lr{rate}_rtg_nnbaseline'
    args = [PYTHON, '-m', 'rob831.scripts.run_hw2', '--env_name', 'HalfCheetah-v4', '--ep_len', '150', '--discount', '0.95', '-n', '100', '-l', '2', '-s', '32', '-b', str(batch), '-lr', rate, '-rtg', '--nn_baseline', '--no_gpu', '--exp_name', name]
    record = {'command': shlex.join(args), 'batch': batch, 'learning_rate': float(rate), 'seed': 1, 'environment': {key: ENV[key] for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}, 'started_at': time.time()}
    metadata = OUTPUT / f'{name}.json'
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    print('START', name, flush=True)
    with (OUTPUT / f'{name}.log').open('w') as log:
        proc = subprocess.run(args, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=proc.returncode, finished_at=time.time())
    runs = sorted((ROOT / 'data').glob(f'{name}_HalfCheetah-v4_*'))
    record['run_directory'] = str(runs[-1].relative_to(ROOT)) if runs else None
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    if runs:
        (runs[-1] / 'run_metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    print('DONE', name, 'exit', proc.returncode, flush=True)
    return proc.returncode

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        tasks = [pool.submit(run, b, r) for b in [10000, 30000, 50000] for r in ['0.005', '0.01', '0.02']]
        statuses = [task.result() for task in tasks]
    sys.exit(any(statuses))
