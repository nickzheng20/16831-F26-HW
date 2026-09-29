"""Run the four HalfCheetah estimators with the completed grid's selected pair."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/q4'
PYTHON=str((ROOT/'../.conda/rob831/bin/python').resolve())
ENV=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')


def run(suffix, flags, batch, rate):
    name=f'q4_b{batch}_r{rate}{suffix}'
    args=[PYTHON,'-m','rob831.scripts.run_hw2','--env_name','HalfCheetah-v4','--ep_len','150','--discount','0.95','-n','100','-l','2','-s','32','-b',str(batch),'-lr',rate,*flags,'--no_gpu','--exp_name',name]
    record={'command':shlex.join(args),'batch':batch,'learning_rate':float(rate),'seed':1,'environment':{key:ENV[key] for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'started_at':time.time()}
    metadata=OUTPUT/f'{name}.json'
    metadata.write_text(json.dumps(record,indent=2)+'\n')
    print('START', name, flush=True)
    with (OUTPUT/f'{name}.log').open('w') as log:
        result=subprocess.run(args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode,finished_at=time.time())
    runs=sorted((ROOT/'data').glob(f'{name}_HalfCheetah-v4_*'))
    record['run_directory']=str(runs[-1].relative_to(ROOT)) if runs else None
    metadata.write_text(json.dumps(record,indent=2)+'\n')
    if runs:
        (runs[-1]/'run_metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    print('DONE',name,'exit',result.returncode,flush=True)
    return result.returncode


if __name__=='__main__':
    # The grid may still be running when this follow-on job is started.
    names=[f'q4_search_b{b}_lr{r}_rtg_nnbaseline' for b in [10000,30000,50000] for r in ['0.005','0.01','0.02']]
    print('Waiting for all nine search jobs to finish.',flush=True)
    while True:
        records=[json.loads((OUTPUT/f'{name}.json').read_text()) for name in names]
        failed=[row for row in records if row.get('returncode',0)!=0]
        if failed:
            raise RuntimeError(f'Search failed: {failed}')
        if all('returncode' in row for row in records):
            break
        time.sleep(15)
    subprocess.run([PYTHON,'-m','rob831.scripts.plot_hw2_q4','--search-only'],cwd=ROOT,env=ENV,check=True)
    best=json.loads((OUTPUT/'best.json').read_text())
    batch,rate=best['batch_size'],str(best['learning_rate'])
    variants=[('',[]),('_rtg',['-rtg']),('_nnbaseline',['--nn_baseline']),('_rtg_nnbaseline',['-rtg','--nn_baseline'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run,suffix,flags,batch,rate) for suffix,flags in variants]
        statuses=[job.result() for job in jobs]
    if any(statuses):
        sys.exit(1)
    subprocess.run([PYTHON,'-m','rob831.scripts.plot_hw2_q4'],cwd=ROOT,env=ENV,check=True)
