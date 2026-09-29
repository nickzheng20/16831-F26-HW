"""Reproduce the HalfCheetah search and ablation figures from complete runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_run(data, name):
    matches = sorted(data.glob(f'{name}_HalfCheetah-v4_*'))
    complete = []
    for run in matches:
        events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
        if 'Eval_AverageReturn' not in events.Tags()['scalars']:
            continue
        points = events.Scalars('Eval_AverageReturn')
        x = np.array([p.step for p in points])
        y = np.array([p.value for p in points])
        if np.array_equal(x, np.arange(100)) and np.isfinite(y).all():
            complete.append((run, x, y))
    if len(complete) != 1:
        raise ValueError(f'Expected one complete run for {name}, got {len(complete)}')
    return complete[0]


def save_plot(output, stem, curves, title):
    fig, ax = plt.subplots(figsize=(9, 5.3), layout='constrained')
    for label, x, y, color, style in curves:
        ax.plot(x, y, label=label, color=color, linestyle=style, linewidth=1.2, alpha=.85)
    ax.axhline(200, color='#888888', linestyle=':', linewidth=.9)
    ax.set(title=title, xlabel='Training iteration (zero-based)', ylabel='Average evaluation return', xlim=(0,99))
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, ncol=3 if len(curves)>4 else 2, loc='upper left')
    fig.text(.5, -.02, 'Raw evaluation curves; seed 1, one run per configuration.', ha='center', fontsize=9, color='#555555')
    for ext in ['pdf', 'png']:
        fig.savefig(output/f'{stem}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=root/'data')
    p.add_argument('--output-dir', type=Path, default=root/'results/q4')
    p.add_argument('--search-only', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, curves, raw = [], [], []
    colors = ['#4269bd', '#25856c', '#bd5b35']
    for bi, b in enumerate([10000,30000,50000]):
        for ri, r in enumerate(['0.005','0.01','0.02']):
            name = f'q4_search_b{b}_lr{r}_rtg_nnbaseline'
            run, x, y = read_run(args.data_dir, name)
            rows.append(dict(experiment=name, source_directory=run.name, batch_size=b, learning_rate=float(r), iterations=len(y), final_eval_return=float(y[-1]), last10_mean_eval_return=float(y[-10:].mean()), best_eval_return=float(y.max())))
            curves.append((f'b={b:,}, lr={r}', x, y, colors[bi], ['-','--','-.'][ri]))
            raw.extend((name, int(step), float(value)) for step,value in zip(x,y))
    best = max(rows, key=lambda row: row['last10_mean_eval_return'])
    (args.output_dir/'best.json').write_text(json.dumps(dict(best, selection_rule='Highest mean evaluation return over the last 10 training iterations among the 9 grid configurations.'), indent=2)+'\n')
    save_plot(args.output_dir, 'q4_search', curves, 'HalfCheetah-v4 | reward-to-go + neural baseline: parameter search')
    best_curve = [curve for curve,row in zip(curves,rows) if row is best]
    save_plot(args.output_dir, 'q4_best', best_curve, 'HalfCheetah-v4 | selected configuration')
    if not args.search_only:
        b, r = best['batch_size'], str(best['learning_rate'])
        curves = []
        variants = [('', 'Full trajectory'), ('_rtg', 'Reward-to-go'), ('_nnbaseline', 'Full trajectory + baseline'), ('_rtg_nnbaseline', 'Reward-to-go + baseline')]
        for i,(suffix,label) in enumerate(variants):
            name = f'q4_b{b}_r{r}{suffix}'
            run,x,y = read_run(args.data_dir,name)
            rows.append(dict(experiment=name, source_directory=run.name, batch_size=b, learning_rate=float(r), iterations=len(y), final_eval_return=float(y[-1]), last10_mean_eval_return=float(y[-10:].mean()), best_eval_return=float(y.max())))
            curves.append((label,x,y,['#bd5b35','#4269bd','#9861bc','#25856c'][i],['-','--','-.',':'][i]))
            raw.extend((name,int(step),float(value)) for step,value in zip(x,y))
        save_plot(args.output_dir, 'q4_ablation', curves, f'HalfCheetah-v4 | batch {b:,}, learning rate {r}')
    with (args.output_dir/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir/'learning_curves.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['experiment','iteration','eval_average_return']); writer.writerows(raw)
    print(json.dumps(best,indent=2))

if __name__=='__main__':
    main()
