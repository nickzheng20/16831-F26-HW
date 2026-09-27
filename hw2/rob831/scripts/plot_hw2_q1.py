"""Plot the six HW2 CartPole experiments from their TensorBoard scalar logs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--output-dir', type=Path, default=root / 'results' / 'q1')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ('no_rtg_dsa', 'Full trajectory, no standardization', '#bd5b35', '-'),
        ('rtg_dsa', 'Reward-to-go, no standardization', '#25856c', '--'),
        ('rtg_na', 'Reward-to-go + standardization', '#4269bd', '-.'),
    ]
    summaries = []
    for group, batch_size in [('sb', 1500), ('lb', 6000)]:
        fig, ax = plt.subplots(figsize=(9, 5.3), layout='constrained')
        for suffix, label, color, linestyle in variants:
            exp_name = f'q1_{group}_{suffix}'
            matches = sorted(args.data_dir.glob(f'{exp_name}_CartPole-v0_*'))
            if len(matches) != 1:
                raise ValueError(f'Expected one run for {exp_name}, found {len(matches)}: {matches}')
            run = matches[0]
            events = EventAccumulator(str(run), size_guidance={'scalars': 0})
            events.Reload()
            points = events.Scalars('Eval_AverageReturn')
            steps = np.array([p.step for p in points])
            returns = np.array([p.value for p in points])
            if not len(points) or not np.isfinite(returns).all():
                raise ValueError(f'Missing or non-finite evaluation returns: {run}')
            ax.plot(steps, returns, label=label, color=color, linestyle=linestyle,
                    linewidth=1.6, alpha=0.9)
            first200 = steps[returns >= 200]
            summaries.append({
                'experiment': exp_name,
                'source_directory': run.name,
                'iterations': len(points),
                'last_iteration': int(steps[-1]),
                'final_eval_return': float(returns[-1]),
                'last20_mean_eval_return': float(returns[-20:].mean()),
                'last20_std_eval_return_across_iterations': float(returns[-20:].std()),
                'best_eval_return': float(returns.max()),
                'first_iteration_at_200': int(first200[0]) if len(first200) else '',
                'training_env_steps': int(events.Scalars('Train_EnvstepsSoFar')[-1].value),
                'elapsed_seconds': float(events.Scalars('TimeSinceStart')[-1].value),
            })
        ax.axhline(200, color='#999999', linestyle=':', linewidth=0.9, zorder=0)
        ax.set(title=f'CartPole-v0 | batch size {batch_size:,}',
               xlabel='Training iteration (zero-based)', ylabel='Average evaluation return',
               xlim=(0, 149), ylim=(0, 210))
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        fig.text(0.5, -0.015, 'Raw evaluation curves; one run per configuration.',
                 ha='center', fontsize=9, color='#555555')
        for extension in ['png', 'pdf']:
            output = args.output_dir / f'q1_{group}.{extension}'
            fig.savefig(output, dpi=180, bbox_inches='tight')
            print(output)
        plt.close(fig)
    summary_path = args.output_dir / 'summary.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(summary_path)


if __name__ == '__main__':
    main()
