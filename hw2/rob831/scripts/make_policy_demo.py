"""Train an extra CartPole demo and export actual before/after policy rollouts."""
import argparse
from contextlib import redirect_stdout
import copy
import json
import os
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from rob831.infrastructure import pytorch_util as ptu
from rob831.scripts.run_hw2 import PG_Trainer


def rollout(policy, seed, capture=False):
    env = gym.make('CartPole-v0')
    records, frames = [], []
    total_reward = 0.0
    # Evaluations do not consume the training policy's random-number stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        env.seed(seed)
        obs = env.reset()
        try:
            for _ in range(200):
                if capture:
                    frames.append(Image.fromarray(env.render(mode='rgb_array')))
                with torch.no_grad():
                    distribution = policy(ptu.from_numpy(obs[None]))
                    probabilities = ptu.to_numpy(distribution.probs[0]).tolist()
                    action = int(distribution.sample().item())
                next_obs, reward, done, _ = env.step(action)
                records.append({'observation': obs.tolist(), 'action': action,
                                'probabilities': probabilities, 'reward': float(reward)})
                total_reward += reward
                obs = next_obs
                if done:
                    break
            if capture:
                frames.append(Image.fromarray(env.render(mode='rgb_array')))
        finally:
            env.close()
    return {'total_reward': total_reward, 'steps': records,
            'final_observation': obs.tolist()}, frames


def save_comparison(before_frames, after_frames, before, after, output):
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    length = max(len(before_frames), len(after_frames))
    for step in range(0, length, 2):
        canvas = Image.new('RGB', (1200, 468), '#f1f5f9')
        draw = ImageDraw.Draw(canvas)
        for col, (label, images, record) in enumerate([
                ('BEFORE TRAINING', before_frames, before),
                ('AFTER TRAINING', after_frames, after)]):
            canvas.paste(images[min(step, len(images)-1)], (col*600, 68))
            status = 'finished' if step >= len(record['steps']) else 'balancing'
            score = min(step, len(record['steps']))
            draw.text((col*600+20, 9), label, fill='#0f172a', font=font)
            draw.text((col*600+20, 36), f'Return: {score} / 200 | {status}', fill='#475569', font=font)
        frames.append(canvas)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=[40]*(len(frames)-1)+[1800], loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'results' / 'policy_demo')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'rollouts.json').exists():
        parser.error(f'{out} already contains a demo; choose a new --output-dir to preserve it')
    torch.set_num_threads(1)
    params = dict(env_name='CartPole-v0', exp_name='extra_policy_demo',
                  n_iter=args.iterations, reward_to_go=True, nn_baseline=False,
                  gae_lambda=None, dont_standardize_advantages=False,
                  batch_size=1500, eval_batch_size=400, train_batch_size=1500,
                  num_agent_train_steps_per_iter=1, discount=1., learning_rate=.005,
                  n_layers=2, size=64, ep_len=None, seed=1, no_gpu=True, which_gpu=0,
                  video_log_freq=-1, scalar_log_freq=1, save_params=False,
                  action_noise_std=0., logdir=str(out / 'training_logs'))
    with (out / 'training.log').open('w', buffering=1) as log, redirect_stdout(log):
        trainer = PG_Trainer(params)
        initial_policy = copy.deepcopy(trainer.rl_trainer.agent.actor)
        trainer.run_training_loop()
        final_policy = trainer.rl_trainer.agent.actor
        initial_policy.save(out / 'before.pt')
        final_policy.save(out / 'after.pt')
        trainer.rl_trainer.logger._summ_writer.close()
        trainer.rl_trainer.env.close()
    seeds = [101, 102, 103]
    before, before_frames = rollout(initial_policy, seeds[0], capture=True)
    after, after_frames = rollout(final_policy, seeds[0], capture=True)
    before['label'], after['label'] = '训练前', '训练后'
    before_returns = [before['total_reward']] + [rollout(initial_policy, s)[0]['total_reward'] for s in seeds[1:]]
    after_returns = [after['total_reward']] + [rollout(final_policy, s)[0]['total_reward'] for s in seeds[1:]]
    payload = {
        'env_name': 'CartPole-v0',
        'note': '使用当前作业算法额外训练的演示，原来的正式实验只保存了分数日志；这里不是旧实验的录像回放。',
        'training': {'iterations': args.iterations, 'batch_size': 1500, 'seed': 1,
                     'reward_to_go': True, 'standardize_advantages': True,
                     'nn_baseline': False, 'gamma': 1., 'learning_rate': .005,
                     'n_layers': 2, 'size': 64, 'device': 'cpu'},
        'before': before, 'after': after,
        'evaluation': {'seeds': seeds, 'before_returns': before_returns,
                       'after_returns': after_returns},
    }
    (out / 'rollouts.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    save_comparison(before_frames, after_frames, before, after, out / 'before_after.gif')
    print(json.dumps({'before_returns': before_returns, 'after_returns': after_returns,
                      'output_directory': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
