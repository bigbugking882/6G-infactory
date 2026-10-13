"""
Entry point for the warehouse AGV computing-offloading scenario.

Trains a MAPPO policy (heterogeneous GNN encoder + centralised critic) and compares it
against heuristic baselines (random / greedy / all-local / all-edge).

Run from the code/ directory:
    python main_warehouse.py --episodes 200 --eval 20
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.warehouse.config import WarehouseConfig
from src.warehouse.runner import train, evaluate_all


def plot_learning_curve(history, out_dir):
    eps = [h['episode'] for h in history]
    plt.figure(figsize=(8, 3.5))
    plt.plot(eps, [h['completion_rate'] for h in history], label='completion rate')
    plt.plot(eps, [h['deadline_rate'] for h in history], label='deadline rate')
    plt.plot(eps, [h['link_imbalance'] for h in history], label='link imbalance')
    plt.xlabel('episode')
    plt.ylabel('metric')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'learning_curve.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved learning curve: {path}')


def plot_comparison(results, out_dir):
    names = list(results.keys())
    completion = [results[n]['completion_rate'] for n in names]
    deadline = [results[n]['deadline_rate'] for n in names]
    imbalance = [results[n]['link_imbalance'] for n in names]

    x = np.arange(len(names))
    w = 0.28
    plt.figure(figsize=(9, 4))
    plt.bar(x - w, completion, w, label='completion rate')
    plt.bar(x, deadline, w, label='deadline rate')
    plt.bar(x + w, imbalance, w, label='link imbalance')
    plt.xticks(x, names)
    plt.ylabel('metric')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    path = out_dir / 'comparison.png'
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved comparison: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--eval', type=int, default=20)
    parser.add_argument('--n-agents', type=int, default=10)
    args = parser.parse_args()

    cfg = WarehouseConfig(episodes=args.episodes, n_agents=args.n_agents)
    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'warehouse'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Warehouse scenario: {cfg.n_agents} AGVs, {cfg.n_links} links, '
          f'{cfg.horizon} steps/episode, {cfg.episodes} episodes')
    print(f'Action space: {cfg.n_actions} (0 idle, 1 local, 2..{cfg.n_links + 1} offload link)')

    mappo, history = train(cfg)

    print('\n--- Evaluating against baselines ---')
    results = evaluate_all(cfg, mappo, eval_episodes=args.eval)

    print('\n===== Comparison (averaged over {} episodes) ====='.format(args.eval))
    header = f'{"policy":<12} {"completion":>12} {"deadline":>12} {"missed":>12} {"avg_time":>12} {"link_imb":>12}'
    print(header)
    for name, m in results.items():
        avg_time = m['avg_completion_time']
        print(f'{name:<12} {m["completion_rate"]:>12.3f} {m["deadline_rate"]:>12.3f} '
              f'{m["missed_rate"]:>12.3f} {avg_time:>12.2f} {m["link_imbalance"]:>12.3f}')

    plot_learning_curve(history, out_dir)
    plot_comparison(results, out_dir)
    print(f'\nResults saved to {out_dir}')


if __name__ == '__main__':
    main()
