"""Train a few episodes and check whether the policy weights actually move,
whether the action distribution diverges from uniform, and whether the critic
learns the return scale. Distinguishes 'gradient bug' from 'exploration issue'."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.warehouse.config import WarehouseConfig
from src.warehouse.agv_env import WarehouseAGVEnv
from src.warehouse.ppo import MAPPO


def train_and_diagnose(episodes=100):
    cfg = WarehouseConfig(episodes=episodes)
    torch.manual_seed(cfg.seed)
    env = WarehouseAGVEnv(cfg)
    mappo = MAPPO(cfg)

    initial = {name: p.detach().clone() for name, p in mappo.policy.named_parameters()}

    for ep in range(episodes):
        env.reset()
        graph = env.get_graph()
        graphs, actions, logprobs, values, rewards = [], [], [], [], []
        done = False
        while not done:
            act, logp, val = mappo.act(graph)
            nxt, r, done, _ = env.step(act)
            graphs.append(graph)
            actions.append(torch.tensor(act, dtype=torch.long))
            logprobs.append(logp)
            values.append(val)
            rewards.append(r)
            graph = nxt
        stats = mappo.update(graphs, actions, logprobs, values, rewards)
        if ep % 10 == 0 or ep == episodes - 1:
            print(f'ep {ep:4d} | a_loss {stats["actor_loss"]:+.5f} | '
                  f'c_loss {stats["critic_loss"]:8.3f} | entropy {stats["entropy"]:.4f}')

    # weight movement
    total = 0.0
    for name, p in mappo.policy.named_parameters():
        total += (p.detach() - initial[name]).abs().sum().item()
    print(f'\nsum |delta weight| over all params: {total:.4f}')

    # post-train action distribution + value scale
    env.reset()
    graph = env.get_graph()
    acts_all, vals_all = [], []
    done = False
    while not done:
        act, logp, val = mappo.act(graph)
        acts_all.append(act)
        vals_all.append(val.item())
        graph, r, done, _ = env.step(act)

    acts = np.concatenate(acts_all)
    names = ['idle', 'local', 'link0', 'link1', 'link2']
    print('\npost-train action distribution:')
    for a in range(cfg.n_actions):
        print(f'  {names[a]:6s}: {(acts == a).mean():.3f}')
    print(f'post-train value mean/std: {np.mean(vals_all):.3f} {np.std(vals_all):.3f}')


if __name__ == '__main__':
    train_and_diagnose(100)
