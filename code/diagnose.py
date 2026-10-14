"""Diagnose why MAPPO's actor loss stays ~0: inspect action distribution,
reward sparsity, advantage scale and value scale over one rollout."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.warehouse.config import WarehouseConfig
from src.warehouse.agv_env import WarehouseAGVEnv
from src.warehouse.ppo import MAPPO


def rollout(cfg):
    torch.manual_seed(cfg.seed)
    env = WarehouseAGVEnv(cfg)
    mappo = MAPPO(cfg)

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
    return mappo, graphs, actions, logprobs, values, rewards, env


def main():
    cfg = WarehouseConfig()
    mappo, graphs, actions, logprobs, values, rewards, env = rollout(cfg)

    N = cfg.n_agents
    n_steps = len(graphs)
    print(f'steps={n_steps}, agents={N}, total decisions={n_steps * N}')

    # 1) action distribution
    acts = np.concatenate([a.numpy() for a in actions])
    names = ['idle', 'local'] + [f'link{k}' for k in range(cfg.n_links)]
    print('\n-- action distribution --')
    for a in range(cfg.n_actions):
        print(f'  {names[a]:6s} ({a}): {int((acts == a).sum()):5d}  '
              f'{(acts == a).mean():.3f}')

    # 2) reward sparsity
    r_t = torch.tensor(np.stack(rewards), dtype=torch.float32)
    nz = int((r_t != 0).sum().item())
    print(f'\n-- reward sparsity --')
    print(f'  nonzero entries: {nz}/{r_t.numel()} ({nz / r_t.numel():.3f})')
    print(f'  reward mean/std/min/max: {r_t.mean():.3f} {r_t.std():.3f} '
          f'{r_t.min():.3f} {r_t.max():.3f}')

    # 3) value scale
    val_t = torch.stack(values)
    print(f'\n-- value (critic) scale --')
    print(f'  value mean/std/min/max: {val_t.mean():.3f} {val_t.std():.3f} '
          f'{val_t.min():.3f} {val_t.max():.3f}')

    # 4) logprob scale
    lp = torch.stack(logprobs)
    print(f'\n-- logprob scale --')
    print(f'  logprob mean/std: {lp.mean():.3f} {lp.std():.3f}')
    print(f'  implied entropy upper bound ln(5)={np.log(cfg.n_actions):.3f}')

    # 5) GAE advantage (reproduce ppo.update)
    gamma, lam = cfg.gamma, cfg.lam
    old_values = torch.stack(values, dim=0)
    advantages = torch.zeros_like(r_t)
    gae = torch.zeros(N)
    for s in reversed(range(n_steps)):
        nv = old_values[s + 1] if s + 1 < n_steps else 0.0
        delta = r_t[s] + gamma * nv - old_values[s]
        gae = delta + gamma * lam * gae
        advantages[s] = gae
    print(f'\n-- GAE advantage (raw) --')
    print(f'  mean/std/min/max: {advantages.mean():.3f} {advantages.std():.3f} '
          f'{advantages.min():.3f} {advantages.max():.3f}')
    print(f'  nonzero: {int((advantages != 0).sum())}/{advantages.numel()}')

    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    print(f'\n-- GAE advantage (normalized) --')
    print(f'  mean/std/min/max: {adv_norm.mean():.3f} {adv_norm.std():.3f} '
          f'{adv_norm.min():.3f} {adv_norm.max():.3f}')

    # 6) one-step PPO actor loss at current (untrained) policy
    ratio = torch.ones_like(advantages)
    adv = adv_norm
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip) * adv
    actor_loss = -torch.min(surr1, surr2).mean()
    print(f'\n-- actor loss with ratio=1 (untrained) --')
    print(f'  actor_loss = {actor_loss.item():.5f}  (should be ~0 because adv mean=0)')


if __name__ == '__main__':
    main()
