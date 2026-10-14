"""Training and evaluation loops for the warehouse offloading MAPPO."""
import numpy as np
import torch

from .agv_env import WarehouseAGVEnv
from .ppo import MAPPO


def evaluate_episode(env, policy_fn, greedy=False):
    """Run one episode with a policy function `policy_fn(graph, env) -> actions`."""
    env.reset()
    graph = env.get_graph()
    done = False
    while not done:
        actions = policy_fn(graph, env)
        graph, _, done, _ = env.step(actions)
    return env.metrics()


def make_mappo_policy(mappo, greedy=False):
    def policy(graph, env):
        actions, _, _ = mappo.act(graph, greedy=greedy)
        return actions
    return policy


def train(cfg, seed=None):
    """Train MAPPO for cfg.episodes episodes; return (mappo, history)."""
    torch.manual_seed(cfg.seed if seed is None else seed)
    env = WarehouseAGVEnv(cfg)
    mappo = MAPPO(cfg)
    history = []

    for ep in range(cfg.episodes):
        frac = ep / max(cfg.episodes - 1, 1)
        mappo.entropy_coef = cfg.entropy_coef_min + (cfg.entropy_coef - cfg.entropy_coef_min) * (1 - frac)

        env.reset()
        graph = env.get_graph()
        graphs, actions, logprobs, values, rewards = [], [], [], [], []

        done = False
        ep_reward = 0.0
        while not done:
            act, logp, val = mappo.act(graph)
            nxt_graph, reward, done, _ = env.step(act)

            graphs.append(graph)
            actions.append(torch.tensor(act, dtype=torch.long))
            logprobs.append(logp)
            values.append(val)
            rewards.append(reward)

            ep_reward += float(reward.sum())
            graph = nxt_graph

        stats = mappo.update(graphs, actions, logprobs, values, rewards)

        if ep % 10 == 0 or ep == cfg.episodes - 1:
            metrics = env.metrics()
            history.append({'episode': ep, 'reward': ep_reward, **metrics})
            print(f'ep {ep:4d} | reward {ep_reward:7.2f} | '
                  f'completion {metrics["completion_rate"]:.3f} | '
                  f'deadline {metrics["deadline_rate"]:.3f} | '
                  f'link_imb {metrics["link_imbalance"]:.3f} | '
                  f'a_loss {stats["actor_loss"]:.3f} | '
                  f'c_loss {stats["critic_loss"]:.3f}')

    return mappo, history


def evaluate_all(cfg, mappo, eval_episodes=20):
    """Evaluate MAPPO (greedy) and every baseline; return a dict of averaged metrics."""
    from .baselines import BASELINES

    policies = {'mappo': make_mappo_policy(mappo, greedy=True)}
    for name, fn in BASELINES.items():
        policies[name] = lambda graph, env, _fn=fn: _fn(env)

    results = {}
    for name, policy in policies.items():
        agg = {k: [] for k in
               ['completion_rate', 'deadline_rate', 'missed_rate',
                'avg_completion_time', 'link_imbalance', 'avg_rate', 'avg_sinr_db']}
        for _ in range(eval_episodes):
            env = WarehouseAGVEnv(cfg)
            m = evaluate_episode(env, policy)
            for k in agg:
                v = m[k]
                agg[k].append(v if v == v else 0.0)  # replace NaN with 0
        results[name] = {k: float(np.mean(v)) for k, v in agg.items()}
    return results
