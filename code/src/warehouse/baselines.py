"""Heuristic baselines for the warehouse offloading problem.

Each baseline returns an action vector (shape n_agents) for the current step.
Actions follow the same convention as the environment:
  0 idle, 1 local, 2..n_links+1 offload via link (k-2).
"""
import numpy as np


def random_actions(env):
    rng = env.rng
    actions = np.zeros(env.n_agents, dtype=int)
    idle = env._idle_agents()
    actions[idle] = rng.randint(0, env.n_actions, size=len(idle))
    return actions


def _assign_nearest(env, mode_fn):
    """Global greedy assignment: deadline-sorted tasks to nearest idle AGV."""
    actions = np.zeros(env.n_agents, dtype=int)
    idle = set(env._idle_agents().tolist())
    pending = sorted(env._pending_tasks(), key=lambda k: env.tasks[k]['deadline'])
    for tgt in pending:
        if not idle:
            break
        task = env.tasks[tgt]
        best_i, best_d = None, np.inf
        for i in idle:
            d = np.linalg.norm(task['pos'] - env.agv_pos[i])
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            continue
        idle.remove(best_i)
        actions[best_i] = mode_fn(env, best_i, task)
    return actions


def greedy_actions(env):
    """Offload when edge (best case) is faster than local; otherwise local."""
    best_bw = max(env.link_bw)
    f_edge = env.cfg.f_edge

    def mode(env, i, task):
        local = task['C'] / env.agv_cpu[i]
        edge = task['D'] / best_bw + task['C'] / f_edge
        if edge < local:
            m = int(np.argmin(env.link_free))
            return 2 + m
        return 1

    return _assign_nearest(env, mode)


def all_local_actions(env):
    def mode(env, i, task):
        return 1

    return _assign_nearest(env, mode)


def all_edge_actions(env):
    def mode(env, i, task):
        m = int(np.argmin(env.link_free))
        return 2 + m

    return _assign_nearest(env, mode)


BASELINES = {
    'random': random_actions,
    'greedy': greedy_actions,
    'all_local': all_local_actions,
    'all_edge': all_edge_actions,
}
