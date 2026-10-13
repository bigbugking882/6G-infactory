"""Scalability study: sweep AGV count, link count and task arrival rate,
training MAPPO against baselines for each configuration and writing a summary CSV.

Supports batching via --start/--num so long sweeps can be split across processes;
each batch writes its own CSV (scalability_<start>.csv).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.warehouse.config import WarehouseConfig
from src.warehouse.runner import train, evaluate_all


def link_bw_for(n_links):
    """Heterogeneous links; fastest fixed at 80 Mbps, slower links scale geometrically."""
    return [80e6 / (2 ** (n_links - 1 - i)) for i in range(n_links)]


CONFIGS = [
    (10, 3, 0.5),   # baseline
    (20, 3, 0.5),   # more AGVs
    (30, 3, 0.5),   # even more AGVs
    (10, 3, 0.3),   # light load
    (10, 3, 0.8),   # heavy load
    (10, 5, 0.5),   # more links
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--eval', type=int, default=20)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num', type=int, default=len(CONFIGS))
    args = parser.parse_args()

    selected = CONFIGS[args.start:args.start + args.num]

    out_dir = Path(__file__).resolve().parent.parent / 'results' / 'warehouse'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'scalability_{args.start}.csv'

    rows = []
    for n_agents, n_links, arrival_rate in selected:
        cfg = WarehouseConfig(
            n_agents=n_agents,
            n_links=n_links,
            link_bw=link_bw_for(n_links),
            arrival_rate=arrival_rate,
            episodes=args.episodes,
            seed=123,
        )
        mappo, _ = train(cfg)
        res = evaluate_all(cfg, mappo, args.eval)
        m, g = res['mappo'], res['greedy']

        rows.append({
            'n_agents': n_agents, 'n_links': n_links, 'arrival_rate': arrival_rate,
            'm_comp': round(m['completion_rate'], 4),
            'm_ddl': round(m['deadline_rate'], 4),
            'm_imb': round(m['link_imbalance'], 4),
            'g_comp': round(g['completion_rate'], 4),
            'g_ddl': round(g['deadline_rate'], 4),
            'g_imb': round(g['link_imbalance'], 4),
        })
        print(f"\n[{n_agents}a/{n_links}l/lam={arrival_rate}] "
              f"MAPPO comp={m['completion_rate']:.3f} ddl={m['deadline_rate']:.3f} "
              f"imb={m['link_imbalance']:.3f} | "
              f"greedy comp={g['completion_rate']:.3f} ddl={g['deadline_rate']:.3f} "
              f"imb={g['link_imbalance']:.3f}")

    hdr = (f"{'config':>16} | {'m_comp':>7} {'g_comp':>7} | {'m_ddl':>7} {'g_ddl':>7} | "
           f"{'m_imb':>7} {'g_imb':>7}")
    print('\n' + hdr)
    print('-' * len(hdr))
    for r in rows:
        cfg_s = f"{r['n_agents']}a/{r['n_links']}l/l{r['arrival_rate']}"
        print(f"{cfg_s:>16} | {r['m_comp']:>7.3f} {r['g_comp']:>7.3f} | "
              f"{r['m_ddl']:>7.3f} {r['g_ddl']:>7.3f} | "
              f"{r['m_imb']:>7.3f} {r['g_imb']:>7.3f}")

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved CSV: {csv_path}')


if __name__ == '__main__':
    main()
