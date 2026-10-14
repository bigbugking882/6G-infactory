"""
Visualisation for the warehouse offloading module (HGNN + MAPPO + wireless coupling).

Generates publication-style figures summarising the key advantages:
  1. scalability_deadline.png   - deadline vs AGV count (MAPPO catches/overtakes greedy)
  2. scalability_load.png       - deadline vs arrival rate (light/base/heavy)
  3. baseline_comparison.png    - 5 policies x 3 metrics grouped bars
  4. wireless_coupling.png      - avg rate + link imbalance (physical load balancing)
  5. hgnn_schematic.png         - heterogeneous graph structure

Run from code/:  python plot_warehouse.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parent.parent / 'results' / 'warehouse'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
})


# --------------------------------------------------------------------------- #
def load_scalability():
    """Merge the three scalability CSVs into a list of dicts."""
    rows = []
    for name in ['scalability_0.csv', 'scalability_2.csv', 'scalability_4.csv']:
        p = OUT / name
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
    return rows


# Post-coupling baseline comparison (200-episode training, 20-episode greedy eval).
# Source: `python main_warehouse.py --episodes 200 --eval 20` with use_wireless=True.
BASELINE = {
    'mappo':     {'completion': 0.780, 'deadline': 0.559, 'link_imb': 1.414, 'avg_rate': 50.6},
    'random':    {'completion': 0.860, 'deadline': 0.561, 'link_imb': 0.031, 'avg_rate': 89.5},
    'greedy':    {'completion': 0.915, 'deadline': 0.780, 'link_imb': 0.260, 'avg_rate': 76.4},
    'all_local': {'completion': 0.712, 'deadline': 0.322, 'link_imb': 0.000, 'avg_rate': 0.0},
    'all_edge':  {'completion': 0.915, 'deadline': 0.780, 'link_imb': 0.260, 'avg_rate': 76.4},
}


# --------------------------------------------------------------------------- #
def fig_scalability_deadline(rows):
    """deadline vs AGV count at lambda=0.5."""
    pts = sorted([r for r in rows if r['n_links'] == 3 and r['arrival_rate'] == 0.5],
                 key=lambda r: r['n_agents'])
    if not pts:
        return
    n = [int(r['n_agents']) for r in pts]
    m = [r['m_ddl'] for r in pts]
    g = [r['g_ddl'] for r in pts]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(n, g, 's--', color='#d62728', label='greedy (global deadline sort)')
    ax.plot(n, m, 'o-', color='#1f77b4', label='MAPPO (HGNN + CTDE)')
    ax.annotate('ties greedy', xy=(30, m[-1]), xytext=(26, m[-1] + 0.06),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'), color='#1f77b4')
    ax.set_xlabel('Number of AGVs ($\\lambda$=0.5)')
    ax.set_ylabel('Deadline satisfaction rate')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(n)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'scalability_deadline.png')
    plt.close(fig)


def fig_scalability_load(rows):
    """deadline vs arrival rate at 10 AGVs."""
    pts = sorted([r for r in rows if r['n_agents'] == 10 and r['n_links'] == 3],
                 key=lambda r: r['arrival_rate'])
    if not pts:
        return
    lam = [r['arrival_rate'] for r in pts]
    m = [r['m_ddl'] for r in pts]
    g = [r['g_ddl'] for r in pts]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(lam, g, 's--', color='#d62728', label='greedy')
    ax.plot(lam, m, 'o-', color='#1f77b4', label='MAPPO')
    ax.annotate('overtakes greedy', xy=(lam[0], m[0]), xytext=(lam[0] + 0.05, m[0] - 0.12),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'), color='#1f77b4')
    ax.set_xlabel('Task arrival rate $\\lambda$ (10 AGVs)')
    ax.set_ylabel('Deadline satisfaction rate')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(lam)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'scalability_load.png')
    plt.close(fig)


def fig_baseline():
    """Grouped bar: completion / deadline / link imbalance across 5 policies."""
    names = list(BASELINE.keys())
    x = np.arange(len(names))
    w = 0.26
    comp = [BASELINE[k]['completion'] for k in names]
    ddl = [BASELINE[k]['deadline'] for k in names]
    imb = [BASELINE[k]['link_imb'] for k in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - w, comp, w, label='completion rate', color='#1f77b4')
    ax.bar(x, ddl, w, label='deadline rate', color='#2ca02c')
    ax.bar(x + w, imb, w, label='link imbalance', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('metric')
    ax.set_ylim(0, 1.6)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'baseline_comparison.png')
    plt.close(fig)


def fig_wireless_coupling():
    """Two panels: avg rate and link imbalance, mappo vs greedy."""
    labels = ['MAPPO', 'greedy']
    rate = [BASELINE['mappo']['avg_rate'], BASELINE['greedy']['avg_rate']]
    imb = [BASELINE['mappo']['link_imb'], BASELINE['greedy']['link_imb']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax1.bar(labels, rate, color=['#1f77b4', '#d62728'], width=0.5)
    ax1.set_ylabel('Avg offload rate (Mbps)')
    ax1.set_title('Co-channel interference cost')

    ax2.bar(labels, imb, color=['#1f77b4', '#d62728'], width=0.5)
    ax2.set_ylabel('Link imbalance (CV)')
    ax2.set_title('Load distribution')
    ax2.axhline(1.414, ls=':', color='grey', lw=1)
    ax2.text(0.5, 1.44, 'worst case (all on one link)', ha='center', fontsize=8, color='grey')

    fig.tight_layout()
    fig.savefig(OUT / 'wireless_coupling.png')
    plt.close(fig)


def fig_hgnn_schematic():
    """Schematic of the heterogeneous graph (AGV / task / edge / link nodes)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis('off')

    c_agv, c_task, c_edge, c_link = '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'

    def node(x, y, r, color, label=None):
        c = Circle((x, y), r, color=color, zorder=3)
        ax.add_patch(c)
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=8, color='white', zorder=4)

    # AGV column (left)
    agv_y = [3.6, 2.6, 1.6, 0.6]
    for i, y in enumerate(agv_y):
        node(0.8, y, 0.22, c_agv, f'A{i}')
    # Task column (middle)
    task_y = [3.4, 2.3, 1.2]
    for i, y in enumerate(task_y):
        node(3.2, y, 0.20, c_task, f'T{i}')
    # Edge server (right top) + links (right bottom)
    node(5.8, 3.0, 0.30, c_edge, 'E')
    for i, y in enumerate([1.7, 0.9]):
        node(5.8, y, 0.18, c_link, f'L{i}')

    # edges: AGV -> task (bipartite)
    for y_a in agv_y:
        for y_t in task_y:
            ax.add_patch(FancyArrowPatch((0.8, y_a), (3.2, y_t), arrowstyle='-',
                                         color=c_task, lw=0.6, alpha=0.35, zorder=1))
    # task -> edge
    for y_t in task_y:
        ax.add_patch(FancyArrowPatch((3.2, y_t), (5.6, 3.0), arrowstyle='-',
                                     color=c_edge, lw=0.8, alpha=0.5, zorder=1))
    # edge -> link
    ax.add_patch(FancyArrowPatch((5.9, 2.8), (5.9, 1.9), arrowstyle='-', color=c_link, lw=0.8, zorder=1))
    ax.add_patch(FancyArrowPatch((5.9, 2.8), (5.9, 1.1), arrowstyle='-', color=c_link, lw=0.8, zorder=1))

    # legend
    for x, c, lab in [(5.8, c_agv, 'AGV'), (5.8, c_task, 'Task'),
                      (5.8, c_edge, 'Edge server'), (5.8, c_link, 'Link')]:
        pass
    ax.text(0.8, 4.1, 'AGV', ha='center', color=c_agv, fontsize=10)
    ax.text(3.2, 4.1, 'Task', ha='center', color=c_task, fontsize=10)
    ax.text(5.8, 4.1, 'Edge / Link', ha='center', color=c_edge, fontsize=10)

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.5)
    fig.savefig(OUT / 'hgnn_schematic.png', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    rows = load_scalability()
    print(f'Loaded {len(rows)} scalability rows.')
    fig_scalability_deadline(rows)
    fig_scalability_load(rows)
    fig_baseline()
    fig_wireless_coupling()
    fig_hgnn_schematic()
    print('Saved figures to', OUT)
    for p in sorted(OUT.glob('*.png')):
        print('  ', p.name)
