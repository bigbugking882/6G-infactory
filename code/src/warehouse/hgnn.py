"""
Hand-written heterogeneous graph neural network (relational-GCN style) for the warehouse
scenario. Node types: AGV / task / edge-server / link. Message passing is done with a
per-relation linear map and mean aggregation; no external GNN library is required.
"""
import torch
import torch.nn as nn

FEAT_DIMS = {'agv': 5, 'task': 6, 'edge': 4, 'link': 3}
RELATIONS = [
    ('agv', 'task', 'agv_task'),
    ('task', 'agv', 'task_agv'),
    ('task', 'edge', 'task_edge'),
    ('edge', 'task', 'edge_task'),
    ('edge', 'link', 'edge_link'),
    ('link', 'edge', 'link_edge'),
]


class HeteroGNN(nn.Module):
    def __init__(self, hidden=64, layers=2, n_links=3):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.n_links = n_links

        self.emb = nn.ModuleDict({
            t: nn.Linear(FEAT_DIMS[t], hidden) for t in FEAT_DIMS
        })

        self.rel_w = nn.ModuleList()
        self.upd = nn.ModuleList()
        for _ in range(layers):
            self.rel_w.append(nn.ModuleDict({
                r: nn.Linear(hidden, hidden, bias=False) for (_, _, r) in RELATIONS
            }))
            self.upd.append(nn.ModuleDict({
                t: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
                for t in FEAT_DIMS
            }))

        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
        )

    def _to(self, x, device):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def forward(self, graph):
        device = next(self.parameters()).device
        edge_index = graph['edge_index']
        n_task = int(graph['n_task'])

        h = {
            'agv': self.emb['agv'](self._to(graph['agv'], device)),
            'task': self.emb['task'](self._to(graph['task'], device)),
            'edge': self.emb['edge'](self._to(graph['edge'], device)),
            'link': self.emb['link'](self._to(graph['link'], device)),
        }
        # zero out padded dummy task nodes so they don't leak into pooling
        if h['task'].shape[0] > n_task:
            h['task'][n_task:] = 0.0

        for l in range(self.layers):
            acc = {t: torch.zeros_like(h[t]) for t in h}
            cnt = {t: torch.zeros((h[t].shape[0], 1), device=device) for t in h}
            for (st, dt, rel) in RELATIONS:
                ei = edge_index.get(rel)
                if ei is None or ei.shape[1] == 0:
                    continue
                src = torch.as_tensor(ei[0], dtype=torch.long, device=device)
                dst = torch.as_tensor(ei[1], dtype=torch.long, device=device)
                msg = self.rel_w[l][rel](h[st][src])
                acc[dt].index_add_(0, dst, msg)
                cnt[dt].index_add_(0, dst, torch.ones(msg.shape[0], 1, device=device))

            for t in h:
                h[t] = h[t] + self.upd[l][t](acc[t] / (cnt[t] + 1e-6))

        # global state: mean-pool each node type, concatenate
        task_mean = h['task'][:n_task].mean(0) if n_task > 0 else torch.zeros(self.hidden, device=device)
        g = torch.cat([h['agv'].mean(0), task_mean, h['edge'].mean(0), h['link'].mean(0)], dim=0)
        g = self.global_proj(g)

        return h['agv'], g
