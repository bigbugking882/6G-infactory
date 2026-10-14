"""
Warehouse logistics environment: multi-AGV picking with Wi-Fi 7 compute offloading.

AGVs move on a rectangular warehouse floor. Picking tasks arrive over time, each with a
shelf location, data size D (bits), compute demand C (cycles) and a deadline. Every AGV
has limited local CPU; it may process a task locally or offload it to a single edge server
through one of several Wi-Fi 7 links. The environment exposes a heterogeneous graph (AGV /
task / edge-server / link nodes) so an RL agent can learn the joint decision of
task assignment, execution location and link selection.

The environment is pure numpy; the GNN wrapper converts the graph to torch tensors.
"""
import numpy as np

from .config import WarehouseConfig
from .wireless import WirelessChannel


class WarehouseAGVEnv:
    def __init__(self, cfg: WarehouseConfig):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.n_links = cfg.n_links
        self.n_actions = cfg.n_actions
        self.rng = np.random.RandomState(cfg.seed)

        self.W, self.H = cfg.map_size
        self.edge_pos = np.array([self.W / 2.0, self.H / 2.0], dtype=np.float64)

        # AGV static properties
        self.agv_speed = cfg.speed
        self.agv_cpu = np.clip(
            self.rng.normal(cfg.f_local_mean, cfg.f_local_std, self.n_agents),
            cfg.f_local_mean * 0.3, cfg.f_local_mean * 2.0,
        )
        self.link_bw = np.array(cfg.link_bw, dtype=np.float64)
        self.wireless = WirelessChannel(cfg) if cfg.use_wireless else None

        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        cfg = self.cfg
        self.t = 0.0
        self.step_idx = 0
        self.done = False

        # AGV dynamic state
        self.agv_pos = self.rng.rand(self.n_agents, 2) * np.array([self.W, self.H])
        self.agv_free = np.zeros(self.n_agents)          # time when AGV becomes idle
        self.agv_task = np.full(self.n_agents, -1, dtype=int)  # task currently served

        # Link / edge state
        self.link_free = np.zeros(self.n_links)          # time when link free (serial mode only)
        self.link_active = [[] for _ in range(self.n_links)]  # (agv_idx, end_time) per link (wireless mode)
        self.link_load = np.zeros(self.n_links)          # cumulative bits uploaded
        self.edge_free = 0.0                             # time when edge CPU becomes free
        self.rate_samples = []                           # offload rates [bps] (wireless mode)
        self.sinr_samples = []                           # offload SINR [dB] (wireless mode)

        # Tasks
        self.tasks = []                                  # list of dicts
        self.task_counter = 0
        self.next_task_id = 0

        # Metrics
        self.total_arrived = 0
        self.num_completed = 0
        self.num_met_deadline = 0
        self.num_missed = 0
        self.completion_times = []

        self._arrivals()
        return self.get_graph()

    # ------------------------------------------------------------------ #
    def _arrivals(self):
        """Generate Poisson task arrivals at the current time."""
        n = self.rng.poisson(self.cfg.arrival_rate)
        for _ in range(n):
            self._spawn_task()

    def _spawn_task(self):
        cfg = self.cfg
        shelf = self.rng.rand(2) * np.array([self.W, self.H])
        D = self.rng.uniform(cfg.data_min, cfg.data_max)
        C = self.rng.uniform(cfg.comp_min, cfg.comp_max)
        ddl = self.t + self.rng.uniform(cfg.deadline_min, cfg.deadline_max)
        self.tasks.append({
            'id': self.next_task_id,
            'arrival': self.t,
            'pos': shelf,
            'D': D,
            'C': C,
            'deadline': ddl,
            'status': 'pending',        # pending / assigned / done / expired
            'done_time': np.inf,
            'agv': -1,
            'link': -1,
        })
        self.next_task_id += 1
        self.total_arrived += 1

    # ------------------------------------------------------------------ #
    def _idle_agents(self):
        return np.where(self.agv_free <= self.t + 1e-9)[0]

    def _pending_tasks(self):
        return [i for i, tk in enumerate(self.tasks) if tk['status'] == 'pending']

    def _nearest_task(self, agv_idx):
        """Return the pending task closest to AGV agv_idx, or None."""
        pending = self._pending_tasks()
        if not pending:
            return None
        pos = self.agv_pos[agv_idx]
        best = None
        best_d = np.inf
        for k in pending:
            d = np.linalg.norm(self.tasks[k]['pos'] - pos)
            if d < best_d:
                best_d = d
                best = k
        return best

    # ------------------------------------------------------------------ #
    def _prune_links(self):
        """Drop uploads that have already finished (wireless shared-medium mode)."""
        for m in range(self.n_links):
            self.link_active[m] = [(a, e) for (a, e) in self.link_active[m] if e > self.t]

    def link_congestion(self):
        """Per-link busy indicator: active upload count (wireless) or busy flag (serial)."""
        if self.wireless is not None:
            return np.array([len(a) for a in self.link_active], dtype=np.float64)
        return (self.link_free > self.t).astype(np.float64)

    def rate_upper(self, i, m):
        """No-interference offload rate for AGV i on link m [bps]."""
        if self.wireless is None:
            return float(self.link_bw[m])
        dist = np.linalg.norm(self.agv_pos[i] - self.edge_pos)
        rate, _ = self.wireless.compute(dist, self.link_bw[m], [])
        return rate

    # ------------------------------------------------------------------ #
    def step(self, actions):
        """Apply one step. `actions` is an int array of shape (n_agents,)."""
        cfg = self.cfg
        t0 = self.t
        reward = np.zeros(self.n_agents, dtype=np.float64)

        self._prune_links()

        # 1) idle AGVs pick tasks according to their actions
        idle = self._idle_agents()
        has_pending = len(self._pending_tasks()) > 0
        picking = {}  # task id -> list of (agv, dist)
        for i in idle:
            a = int(actions[i])
            if a == 0:  # idle
                if has_pending:
                    reward[i] -= cfg.w_idle  # shaping: don't idle while work is available
                continue
            tgt = self._nearest_task(i)
            if tgt is None:
                continue
            d = np.linalg.norm(self.tasks[tgt]['pos'] - self.agv_pos[i])
            picking.setdefault(tgt, []).append((i, d))

        # 2) conflict resolution: nearest AGV wins each task
        offloads = []  # (agent_idx, link_idx) for this step's offload decisions
        for tgt, cands in picking.items():
            cands.sort(key=lambda x: x[1])
            winner = cands[0][0]
            link = self._assign(winner, tgt, actions[winner])
            reward[winner] += cfg.w_assign  # shaping: accepting a task is good
            if link >= 0:
                offloads.append((winner, link))

        # 3) advance time and process events
        self.t += cfg.dt
        self.step_idx += 1

        # 4) completion + expiry, with per-agent credit
        completed_now = 0
        met_now = 0
        missed_now = 0
        for k in self.tasks:
            if k['status'] == 'assigned' and k['done_time'] <= self.t:
                k['status'] = 'done'
                self.num_completed += 1
                self.completion_times.append(k['done_time'] - k['arrival'])
                completed_now += 1
                i = k['agv']
                reward[i] += cfg.w_complete
                if k['done_time'] <= k['deadline']:
                    self.num_met_deadline += 1
                    met_now += 1
                    reward[i] += cfg.w_deadline
                else:
                    self.num_missed += 1
                    missed_now += 1
                    reward[i] -= cfg.w_miss
            elif k['status'] == 'pending' and self.t > k['deadline']:
                # task was never picked before its deadline passed (system miss)
                k['status'] = 'expired'
                self.num_missed += 1
                missed_now += 1

        # 5) new arrivals for the next interval
        self._arrivals()

        # 6) reward: link-load balancing
        load_imbalance = self._link_imbalance()
        reward -= cfg.w_load * load_imbalance  # global shared penalty (weak)

        # targeted penalty: offloading to a link that is already over-loaded
        total_load = self.link_load.sum()
        if total_load > 1e-9:
            for i, m in offloads:
                share = self.link_load[m] / total_load
                excess = max(0.0, share - 1.0 / self.n_links)
                reward[i] -= cfg.w_link_choice * excess

        # 7) terminal flag
        self.done = self.step_idx >= cfg.horizon

        info = {
            'completed': completed_now,
            'met_deadline': met_now,
            'missed': missed_now,
            'load_imbalance': load_imbalance,
        }
        return self.get_graph(), reward, self.done, info

    # ------------------------------------------------------------------ #
    def _assign(self, i, tgt, action):
        """Assign task `tgt` to AGV `i` with the given action (1=local, >=2=edge link)."""
        cfg = self.cfg
        task = self.tasks[tgt]
        task['status'] = 'assigned'
        task['agv'] = i

        travel = np.linalg.norm(task['pos'] - self.agv_pos[i]) / self.agv_speed
        self.agv_pos[i] = task['pos']  # AGV moves to the shelf
        self.agv_task[i] = tgt

        if action == 1:  # local
            comp = task['C'] / self.agv_cpu[i]
            done_time = self.t + travel + comp
            self.agv_free[i] = done_time
            task['done_time'] = done_time
            return -1

        # offload via link (action - 2)
        m = int(action) - 2
        m = min(max(m, 0), self.n_links - 1)
        arrival = self.t + travel
        if self.wireless is not None:
            # shared medium: co-channel AGVs still uploading when this AGV arrives interfere
            dist_i = np.linalg.norm(self.agv_pos[i] - self.edge_pos)
            interferer_dists = [np.linalg.norm(self.agv_pos[j] - self.edge_pos)
                                for (j, e) in self.link_active[m] if e > arrival]
            rate, sinr_db = self.wireless.compute(dist_i, self.link_bw[m], interferer_dists)
            upload_end = arrival + task['D'] / rate
            self.link_active[m].append((i, upload_end))
            self.rate_samples.append(rate)
            self.sinr_samples.append(sinr_db)
        else:
            up = task['D'] / self.link_bw[m]
            upload_start = max(arrival, self.link_free[m])
            upload_end = upload_start + up
            self.link_free[m] = upload_end
        self.link_load[m] += task['D']
        comp_start = max(upload_end, self.edge_free)
        done_time = comp_start + task['C'] / self.cfg.f_edge
        self.edge_free = done_time
        self.agv_free[i] = upload_end
        task['link'] = m
        task['done_time'] = done_time
        return m

    # ------------------------------------------------------------------ #
    def _link_imbalance(self):
        """Coefficient of variation of cumulative link loads."""
        total = self.link_load.sum()
        if total < 1e-9:
            return 0.0
        mean = total / self.n_links
        std = np.sqrt(np.mean((self.link_load - mean) ** 2))
        return float(std / mean)

    # ------------------------------------------------------------------ #
    def get_graph(self):
        """Build the heterogeneous graph as numpy arrays for the GNN."""
        cfg = self.cfg

        # AGV nodes
        agv_feat = np.zeros((self.n_agents, 5), dtype=np.float32)
        agv_feat[:, 0] = self.agv_pos[:, 0] / self.W
        agv_feat[:, 1] = self.agv_pos[:, 1] / self.H
        agv_feat[:, 2] = (self.agv_free > self.t).astype(np.float32)
        agv_feat[:, 3] = np.clip((self.agv_free - self.t) / (cfg.horizon * cfg.dt), 0, 1)
        agv_feat[:, 4] = self.agv_cpu / cfg.f_edge

        # Task nodes (highest-priority pending tasks first)
        pending = sorted(self._pending_tasks(),
                         key=lambda k: self.tasks[k]['deadline'])
        pending = pending[:cfg.max_tasks]
        n_task = len(pending)
        task_feat = np.zeros((max(n_task, 1), 6), dtype=np.float32)
        task_pos = np.zeros((max(n_task, 1), 2), dtype=np.float32)
        for j, k in enumerate(pending):
            tk = self.tasks[k]
            task_feat[j, 0] = tk['pos'][0] / self.W
            task_feat[j, 1] = tk['pos'][1] / self.H
            task_feat[j, 2] = tk['D'] / cfg.data_max
            task_feat[j, 3] = tk['C'] / cfg.comp_max
            task_feat[j, 4] = np.clip((tk['deadline'] - self.t) / (cfg.deadline_max * 2), -1, 1)
            task_feat[j, 5] = np.clip((self.t - tk['arrival']) / (cfg.horizon * cfg.dt), 0, 1)
            task_pos[j] = tk['pos']

        # Edge server node
        edge_feat = np.zeros((1, 4), dtype=np.float32)
        edge_feat[0, 0] = self.edge_pos[0] / self.W
        edge_feat[0, 1] = self.edge_pos[1] / self.H
        edge_feat[0, 2] = 1.0
        edge_feat[0, 3] = np.clip(max(0.0, self.edge_free - self.t) / (cfg.horizon * cfg.dt), 0, 1)

        # Link nodes
        link_feat = np.zeros((self.n_links, 3), dtype=np.float32)
        link_feat[:, 0] = self.link_bw / max(self.link_bw)
        link_feat[:, 1] = self.link_load / (self.link_load.max() + 1e-6)
        if self.wireless is not None:
            link_feat[:, 2] = self.link_congestion() / max(self.n_agents, 1)
        else:
            link_feat[:, 2] = np.clip((self.link_free - self.t) / (cfg.horizon * cfg.dt), 0, 1)

        # Edges (heterogeneous, each as (2, E) index arrays into a per-type node space)
        edge_index = {}

        # AGV <-> task: bipartite, connect to nearest K tasks
        if n_task > 0:
            K = 4
            src_agv, dst_task, src_task, dst_agv = [], [], [], []
            for i in range(self.n_agents):
                d = np.linalg.norm(task_pos[:n_task] - self.agv_pos[i], axis=1)
                top = np.argsort(d)[:K]
                for j in top:
                    src_agv.append(i); dst_task.append(j)
                    src_task.append(j); dst_agv.append(i)
            edge_index['agv_task'] = np.array([src_agv, dst_task], dtype=np.int64)
            edge_index['task_agv'] = np.array([src_task, dst_agv], dtype=np.int64)

        # task -> edge and edge -> task
        if n_task > 0:
            edge_index['task_edge'] = np.array([np.arange(n_task), np.zeros(n_task, dtype=np.int64)], dtype=np.int64)
            edge_index['edge_task'] = np.array([np.zeros(n_task, dtype=np.int64), np.arange(n_task)], dtype=np.int64)

        # edge -> link and link -> edge
        links = np.arange(self.n_links, dtype=np.int64)
        edge_index['edge_link'] = np.array([np.zeros(self.n_links, dtype=np.int64), links], dtype=np.int64)
        edge_index['link_edge'] = np.array([links, np.zeros(self.n_links, dtype=np.int64)], dtype=np.int64)

        return {
            'agv': agv_feat,
            'task': task_feat,
            'edge': edge_feat,
            'link': link_feat,
            'n_task': n_task,
            'edge_index': edge_index,
        }

    # ------------------------------------------------------------------ #
    def metrics(self):
        """Episode-level metrics used for evaluation."""
        total = max(self.total_arrived, 1)
        avg_comp_time = float(np.mean(self.completion_times)) if self.completion_times else float('nan')
        avg_rate = float(np.mean(self.rate_samples)) if self.rate_samples else float('nan')
        avg_sinr_db = float(np.mean(self.sinr_samples)) if self.sinr_samples else float('nan')
        return {
            'completion_rate': self.num_completed / total,
            'deadline_rate': self.num_met_deadline / total,
            'missed_rate': self.num_missed / total,
            'avg_completion_time': avg_comp_time,
            'link_imbalance': self._link_imbalance(),
            'avg_rate': avg_rate,
            'avg_sinr_db': avg_sinr_db,
        }
