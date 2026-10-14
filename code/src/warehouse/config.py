"""Configuration for the warehouse AGV computing-offloading scenario."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class WarehouseConfig:
    # --- Environment ---
    n_agents: int = 10              # number of AGVs
    n_links: int = 3                # number of Wi-Fi 7 links (multi-link operation)
    map_size: tuple = (60.0, 40.0)  # warehouse area [m, m]
    seed: int = 123
    horizon: int = 120              # decision steps per episode
    dt: float = 0.5                 # seconds per step

    # --- AGV / compute ---
    speed: float = 3.0              # AGV travel speed [m/s]
    f_local_mean: float = 1.0e9     # mean local CPU [cycles/s]
    f_local_std: float = 0.3e9      # spread of local CPU (heterogeneous AGVs)
    f_edge: float = 15.0e9          # edge server CPU [cycles/s]

    # --- Wi-Fi 7 multi-link ---
    link_bw: List[float] = field(default_factory=lambda: [20.0e6, 40.0e6, 80.0e6])  # base bandwidth per link [Hz]

    # --- Wireless coupling (reuses the 3GPP path-loss model from the original project) ---
    use_wireless: bool = True         # False -> fall back to fixed link_bw (ablation)
    fc: float = 6e9                   # carrier frequency [Hz]
    tx_pow_dbm: float = -10.0         # AGV transmit power [dBm]
    noise_floor_dbm: float = -174.0   # thermal noise [dBm/Hz]
    noise_fig_db: float = 10.0        # receiver noise figure [dB]
    min_dist: float = 1.0             # clamp path-loss distance to avoid 0-distance singularity
    max_spec_eff: float = 12.0        # cap on spectral efficiency [bps/Hz]

    # --- Task arrival / workload ---
    arrival_rate: float = 0.5        # Poisson mean tasks per step
    data_min: float = 5.0e6         # bits
    data_max: float = 20.0e6        # bits
    comp_min: float = 1.0e9         # cycles
    comp_max: float = 5.0e9         # cycles
    deadline_min: float = 4.0       # relative deadline lower bound [s]
    deadline_max: float = 14.0      # relative deadline upper bound [s]
    max_tasks: int = 50             # cap on task nodes fed to the GNN

    # --- Reward weights ---
    w_complete: float = 1.0
    w_deadline: float = 1.0         # bonus for meeting deadline
    w_miss: float = 0.5             # penalty for missing deadline
    w_load: float = 0.5             # global link-imbalance penalty (absorbed by centralised critic)
    w_link_choice: float = 1.5      # per-agent penalty for offloading to an over-loaded link
    w_assign: float = 0.3           # immediate reward for accepting a task (shaping)
    w_idle: float = 0.1             # penalty for idling while tasks are pending (shaping)

    # --- Training ---
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95               # GAE lambda
    eps_clip: float = 0.2
    ppo_epochs: int = 4
    episodes: int = 400
    hidden: int = 64
    gnn_layers: int = 2
    entropy_coef: float = 0.01      # initial entropy bonus (exploration)
    entropy_coef_min: float = 0.001 # annealed lower bound (convergence)

    @property
    def n_actions(self) -> int:
        """0: idle, 1: local, 2..n_links+1: offload via link k-2."""
        return 1 + 1 + self.n_links
