"""
Wireless layer for the warehouse offloading scenario.

Computes each AGV's offload rate from the 3GPP factory path-loss model (reused from the
original 6G in-factory subnetwork environment `infactory_env.channel_pathLoss`) plus
co-channel interference. Each Wi-Fi 7 link is a shared channel: AGVs offloading on the same
link interfere, so an AGV's rate degrades with distance to the edge server and with the
number of co-channel AGVs currently uploading.

Pure numpy; independent of torch so it can be unit-tested standalone.
"""
import numpy as np


class WirelessChannel:
    def __init__(self, cfg):
        self.fc = float(cfg.fc)
        self.tx_pow_dbm = float(cfg.tx_pow_dbm)
        self.noise_floor_dbm = float(cfg.noise_floor_dbm)
        self.noise_fig_db = float(cfg.noise_fig_db)
        self.min_dist = float(cfg.min_dist)
        self.max_spec_eff = float(cfg.max_spec_eff)

    def path_loss_db(self, dist):
        """3GPP factory LoS path loss in dB (from infactory_env.channel_pathLoss)."""
        d = np.maximum(np.asarray(dist, dtype=np.float64), self.min_dist)
        return 31.84 + 21.5 * np.log10(d) + 19.0 * np.log10(self.fc / 1e9)

    def noise_lin(self, bw):
        """Thermal noise power (linear) for a channel of bandwidth `bw` [Hz]."""
        return 10.0 ** ((self.noise_floor_dbm + self.noise_fig_db + 10.0 * np.log10(bw)) / 10.0)

    def rx_power_lin(self, dist):
        """Received power (linear) of a transmitter at distance `dist` [m]."""
        return 10.0 ** ((self.tx_pow_dbm - self.path_loss_db(dist)) / 10.0)

    def compute(self, dist_to_edge, bw, interferer_dists):
        """Return (rate [bps], sinr [dB]) for one AGV offloading on a link.

        dist_to_edge: float, distance AGV -> edge server [m].
        bw: float, channel bandwidth of the link [Hz].
        interferer_dists: array-like of distances (m) of the co-channel AGVs that are
            currently uploading on the same link.
        """
        rx = float(self.rx_power_lin(dist_to_edge))
        ids = np.asarray(interferer_dists, dtype=np.float64)
        interferers = float(np.sum(self.rx_power_lin(ids))) if ids.size > 0 else 0.0
        sinr = rx / (interferers + self.noise_lin(bw))
        sinr_db = 10.0 * np.log10(sinr)
        spec_eff = min(np.log2(1.0 + sinr), self.max_spec_eff)
        return float(bw * spec_eff), float(sinr_db)
