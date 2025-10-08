"""
algorithms/osar_routing.py

OSARRouter class implementing the relay-selection logic and a SimPy-style
hop-by-hop forwarding process.

Key methods:
  - sense_and_beacon(node, pu_active, env): simulate sensing and fill idle channels
  - data_rate(dist_m, ch): wrapper to compute data rate
  - transmission_delay(node_i, node_j, ch): compute TD for link i->j
  - select_relay(node, candidates, dest): choose best next hop (min TD)
  - run_packet_forwarding(env, source, dest): generator-style forwarding process
"""
import numpy as np
from typing import List, Optional, Tuple
import simpy

from ..models.transmission_delay import compute_transmission_delay, compute_data_rate
from ..models.channel_model import propagation_delay, snr
from ..utils.utils import UnderwaterNode, get_distance
from .EB import random_sensing


class OSARRouter:
    def __init__(self, num_channels: int = 5, packet_size_bytes: int = 64, bandwidth_hz: float = 6000.0):
        self.M = num_channels
        self.LP = packet_size_bytes * 8  # bits
        self.dest_depth = 0.0  # surface
        self.B_ch = bandwidth_hz
        # center frequencies (kHz) for channels — can be parameterized
        self.center_freqs = np.array([13, 19, 25, 31, 37])

    def sense_and_beacon(self, node: UnderwaterNode, pu_active: bool, env: simpy.Environment):
        """Simplified sensing: Random idle channels (no OFDM)."""
        node.idle_channels.clear()
        for ch in range(self.M):
            if pu_active and np.random.rand() < 0.5:
                continue
            node.idle_channels.append(ch)
        yield env.timeout(0.001)  # ✅ This makes it a generator

    def data_rate(self, dist_m: float, ch: int, Pt: float = 1.0) -> float:
        """
        Return data rate in bits/sec for a given distance and channel index.
        """
        f_khz = float(self.center_freqs[ch])
        return compute_data_rate(dist_m, f_khz, self.B_ch, Pt=Pt)

    def transmission_delay(self, node_i: UnderwaterNode, node_j: UnderwaterNode, ch: int) -> float:
        """
        Compute TD between node_i and node_j when transmitting on channel ch.
        Returns TD in seconds (tx_time + pd * est_hops). If invalid link, returns inf.
        """
        # ensure upward progress
        if node_j.depth >= node_i.depth:
            return float('inf')
        f_khz = float(self.center_freqs[ch])
        TD, pd = compute_transmission_delay(node_i.pos, node_i.depth, node_j.pos, node_j.depth,
                                            f_khz, LP_bits=self.LP, bandwidth_hz=self.B_ch)
        return TD

    def select_relay(self, node: UnderwaterNode, candidates: List[UnderwaterNode],
                     dest_pos: np.ndarray) -> Optional[Tuple[UnderwaterNode, int, float]]:
        """
        Select best relay among candidates that share at least one idle channel.
        Returns tuple (best_node, chosen_channel, td) or None if no candidate.
        """
        common_ch = set(node.idle_channels)
        best = None
        best_td = float('inf')
        best_ch = None
        for cand in candidates:
            if cand.depth >= node.depth:
                continue
            shared = common_ch & set(cand.idle_channels)
            if not shared:
                continue
            # estimate best channel for that candidate (minimum TD)
            for ch in shared:
                td = self.transmission_delay(node, cand, ch)
                if td < best_td:
                    best_td = td
                    best = cand
                    best_ch = ch
        if best is None:
            return None
        return best, best_ch, best_td


def run_packet_forwarding(env: simpy.Environment, source: UnderwaterNode, dest: UnderwaterNode,
                          router: OSARRouter, G, nodes_by_id: dict, tx_range: float = 250.0):
    """
    SimPy process that forwards a single packet hop-by-hop from source to dest.
    Yields timeouts equal to propagation delays for each hop and returns (total_delay, success, path).
    """
    path = [source.id]
    current = source
    total_delay = 0.0

    while current.id != dest.id:
        # build candidate list from neighbor IDs in current.neighbor_nodes
        candidates = []
        for nid in current.neighbor_nodes.keys():
            if nid in nodes_by_id:
                candidate = nodes_by_id[nid]
                # within tx_range
                if np.linalg.norm(current.pos - candidate.pos) > tx_range:
                    continue
                # only upward
                if candidate.depth >= current.depth:
                    continue
                candidates.append(candidate)

        # select best relay
        sel = router.select_relay(current, candidates, dest.pos)
        if sel is None:
            # drop
            return total_delay, False, path
        next_hop, ch, td = sel
        # simulate propagation delay only (transmission/tx_time already accounted in TD)
        dist = np.linalg.norm(current.pos - next_hop.pos)
        pd = propagation_delay(dist, z_km=(
            max(current.depth, next_hop.depth) / 1000.0))
        total_delay += td
        # yield the propagation portion so SimPy advances
        yield env.timeout(pd)
        path.append(next_hop.id)
        current = next_hop

        # safety: avoid infinite loops
        if len(path) > 100:
            return total_delay, False, path

    return total_delay, True, path
