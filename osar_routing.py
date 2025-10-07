import simpy
import numpy as np
from typing import List, Optional, Tuple
from utils import UnderwaterNode,G,setup_small_net
from channel_model import snr, propagation_delay  

class OSARRouter:
    def __init__(self, num_channels: int = 5, packet_size_bytes: int = 64):
        self.M = num_channels
        self.LP = packet_size_bytes * 8  # Bits
        self.dest_depth = 0  # Surface
        self.B_ch = 6000.0  # Hz per channel
        self.center_freqs = np.array([13, 19, 25, 31, 37])  # kHz (for SNR)

    def sense_and_beacon(self, node: UnderwaterNode, pu_active: bool, env: simpy.Environment):
        """Simplified sensing: Random idle channels (no OFDM)."""
        node.idle_channels.clear()  # Reset
        for ch in range(self.M):
            # PU busy prob ~50% if active (exponential on/off approx)
            if pu_active and np.random.rand() < 0.5:
                continue  # Busy: Skip
            node.idle_channels.append(ch)  # Idle
        yield env.timeout(0.01)  # Sensing time

    def data_rate(self, dist_m: float, ch: int) -> float:
        """Shannon approx: B_ch * log2(1 + SNR) in kbps."""
        f_mid = self.center_freqs[ch]
        snr_lin = snr(dist_m, f_mid)
        return self.B_ch * np.log2(1 + snr_lin) / 1000.0  # kbps

    def transmission_delay(self, node_i: UnderwaterNode, node_j: UnderwaterNode, ch: int, dest: UnderwaterNode) -> float:
        """Eq. 9: TD = LP / rate + prop_delay * est_hops."""
        dist_ij = np.linalg.norm(node_i.pos - node_j.pos)
        rate = self.data_rate(dist_ij, ch)
        prop_delay = propagation_delay(dist_ij)
        dij = node_i.depth - node_j.depth  # Advance (positive upward)
        if dij <= 0:
            return float('inf')  # Downward: Invalid
        est_hops = max(node_i.depth / dij, 1)
        if rate != 0:
            delay = (self.LP / rate) + (prop_delay * est_hops)
        else :
            delay = "inf" 
        return delay

    def select_relay(self, node: UnderwaterNode, candidates: List[UnderwaterNode], dest: UnderwaterNode) -> Optional[UnderwaterNode]:
        """Eq. 10: Min TD over common idle ch neighbors."""
        common_ch = set(node.idle_channels)
        min_td = float('inf')
        best_relay = None
        for cand in candidates:
            if cand.depth >= node.depth:  # Only upward
                continue
            shared_ch = set(cand.idle_channels) & common_ch
            if not shared_ch:
                continue
            ch = np.random.choice(list(shared_ch))  # Random common ch
            td = self.transmission_delay(node, cand, ch, dest)
            if td < min_td:
                min_td = td
                best_relay = cand
        return best_relay

def run_packet_forwarding(env: simpy.Environment, source: UnderwaterNode, dest: UnderwaterNode, 
                          router: OSARRouter, path: List[int] = None):
    """Hop-by-hop forwarding; returns (total_delay, success, path)."""
    node_dict = {node.id: node for node in all_nodes}  # node_dict = {"node_id":UnderwaterNode}
    if path is None:
        path = [source.id]
    current = source
    total_delay = 0.0
    while current != dest:
        # candidates should be list of Underwater Nodes
        # create dictionary of neighbors/ candidates at shallower depth 
        # upward_ids = [neighbors_ids]
        upward_ids = [n_id for n_id, _ in current.neighbor_nodes.items() if G.nodes[n_id]['depth'] < current.depth]
        candidates = [node_dict[n_id] for n_id in upward_ids if n_id in node_dict]
        next_hop = router.select_relay(current, candidates, dest)
        if not next_hop:
            print(f"Drop at {current.id}: No valid relay")
            return total_delay, False, path
        # Compute hop delay (simplified: prop only; rate in selection)
        dist = np.linalg.norm(current.pos - next_hop.pos)
        hop_delay = propagation_delay(dist)
        total_delay += hop_delay
        yield env.timeout(hop_delay)  # Simulate acoustic delay
        path.append(next_hop.id)
        current = next_hop
    print(f"Delivered via path: {' -> '.join(map(str, path))}")
    return total_delay, True, path



# Test 2: Full packet forwarding (one packet)
print("\n--- Test 2: Full Forwarding with SimPy ---")

all_nodes, G, node_dict, source, dest = setup_small_net()
# Setup router and env
router = OSARRouter(num_channels=3)  # Small M=3
env = simpy.Environment()

# Sense for source and neighbors (simplified)
pu_active = np.random.rand() > 0.5  # Random PU
env.process(router.sense_and_beacon(source, pu_active, env))
for node in all_nodes[1:]:  # Sense others
    env.process(router.sense_and_beacon(node, pu_active, env))
env.run(until=0.05)  # Run sensing

# Run forwarding for one packet
def test_forwarding():
    path = []
    delay, success, path = yield env.process(
        run_packet_forwarding(env, source, dest, router, path=path)
    )
    print(f"Forwarding result: Success={success}, Delay={delay:.3f}s, Path={path}")

env.process(test_forwarding())
env.run(until=10.0)  # Short sim time

# Expected: 1-3 hops, e.g., Path=[0, 2, 5], Delay=0.250s