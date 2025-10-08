"""
main.py

Top-level simulation driver that wires together utilities, channel model,
OSAR router and runs a small experiment printing Avg Delay and PDR.

This file expects the package layout:
- models.channel_model
- models.transmission_delay
- algorithms.osar_routing
- algorithms.eb (or eb_without_time)
- algorithms.candidate_selection
- utils.utils
"""
import simpy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .utils.utils import generate_environment, compute_neighbors, visualize_path
from .algorithms.osar_routing import OSARRouter, run_packet_forwarding
from .utils.utils import UnderwaterNode

def log(msg: str, file):
    print(msg)
    if file:
        file.write(msg + "\n")
        file.flush()


def simulate(N: int = 30, num_packets: int = 5, M: int = 5, visualize: bool = True, seed: int = 42):
    sensors, buoys = generate_environment(N, seed=seed)
    nodes_all = sensors + buoys
    G = compute_neighbors(nodes_all, tx_range=250.0)

    # create id -> node map
    nodes_by_id = {n.id: n for n in nodes_all}

    router = OSARRouter(num_channels=M)
    env = simpy.Environment()

    # pick the deepest sensor as source
    source = max(sensors, key=lambda n: n.depth)
    dest = buoys[0]

    # sensing phase (simplified)
    pu_active = np.random.exponential(1.0) > 1.0

    log_file = Path(__file__).parent / "simulation_log.txt"
    f = open(log_file, "w")

    log(f"=== OSAR Simulation Log ===", f)
    log(f"Total Nodes: {N}, Channels: {M}\n", f)

    for node in nodes_all:
        env.process(router.sense_and_beacon(node, pu_active, env))
    env.run(until=0.01)

    # --- Display first 5 nodes details ---
    log(f"--- Node Information (First 5) ---", f)
    for node in nodes_all[:5]:
        neighbor_ids = list(node.neighbor_nodes.keys())
        log(f"Node {node.id}: Depth={node.depth:.1f}m | Neighbors={neighbor_ids} | Idle Channels={node.idle_channels}", f)

    results = []
    delivered = 0
    total_delays = []

    def pkt_proc():
        nonlocal delivered
        for i in range(num_packets):
            # each packet is forwarded via a SimPy process
            log(f"\n=== Packet {i + 1} ===", f)
            res = yield env.process(run_packet_forwarding(env, source, dest, router, G, nodes_by_id))
            # res is (total_delay, success, path)
            total_delay, success, path  = res
            total_delays.append(total_delay if success else float('inf'))
            if success:
                delivered += 1
                log(f"Packet Delivered. Path: {path}", f)
                log(f"Total Delay: {total_delay:.4f} s", f)
            
            else:
                log(f"Packet Dropped. Partial Path: {path}", f)
            results.append(path)
            # wait a bit before sending next packet
            yield env.timeout(0.5)

    env.process(pkt_proc())
    env.run(until=100.0)

    pdr = delivered / num_packets
    valid_delays = [d for d in total_delays if d < float('inf')]
    avg_delay = np.mean(valid_delays) if valid_delays else float('inf')

    log(f"\n--- Simulation Summary ---", f)
    log(f"N={N}, M={M}: Avg Delay={avg_delay:.3f}s, PDR={pdr:.3f}", f)

    f.close()

    if visualize and results:
        visualize_path(G, results[-1], results, title=f"OSAR: N={N}, M={M}")

    print(f"\nDetailed log written to: {log_file}")
    return avg_delay, pdr, results


if __name__ == "__main__":
    Ns = [10, 20, 30]
    summary = []
    for N in Ns:
        avg, pdr, paths = simulate(N=N, num_packets=3, M=5, visualize=True, seed=123)
        summary.append((N, avg, pdr))

    Ns, delays, pdrs = zip(*summary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(Ns, delays, 'bo-', linewidth=2); ax1.set_xlabel('N'); ax1.set_ylabel('Avg Delay (s)'); ax1.grid(True)
    ax2.plot(Ns, pdrs, 'go-', linewidth=2); ax2.set_xlabel('N'); ax2.set_ylabel('PDR'); ax2.grid(True)
    plt.tight_layout(); plt.show()
