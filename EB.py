import numpy as np
import time

# ---------- Sound Speed ----------
def calc_sound_speed(temp=10, salinity=35, depth=100):
    # Simplified empirical equation (m/s)
    return 1448.96 + 4.591*temp - 0.05304*(temp**2) + 0.0163*depth + 1.34*(salinity - 35)

# ---------- Node Class ----------
class Node:
    def __init__(self, node_id, pos, channel_state, temp=10, salinity=35):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state  # 1 = busy, 0 = idle
        self.sound_speed = calc_sound_speed(temp, salinity, self.depth)
        self.neighbors = {}  # {neighbor_id: info}
        self.last_broadcast = 0  # last EB sent time (s)

# ---------- Utility Functions ----------
def place_nodes_uniform(N, area=500.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(0, area, N)
    ys = rng.uniform(0, area, N)
    zs = rng.uniform(0, area, N)
    return np.column_stack((xs, ys, zs))

def distance(a, b):
    return np.linalg.norm(a - b)

# ---------- EB Broadcast Simulation ----------
def broadcast_eb(src, nodes, tx_range, current_time):
    """Simulate EB broadcast from one node; heard by neighbors within range."""
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue

        # Simulate "EB heard" if within tx_range
        dist = distance(src.pos, nbr.pos)
        if dist <= tx_range:
            nbr.neighbors[src.node_id] = {
                "depth": src.depth,
                "pos": src.pos,
                "sound_speed": src.sound_speed,
                "distance": dist,
                "busy_channels": np.where(src.channel_state == 1)[0],
                "last_heard": current_time
            }

def remove_stale_neighbors(nodes, current_time, timeout):
    """Remove neighbors not heard from within timeout period."""
    for node in nodes:
        expired = [
            nid for nid, info in node.neighbors.items()
            if current_time - info["last_heard"] > timeout
        ]
        for nid in expired:
            del node.neighbors[nid]

# ---------- Main Simulation ----------
def main():
    rng = np.random.default_rng(42)
    N = 30           # number of nodes
    M = 128          # number of subcarriers
    AREA = 500.0     # 3D cube size (m)
    tx_range = 150.0 # EB transmission range (m)
    p_busy = 0.3     # probability that a subcarrier is busy
    EB_INTERVAL = 5  # seconds between EBs
    TIMEOUT = 15     # seconds before a neighbor expires
    SIM_TIME = 30    # total simulation duration (seconds)
    TIME_STEP = 1    # simulation step (seconds)

    # Create nodes
    positions = place_nodes_uniform(N, AREA, rng)
    nodes = []
    for i in range(N):
        channel_state = rng.random(M) < p_busy
        nodes.append(Node(i, positions[i], channel_state))

    # Simulation loop
    for t in range(0, SIM_TIME, TIME_STEP):
        for node in nodes:
            if t - node.last_broadcast >= EB_INTERVAL:
                node.last_broadcast = t
                broadcast_eb(node, nodes, tx_range, t)

        remove_stale_neighbors(nodes, t, TIMEOUT)

        if t % 10 == 0:
            print(f"\n Time {t}s - Active neighbor tables:")
            for n in nodes[:5]:  # print first 5 for brevity
                print(f"Node {n.node_id}: {len(n.neighbors)} neighbors")

        # (Optional) time.sleep(0.1)  # Slow down if running interactively

    # Final summary
    print("\nFinal neighbor info (first 5 nodes):")
    for n in nodes[:5]:
        print(f"\nNode {n.node_id} | Depth={n.depth:.1f}m | Sound Speed={n.sound_speed:.1f} m/s")
        print(f"  Neighbors ({len(n.neighbors)}):")
        for nb_id, info in n.neighbors.items():
            print(f"    → Node {nb_id} | Dist={info['distance']:.1f}m | Depth={info['depth']:.1f}m | Busy={len(info['busy_channels'])}")

if __name__ == "__main__":
    main()
