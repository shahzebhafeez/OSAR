import numpy as np

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
        self.neighbors = {}  # neighbor_id : info dict

# ---------- Utility Functions ----------
def place_nodes_uniform(N, area=500.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(0, area, N)
    ys = rng.uniform(0, area, N)
    zs = rng.uniform(0, area, N)
    return np.column_stack((xs, ys, zs))

def pairwise_distances(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

# ---------- EB Broadcast Simulation ----------
def extended_beacon_broadcast(nodes, tx_range, dists):
    """Each node broadcasts EB; neighbors within range update their tables"""
    for i, src in enumerate(nodes):
        for j, nbr in enumerate(nodes):
            if i == j:
                continue
            if dists[i, j] <= tx_range:  # within range
                nbr.neighbors[src.node_id] = {
                    "depth": src.depth,
                    "pos": src.pos,
                    "sound_speed": src.sound_speed,
                    "distance": dists[i, j],
                    "busy_channels": np.where(src.channel_state == 1)[0]
                }

# ---------- Main Simulation ----------
def main():
    rng = np.random.default_rng(42)
    N = 30
    M = 128
    AREA = 500.0
    tx_range = 150.0
    p_busy = 0.3

    # Create nodes
    positions = place_nodes_uniform(N, AREA, rng)
    nodes = []
    for i in range(N):
        channel_state = rng.random(M) < p_busy
        nodes.append(Node(i, positions[i], channel_state))

    # Distances
    dists = pairwise_distances(positions)

    # EB Broadcast
    extended_beacon_broadcast(nodes, tx_range, dists)

    # Display neighbor info
    for n in nodes[:5]:  # print first 5 nodes for brevity
        print(f"\nNode {n.node_id} | Depth={n.depth:.1f}m | Sound Speed={n.sound_speed:.1f} m/s")
        print(f"  Neighbors ({len(n.neighbors)}):")
        for nb_id, info in n.neighbors.items():
            print(f"    → Node {nb_id} | Dist={info['distance']:.1f}m | Depth={info['depth']:.1f}m | Busy={len(info['busy_channels'])}")

if __name__ == "__main__":
    main()
