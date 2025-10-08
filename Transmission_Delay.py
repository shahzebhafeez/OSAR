import numpy as np
import random

# ---------- Parameters ----------
LP = 512  # bits
BANDWIDTH = 6e3  # Hz (6 kHz per channel)
PT = 1.0  # W (transmit power)
NOISE_PSD = 1e-9  # W/Hz (noise spectral density)
AREA = 500.0  # cubic region
TX_RANGE = 250.0
P_BUSY = 0.099
M = 128  # subcarriers
TEMP = 10
SALINITY = 35

# ---------- Utility Functions ----------
def calc_sound_speed(temp=TEMP, salinity=SALINITY, depth=100):
    return 1448.96 + 4.591*temp - 0.05304*(temp**2) + 0.0163*depth + 1.34*(salinity - 35)

def distance(a, b):
    return np.linalg.norm(a - b)

def place_nodes_uniform(N, area=AREA):
    rng = np.random.default_rng()
    xs = rng.uniform(0, area, N)
    ys = rng.uniform(0, area, N)
    zs = rng.uniform(0, area, N)
    return np.column_stack((xs, ys, zs))

# ---------- Node Class ----------
class Node:
    def __init__(self, node_id, pos, channel_state):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = calc_sound_speed(depth=self.depth)

# ---------- Transmission Delay Function ----------
def compute_transmission_delay(src, dest, dest_pos, channel_idx):
    """Compute TD_ij for one channel based on Eq. (9)"""
    Dij = distance(src.pos, dest.pos)
    PDij = abs(src.depth - dest.depth) / src.sound_speed  # propagation delay
    # Approximate SNR inversely proportional to square of distance
    SNR = PT / (NOISE_PSD * BANDWIDTH * (Dij**2 + 1))
    r_ij_ch = BANDWIDTH * np.log2(1 + SNR)
    # Hop factor (projection toward destination)
    vec_sd = dest_pos - src.pos
    vec_sj = dest.pos - src.pos
    proj = np.dot(vec_sj, vec_sd) / (np.linalg.norm(vec_sd) + 1e-12)
    N_hop = max(proj / (Dij + 1e-12), 1)
    TD_ij = LP / r_ij_ch + PDij * N_hop
    return TD_ij

# ---------- Routing Function ----------
def select_best_next_hop(src, nodes, dest_pos, tx_range):
    """Select neighbor j ∈ N_i that minimizes TD_ij"""
    candidates = []
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue

        Dij = distance(src.pos, nbr.pos)
        if Dij > tx_range:
            continue  # not in range

        # must be shallower (closer to surface)
        if nbr.depth >= src.depth:
            continue

        # must make positive progress
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0:
            continue

        # find common idle channels
        common_idle = np.where((src.channel_state == 0) & (nbr.channel_state == 0))[0]
        if len(common_idle) == 0:
            continue

        # pick one random common channel
        ch = random.choice(common_idle)
        TD = compute_transmission_delay(src, nbr, dest_pos, ch)
        candidates.append((nbr, TD, ch))

    if not candidates:
        return None, None, None

    # choose node with minimum TD
    best = min(candidates, key=lambda x: x[1])
    return best  # (node, TD, channel)

# ---------- Main Simulation ----------
def main():
    N = 30
    rng = np.random.default_rng()
    positions = place_nodes_uniform(N)
    nodes = []
    for i in range(N):
        channel_state = rng.random(M) < P_BUSY
        nodes.append(Node(i, positions[i], channel_state))

    # Destination = buoy at sea surface center
    dest_pos = np.array([AREA/2, AREA/2, 0.0])

    # Pick a deep source node (farthest from surface)
    src = max(nodes, key=lambda n: n.depth)
    print(f"Source: Node {src.node_id}, Depth={src.depth:.1f} m")

    route = [src.node_id]
    current = src
    total_delay = 0

    while True:
        best_nbr, TD, ch = select_best_next_hop(current, nodes, dest_pos, TX_RANGE)
        if best_nbr is None:
            print(f"Node {current.node_id}: No forward neighbor found, route broken.")
            break

        total_delay += TD
        route.append(best_nbr.node_id)
        print(f"→ Node {best_nbr.node_id} (Depth={best_nbr.depth:.1f} m) "
              f"| TD={TD*1e3:.3f} ms | Channel={ch}")

        current = best_nbr
        if current.depth <= 5.0:  # reached surface
            print("Reached surface buoy!")
            break

    print("\nFinal Route:", " → ".join(map(str, route)))
    print(f"Total Transmission Delay: {total_delay*1e3:.3f} ms")

if __name__ == "__main__":
    main()
