import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ---------- Parameters ----------
LP = 512          # bits
BANDWIDTH = 6e3   # Hz
PT = 1.0          # W
NOISE_PSD = 1e-9  # W/Hz
AREA = 500.0
TX_RANGE = 250.0
P_BUSY = 0.1
M = 128
TEMP = 10
SALINITY = 35

# ---------- Utility Functions ----------
def calc_sound_speed(temp=TEMP, salinity=SALINITY, depth=100):
    """Empirical sound speed equation (m/s)."""
    return 1448.96 + 4.591 * temp - 0.05304 * (temp ** 2) + 0.0163 * depth + 1.34 * (salinity - 35)

def distance(a, b):
    return np.linalg.norm(a - b)

def place_nodes_uniform(N, area=AREA):
    """Place N nodes evenly in a 3D grid with slight random jitter."""
    side = int(np.ceil(N ** (1/3)))  # ensure enough grid points
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    rng = np.random.default_rng()
    grid += rng.uniform(-area / (4*side), area / (4*side), grid.shape)  # small jitter
    return grid[:N] 

# ---------- Node Class ----------
class Node:
    def __init__(self, node_id, pos, channel_state):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = calc_sound_speed(depth=self.depth)

# ---------- Transmission Delay ----------
def compute_transmission_delay(src, dest, dest_pos, channel_idx):
    """
    Implements Eq. (9) from the paper:
    TD_ij^ch = LP / r_ij^ch + PD_ij * N_ij^hop
    where r_ij^ch = ∫ log2(1 + S(f)/N(f)) df (approximated),
    PD_ij = D_ij / c(z,s,T), and
    N_ij^hop = max(D_iD / <D_ij>_iD, 1).
    """
    # --- Step 1: Propagation-related terms ---
    Dij = np.linalg.norm(src.pos - dest.pos)
    sound_speed = src.sound_speed
    PDij = Dij / sound_speed  # Propagation delay

    # --- Step 2: Hop estimation (Eq. definition) ---
    DiD = np.linalg.norm(dest_pos - src.pos)   # distance from current node to destination
    vec_sd = dest_pos - src.pos                # source → destination vector
    vec_sj = dest.pos - src.pos                # source → neighbor vector
    proj = np.dot(vec_sj, vec_sd) / (np.linalg.norm(vec_sd) + 1e-12)
    avg_proj = max(proj, 1e-6)
    N_hop = max(DiD / avg_proj, 1)

    # --- Step 3: Data rate computation (integral approximation) ---
    # Assume each subcarrier is 46.875 Hz wide (6 kHz / 128 subcarriers)
    ch_bw = BANDWIDTH / M
    f_l = channel_idx * ch_bw
    f_h = f_l + ch_bw

    # Assume power spectral density S(f) = PT / BANDWIDTH (flat)
    # and N(f) = NOISE_PSD (flat noise)
    S_f = PT / BANDWIDTH
    N_f = NOISE_PSD

    # Approximate integral using midpoint frequency
    f_mid = (f_l + f_h) / 2
    SNR_f = S_f / (N_f * (Dij ** 2 + 1))  # attenuation effect
    r_ij_ch = ch_bw * np.log2(1 + SNR_f)  # approximate ∫ over bandwidth

    # --- Step 4: Transmission delay ---
    TD_ij = (LP / r_ij_ch) + (PDij * N_hop)
    return TD_ij

# ---------- Routing Function ----------
def select_best_next_hop(src, nodes, dest_pos, tx_range):
    """Select neighbor j ∈ N_i with minimum TD_ij."""
    candidates = []
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue
        Dij = distance(src.pos, nbr.pos)
        if Dij > tx_range:
            continue
        if nbr.depth >= src.depth:
            continue
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0:
            continue
        common_idle = np.where((src.channel_state == 0) & (nbr.channel_state == 0))[0]
        if len(common_idle) == 0:
            continue
        ch = random.choice(common_idle)
        TD = compute_transmission_delay(src, nbr, dest_pos, ch)
        candidates.append((nbr, TD, ch))
    if not candidates:
        return None, None, None
    best = min(candidates, key=lambda x: x[1])
    return best

# ---------- Main Simulation ----------
def main():
    N = 30
    rng = np.random.default_rng()
    positions = place_nodes_uniform(N)
    nodes = [Node(i, positions[i], rng.random(M) < P_BUSY) for i in range(N)]

    dest_pos = np.array([AREA / 2, AREA / 2, 0.0])
    src = max(nodes, key=lambda n: n.depth)
    print(f"Source: Node {src.node_id}, Depth={src.depth:.1f} m")

    route = [src.node_id]
    current = src
    total_delay = 0
    visited_positions = [current.pos]

    while True:
        best_nbr, TD, ch = select_best_next_hop(current, nodes, dest_pos, TX_RANGE)

        # Retry until success — never give up
        while best_nbr is None:
            print(f"No neighbor found from Node {current.node_id}. Re-sensing channels...")
            for n in nodes:
                n.channel_state = rng.random(M) < P_BUSY
            best_nbr, TD, ch = select_best_next_hop(current, nodes, dest_pos, TX_RANGE)
            time.sleep(0.3)

        total_delay += TD
        route.append(best_nbr.node_id)
        visited_positions.append(best_nbr.pos)
        print(f"→ Node {best_nbr.node_id} | Depth={best_nbr.depth:.1f} m | "
              f"TD={TD*1e3:.3f} ms | Channel={ch}")

        current = best_nbr

        # Update channel states (simulate re-sensing)
        for n in nodes:
            n.channel_state = rng.random(M) < P_BUSY

        if current.depth <= 5.0:
            print("\nReached surface buoy!")
            break

    print("\nFinal Route:", " → ".join(map(str, route)))
    print(f"Total Transmission Delay: {total_delay*1e3:.3f} ms")

    # ---------- Visualization ----------
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    depths = [n.depth for n in nodes]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=depths, cmap="viridis", s=50, alpha=0.6, label="All Nodes")
    ax.scatter(dest_pos[0], dest_pos[1], dest_pos[2],
               c="red", s=120, marker="^", label="Surface Buoy")

    route_positions = np.array(visited_positions)
    ax.plot(route_positions[:, 0], route_positions[:, 1], route_positions[:, 2],
            color="orange", linewidth=3, label="Established Route")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("OSAR Routing Path in Evenly Distributed Underwater Network")
    ax.legend()
    ax.invert_zaxis()
    plt.show()

if __name__ == "__main__":
    main()
