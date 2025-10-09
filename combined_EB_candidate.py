import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ---------- Global Parameters ----------
LP = 512
BANDWIDTH = 6e3
PT = 1.0
NOISE_PSD = 1e-9
AREA = 500.0
TX_RANGE = 150.0
P_BUSY = 0.7
M = 128
TEMP = 10
SALINITY = 35

# ---------- Utility ----------
def calc_sound_speed(temp=TEMP, salinity=SALINITY, depth=100):
    return 1448.96 + 4.591 * temp - 0.05304 * (temp**2) + 0.0163 * depth + 1.34 * (salinity - 35)

def distance(a, b):
    return np.linalg.norm(a - b)

def place_nodes_uniform(N, area=AREA):
    side = int(np.ceil(N ** (1/3)))
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    rng = np.random.default_rng()
    grid += rng.uniform(-area / (4*side), area / (4*side), grid.shape)
    return grid[:N]

# ---------- Node ----------
class Node:
    def __init__(self, node_id, pos, channel_state, temp=TEMP, salinity=SALINITY):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = calc_sound_speed(temp, salinity, self.depth)
        self.neighbors = {}

# ---------- EB ----------
def broadcast_all_EBs(nodes, tx_range):
    for node in nodes:
        node.neighbors.clear()
    for src in nodes:
        for nbr in nodes:
            if nbr.node_id == src.node_id:
                continue
            dist = distance(src.pos, nbr.pos)
            if dist <= tx_range:
                nbr.neighbors[src.node_id] = {
                    "depth": src.depth,
                    "pos": src.pos,
                    "sound_speed": src.sound_speed,
                    "distance": dist,
                    "busy_channels": np.where(src.channel_state == 1)[0]
                }

# ---------- Transmission Delay ----------
def compute_transmission_delay(src, dest, dest_pos, channel_idx):
    Dij = np.linalg.norm(src.pos - dest.pos)
    PDij = Dij / src.sound_speed
    DiD = np.linalg.norm(dest_pos - src.pos)
    vec_sd = dest_pos - src.pos
    vec_sj = dest.pos - src.pos
    proj = np.dot(vec_sj, vec_sd) / (np.linalg.norm(vec_sd) + 1e-12)
    avg_proj = max(proj, 1e-6)
    N_hop = max(DiD / avg_proj, 1)
    ch_bw = BANDWIDTH / M
    SNR_f = (PT / BANDWIDTH) / (NOISE_PSD * (Dij**2 + 1))
    r_ij_ch = ch_bw * np.log2(1 + SNR_f)
    return (LP / r_ij_ch) + (PDij * N_hop)

# ---------- Best Hop ----------
def select_best_next_hop(src, nodes, dest_pos, tx_range):
    candidates = []
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue
        Dij = distance(src.pos, nbr.pos)
        if Dij > tx_range:
            continue
        # Allow small upward moves (within 5m)
        if nbr.depth > src.depth + 5:
            continue
        # Check positive progress
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0:
            continue
        # Check common idle channels
        common_idle = np.where((src.channel_state == 0) & (nbr.channel_state == 0))[0]
        if len(common_idle) == 0:
            continue
        ch = random.choice(common_idle)
        TD = compute_transmission_delay(src, nbr, dest_pos, ch)
        candidates.append((nbr, TD, ch))
    if not candidates:
        return None, None, None
    return min(candidates, key=lambda x: x[1])

# ---------- Main Simulation ----------
def main():
    rng = np.random.default_rng()
    N = 30
    positions = place_nodes_uniform(N)
    nodes = [Node(i, positions[i], rng.random(M) < P_BUSY) for i in range(N)]

    dest_pos = np.array([AREA/2, AREA/2, 0.0])
    broadcast_all_EBs(nodes, TX_RANGE)

    src = max(nodes, key=lambda n: n.depth)
    print(f"\nSource: Node {src.node_id}, Depth={src.depth:.1f} m")

    route = [src.node_id]
    current = src
    total_delay = 0
    visited_positions = [current.pos]
    attempt = 0

    while True:
        best_nbr, TD, ch = select_best_next_hop(current, nodes, dest_pos, TX_RANGE)

        # Retry until success, with fallback
        while best_nbr is None:
            attempt += 1
            print(f"🔄 Attempt {attempt}: Re-sensing channels & rebroadcasting EBs...")
            # update channels
            for n in nodes:
                n.channel_state = rng.random(M) < P_BUSY
            broadcast_all_EBs(nodes, TX_RANGE)
            best_nbr, TD, ch = select_best_next_hop(current, nodes, dest_pos, TX_RANGE)

            # fallback after too many tries
            if attempt > 100:
                nearby = [n for n in nodes if distance(current.pos, n.pos) <= TX_RANGE and n.node_id != current.node_id]
                if not nearby:
                    print(f"⚠️ Node {current.node_id} isolated! Forcing upward jump.")
                    best_nbr = min(nodes, key=lambda n: n.depth)
                else:
                    best_nbr = min(nearby, key=lambda n: distance(n.pos, dest_pos))
                TD = compute_transmission_delay(current, best_nbr, dest_pos, random.randint(0, M-1))
                ch = 0
                print(f"⚙️ Fallback: forced hop to Node {best_nbr.node_id}")
                attempt = 0
                break
            time.sleep(0.05)

        # Successful hop
        total_delay += TD
        route.append(best_nbr.node_id)
        visited_positions.append(best_nbr.pos)
        print(f"→ Node {best_nbr.node_id} | Depth={best_nbr.depth:.1f} m | TD={TD*1e3:.3f} ms | Ch={ch}")

        current = best_nbr
        for n in nodes:
            n.channel_state = rng.random(M) < P_BUSY
        broadcast_all_EBs(nodes, TX_RANGE)

        if current.depth <= 5.0:
            print("\n✅ Reached surface buoy!")
            break

    print("\nFinal Route:", " → ".join(map(str, route)))
    print(f"Total Transmission Delay: {total_delay*1e3:.3f} ms")

    # Visualization
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    depths = [n.depth for n in nodes]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=depths, cmap="viridis", s=50, alpha=0.6)
    ax.scatter(dest_pos[0], dest_pos[1], dest_pos[2], c="red", s=120, marker="^")
    route_pos = np.array(visited_positions)
    ax.plot(route_pos[:, 0], route_pos[:, 1], route_pos[:, 2],
            color="orange", linewidth=3, label="Route")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Robust OSAR Routing with Fallback Logic")
    ax.invert_zaxis()
    plt.show()

if __name__ == "__main__":
    main()
