# osar_simulation.py
# Self-contained simulation with corrected physics formulas.

import numpy as np
import random
import time
import matplotlib.pyplot as plt

# ---------- Global Parameters ----------
LP = 512                # bits per packet
CH_BANDWIDTH = 6e3      # Hz per channel (Δf)

# ## CORRECTED PT ##
# Realistic transmit power (Source Level of 175 dB re 1 μPa @ 1m)
PT = 10**(175.0 / 10.0)

AREA = 500.0              # cubic region (m)
TX_RANGE = 250.0          # m
P_BUSY = 1.0          # probability a PU/busy subcarrier (for EB)
M = 128                   # number of subcarriers (used for PU channel_state shape)
TEMP = 10                 # degC (for sound speed calc)
SALINITY = 35             # ppt
K_SPREAD = 1.5            # path-loss exponent (spreading factor)
EPS = 1e-12

# ---------- Acoustic helpers (paper-style, modified for meters) ----------
def thorp_absorption_db_per_m(f_khz: float) -> float:
    """
    Thorp absorption (approx) in dB/m for f in kHz.
    Original formula is in dB/km, so we divide by 1000.
    """
    f2 = f_khz**2
    db_per_km = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303
    return db_per_km / 1000.0

def path_loss_linear(d_m: float, f_khz: float, k: float = K_SPREAD) -> float:
    """
    Paper A(d,f) linear: A(d,f) = d^k * a(f)^d where a(f) is linear per-meter absorption.
    """
    if d_m <= 0:
        d_m = EPS
    alpha_db_per_m = thorp_absorption_db_per_m(f_khz)
    a_linear_per_m = 10 ** (alpha_db_per_m / 10.0)
    return (d_m ** k) * (a_linear_per_m ** d_m)

def noise_psd_linear(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    """
    ## CORRECTED NOISE ##
    Convert the noise empirical formula (dB) to linear PSD (power per Hz).
    Noise sources are added in linear space, not dB space.
    """
    # 1. Calculate each noise component in dB
    Nt_db = 17 - 30 * np.log10(f_khz + EPS)
    Ns_db = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f_khz + EPS) - 60 * np.log10(f_khz + 0.03)
    Nw_db = 50 + 7.5 * wind + 20 * np.log10(f_khz + EPS) - 40 * np.log10(f_khz + 0.4)
    Nth_db = -15 + 20 * np.log10(f_khz + EPS)

    # 2. Convert each component from dB to its linear power value
    Nt_linear = 10 ** (Nt_db / 10.0)
    Ns_linear = 10 ** (Ns_db / 10.0)
    Nw_linear = 10 ** (Nw_db / 10.0)
    Nth_linear = 10 ** (Nth_db / 10.0)

    # 3. Sum the linear powers to get the correct total noise power
    return Nt_linear + Ns_linear + Nw_linear + Nth_linear

def sound_speed(z_m: float, s_ppt: float = SALINITY, T_c: float = TEMP) -> float:
    """
    Equation (4) from paper (z in meters, T in Celsius). Returns m/s.
    """
    z_km = z_m / 1000.0  # Convert meters to km for the formula
    T = T_c / 10.0
    return (1449.05 + 45.7 * T - 5.21 * (T**2) + 0.23 * (T**3) +
            (1.333 - 0.126 * T + 0.009 * (T**2)) * (s_ppt - 35) +
            16.3 * z_km + 0.18 * (z_km**2))

# ---------- Utility ----------
def distance(a, b):
    return np.linalg.norm(a - b)

def place_nodes_uniform(N, area=AREA):
    # 3D roughly uniform distribution
    side = int(np.ceil(N ** (1/3)))
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    rng = np.random.default_rng()
    jitter = rng.uniform(-area / (4 * side), area / (4 * side), grid.shape)
    grid = grid + jitter
    return grid[:N]

# ---------- Node ----------
class Node:
    def __init__(self, node_id, pos, channel_state, temp=TEMP, salinity=SALINITY):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = sound_speed(self.depth, salinity, temp)
        self.neighbors = {}

# ---------- EB ----------
def broadcast_all_EBs(nodes, tx_range):
    """Populate each node.neighbors with neighbor info (within tx_range)."""
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
                    "busy_channels": np.where(src.channel_state == True)[0].tolist()
                }

# ---------- Transmission Delay (paper-correct) ----------
def compute_transmission_delay_paper(src: Node,
                                     dest: Node,
                                     dest_pos: np.ndarray,
                                     channel_center_khz: float,
                                     LP_bits: int = LP,
                                     ch_bandwidth_hz: float = CH_BANDWIDTH,
                                     Pt: float = PT,
                                     verbose: bool = False):
    """
    Compute TD_ij^ch as defined in the paper.
    """
    # 1) Distances and geometry
    vec_id = dest_pos - src.pos
    vec_ij = dest.pos - src.pos
    DiD = np.linalg.norm(vec_id) + EPS
    proj_len = np.dot(vec_ij, vec_id) / (DiD + EPS)
    if proj_len <= 0:
        proj_len = EPS

    # estimated number of hops
    N_hop = max(DiD / proj_len, 1.0)

    # 2) Propagation delay uses depth difference (vertical advance)
    depth_diff = max(src.depth - dest.depth, 0.0)
    mid_depth_m = max((src.depth + dest.depth) / 2.0, 0.0)
    c = sound_speed(mid_depth_m)
    PD_ij = depth_diff / (c + EPS)

    # 3) Path loss and noise at the mid-frequency
    dist_m = np.linalg.norm(dest.pos - src.pos)
    dist_m = max(dist_m, EPS)

    A_df = path_loss_linear(dist_m, channel_center_khz, k=K_SPREAD)
    Nf = noise_psd_linear(channel_center_khz)
    noise_total = Nf * ch_bandwidth_hz

    # 4) SNR (linear) per paper
    snr_linear = Pt / (A_df * noise_total + EPS)

    # 5) Rate approximation across channel (bits/sec)
    r_ij_ch = ch_bandwidth_hz * np.log2(1.0 + snr_linear + EPS)

    if r_ij_ch <= 1e-12 or not np.isfinite(r_ij_ch):
        return float('inf'), 0.0, snr_linear

    # 6) final TD
    TD = ((LP_bits / r_ij_ch) + PD_ij) * N_hop

    if verbose:
        print(f"[TD] src={src.node_id} -> dest={dest.node_id} dist_m={dist_m:.1f} "
              f"proj_len={proj_len:.2f} N_hop={N_hop:.2f} PD={PD_ij:.6f}s "
              f"SNR_lin={snr_linear:.3e} rate={r_ij_ch:.1f} TD={TD:.6f}s")
    return TD, r_ij_ch, snr_linear

# ---------- Best Hop ----------
def select_best_next_hop(src: Node, nodes, dest_pos, tx_range, channels_center_khz, verbose=False):
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
        common_idle_idxs = np.where((src.channel_state == False) & (nbr.channel_state == False))[0]
        if len(common_idle_idxs) == 0:
            continue
        best_td = float('inf')
        best_for_pair = None
        best_ch = None
        best_snr = None
        for idx in common_idle_idxs:
            ch_idx = idx % len(channels_center_khz)
            f_center_khz = channels_center_khz[ch_idx]
            TD, rate, snr_lin = compute_transmission_delay_paper(src, nbr, dest_pos, f_center_khz)
            if TD < best_td:
                best_td = TD
                best_for_pair = nbr
                best_ch = ch_idx
                best_snr = snr_lin
        if best_for_pair is not None:
            candidates.append((best_for_pair, best_td, best_ch, best_snr))
    if not candidates:
        return None, None, None, None
    best = min(candidates, key=lambda x: x[1])
    if verbose:
        print(f"[select] src={src.node_id} chosen next={best[0].node_id} TD={best[1]:.6f} Ch={best[2]} SNR={10*np.log10(best[3]+EPS):.2f}dB")
    return best

# ---------- Main Simulation ----------
def main():
    rng = np.random.default_rng(123)
    N = 60 # Increased node density for better connectivity
    positions = place_nodes_uniform(N)
    for i in range(N):
        if positions[i, 2] < 1.0:
            positions[i, 2] = rng.uniform(1.0, AREA/2)

    nodes = [Node(i, positions[i], rng.random(M) < P_BUSY) for i in range(N)]
    dest_pos = np.array([AREA/2, AREA/2, 0.0])
    broadcast_all_EBs(nodes, TX_RANGE)

    src = max(nodes, key=lambda n: n.depth)
    print(f"\nSource: Node {src.node_id}, Depth={src.depth:.1f} m")

    channels_center_khz = np.array([13, 19, 25, 31, 37])
    route = [src.node_id]
    current = src
    total_delay = 0.0
    visited_positions = [current.pos]
    attempt = 0

    while True:
        best_nbr, TD, ch, snr_lin = select_best_next_hop(current, nodes, dest_pos, TX_RANGE, channels_center_khz, verbose=False)

        # ## RE-IMPLEMENTED RETRY/FALLBACK LOGIC ##
        while best_nbr is None:
            attempt += 1
            print(f"🔄 Attempt {attempt}: Re-sensing channels & rebroadcasting EBs...")
            for n in nodes:
                n.channel_state = rng.random(M) < P_BUSY
            broadcast_all_EBs(nodes, TX_RANGE)
            best_nbr, TD, ch, snr_lin = select_best_next_hop(current, nodes, dest_pos, TX_RANGE, channels_center_khz, verbose=False)
            
            # Fallback logic after 100 failed attempts
            if attempt > 100:
                print(f"⚠️ Node {current.node_id} is stuck! Initiating fallback...")
                # Find any node within range, even if it doesn't meet ideal criteria (e.g., shallower)
                nearby = [n for n in nodes if distance(current.pos, n.pos) <= TX_RANGE and n.node_id != current.node_id]
                if not nearby:
                    print(f"   Node {current.node_id} is truly isolated! Forcing jump to shallowest node.")
                    # Exclude self from the list of potential targets
                    other_nodes = [n for n in nodes if n.node_id != current.node_id]
                    if not other_nodes:
                        print("   No other nodes to jump to. Aborting.")
                        return
                    best_nbr = min(other_nodes, key=lambda n: n.depth)
                else:
                    # Of the nearby nodes, pick the one closest to the final destination
                    print(f"   Found {len(nearby)} nearby nodes. Choosing closest to destination.")
                    best_nbr = min(nearby, key=lambda n: distance(n.pos, dest_pos))

                # Compute TD for this forced fallback hop on an arbitrary channel (e.g., channel 0)
                # Create a dummy buoy node for the calculation
                buoy_node = Node(node_id='Buoy', pos=dest_pos, channel_state=np.array([False]*M))
                TD, _, snr_lin = compute_transmission_delay_paper(current, best_nbr, dest_pos, channels_center_khz[0])
                ch = 0
                print(f"⚙️ Fallback: Forced hop to Node {best_nbr.node_id}")
                attempt = 0 # Reset attempt counter after a successful fallback
                break # Exit the inner while loop to proceed with the fallback hop

            time.sleep(0.01) # Small delay to prevent tight loop

        if TD == float('inf'):
            print(f"→ Best candidate yields infinite TD (unusable link). Re-sensing...")
            for n in nodes:
                n.channel_state = rng.random(M) < P_BUSY
            broadcast_all_EBs(nodes, TX_RANGE)
            continue

        total_delay += TD
        route.append(best_nbr.node_id)
        visited_positions.append(best_nbr.pos)
        print(f"→ Node {best_nbr.node_id} | Depth={best_nbr.depth:.1f} m | TD={TD:.4f} s | Ch={ch} | SNR={10*np.log10(snr_lin+EPS):.1f} dB")

        current = best_nbr
        for n in nodes:
            n.channel_state = rng.random(M) < P_BUSY
        broadcast_all_EBs(nodes, TX_RANGE)

        # Check if the current node can reach the buoy, then calculate the final hop.
        if distance(current.pos, dest_pos) < TX_RANGE or current.depth <= 5.0:
            buoy_node = Node(node_id='Buoy', pos=dest_pos, channel_state=np.array([False]*M))
            
            best_final_td = float('inf')
            best_final_ch = -1
            best_final_snr = 0
            idle_idxs = np.where(current.channel_state == False)[0]
            
            if len(idle_idxs) == 0:
                idle_idxs = [0] 
                print(f"⚠️ Node {current.node_id} has no idle channels for final hop, using channel 0 as fallback.")

            for idx in idle_idxs:
                ch_idx = idx % len(channels_center_khz)
                f_center_khz = channels_center_khz[ch_idx]
                
                final_td, _, final_snr = compute_transmission_delay_paper(current, buoy_node, dest_pos, f_center_khz)
                
                if final_td < best_final_td:
                    best_final_td = final_td
                    best_final_ch = ch_idx
                    best_final_snr = final_snr

            total_delay += best_final_td
            route.append("Buoy")
            visited_positions.append(dest_pos)
            print(f"→ Buoy (Surface) | Depth={dest_pos[2]:.1f} m | TD={best_final_td:.4f} s | Ch={best_final_ch} | SNR={10*np.log10(best_final_snr+EPS):.1f} dB")
            
            print("\n✅ Reached surface buoy!")
            break

        if len(route) > 200:
            print("⚠️ Route too long; aborting.")
            break

    print("\nFinal Route:", " → ".join(map(str, route)))
    print(f"Total Transmission Delay: {total_delay:.4f} s")

    # Visualization
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    positions_array = np.array([n.pos for n in nodes])
    depths = [n.depth for n in nodes]
    ax.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
               c=depths, cmap="viridis", s=50, alpha=0.8)
    ax.scatter(dest_pos[0], dest_pos[1], dest_pos[2], c="red", s=120, marker="^", label="Buoy")
    route_pos = np.array(visited_positions)
    ax.plot(route_pos[:, 0], route_pos[:, 1], route_pos[:, 2],
            color="orange", linewidth=3, label="Route")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Depth (m)")
    ax.set_title("OSAR Routing with Corrected Physics")
    ax.invert_zaxis()
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

