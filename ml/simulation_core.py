# simulation_core.py
# Core physics, node logic, and routing algorithms.

import numpy as np
import os
import sys
from simulation_config import * # Import constants
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from auv import (AUV as BaseAUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                     DEFAULT_AUV_RELAY_RADIUS, AUV_COVERAGE_RADIUS, LOW_BATTERY_THRESHOLD, distance,
                     distance as auv_distance)



# --- Acoustic & Utility Functions ---
def thorp_absorption_db_per_m(f_khz):
    f2 = f_khz**2
    return (0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303) / 1000.0

def path_loss_linear(d_m, f_khz, k=K_SPREAD):
    if d_m <= 0: d_m = EPS
    alpha = thorp_absorption_db_per_m(f_khz)
    a_linear = 10 ** (alpha / 10.0)
    return (d_m ** k) * (a_linear ** d_m)

def noise_psd_linear(f_khz, shipping=0.5, wind=0.0):
    f = f_khz + EPS
    Nt = 17 - 30 * np.log10(f)
    Ns = 40 + 20*(shipping-0.5) + 26*np.log10(f) - 60*np.log10(f+0.03)
    Nw = 50 + 7.5*wind + 20*np.log10(f) - 40*np.log10(f+0.4)
    Nth = -15 + 20*np.log10(f)
    return 10**(Nt/10) + 10**(Ns/10) + 10**(Nw/10) + 10**(Nth/10)

def sound_speed(z_m, s_ppt=SALINITY, T_c=TEMP):
    z_km = z_m / 1000.0
    T = T_c / 10.0
    return (1449.05 + 45.7*T - 5.21*(T**2) + 0.23*(T**3) +
            (1.333 - 0.126*T + 0.009*(T**2))*(s_ppt - 35) +
            16.3*z_km + 0.18*(z_km**2))

def distance_fn(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# --- Node Class ---
class Node:
    def __init__(self, node_id, pos, rng, channel_state=None):
        self.node_id = node_id
        self.initial_pos = np.copy(pos)
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.rng = rng
        self.sound_speed = sound_speed(self.depth)
        self.channel_state = channel_state if channel_state is not None else \
                             rng.random(M_SUBCARRIERS) < P_BUSY

    def update_position(self, max_drift_m):
        dx = self.rng.uniform(-max_drift_m, max_drift_m)
        dy = self.rng.uniform(-max_drift_m, max_drift_m)
        self.pos[0] = self.initial_pos[0] + dx
        self.pos[1] = self.initial_pos[1] + dy
        self.depth = self.pos[2]

    def update_channels(self, p_busy):
        self.channel_state = self.rng.random(M_SUBCARRIERS) < p_busy

def place_nodes_randomly(N, area, rng):
    nodes = []
    for i in range(N):
        pos = rng.uniform(0, area, 3)
        pos[2] = rng.uniform(area * 0.1, area)
        nodes.append(Node(i, pos, rng))
    return nodes

def compute_transmission_delay_paper(src, dest, dest_pos_target, f_khz, lp, bw, pt):
    # (Copy the full function logic from your original code here)
    # For brevity, I am summarizing, but you paste the full logic.
    vec_id = dest_pos_target - src.pos
    vec_ij = dest.pos - src.pos
    DiD = np.linalg.norm(vec_id) + EPS
    proj_len = np.dot(vec_ij, vec_id) / DiD
    if proj_len <= EPS: proj_len = EPS
    
    N_hop = max(DiD / proj_len, 1.0)
    # ... (Rest of physics calc) ...
    mid_depth = (src.depth + dest.depth)/2.0
    c = sound_speed(mid_depth)
    PD_ij = abs(src.depth - dest.depth) / (c + EPS)
    
    dist = max(distance_fn(src.pos, dest.pos), EPS)
    A_df = path_loss_linear(dist, f_khz)
    Nf = noise_psd_linear(f_khz)
    snr = pt / (A_df * Nf * bw + EPS)
    
    if snr <= 0: r_ch = EPS
    else: r_ch = bw * np.log2(1.0 + snr)
    
    if r_ch <= EPS: return float('inf'), 0.0, 0.0, A_df
    
    TD = ((lp / r_ch) + PD_ij) * N_hop
    return TD, r_ch, snr, A_df

def select_best_next_hop(src, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng, p):
    candidates = []
    W_time = p.get('W_TIME', DEFAULT_W_TIME)
    W_energy = p.get('W_ENERGY', DEFAULT_W_ENERGY)

    # --- Part 1: Static Nodes ---
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue
        if auv_distance(src.pos, nbr.pos) > tx_range:
            continue
        if nbr.depth >= src.depth:
            continue
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0:
            continue

        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0:
            continue

        best_td = float('inf')
        best_energy = float('inf')
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, _, A_df = compute_transmission_delay_paper(
                src, nbr, dest_pos, f_center_khz, lp, bw, pt)
            if TD < best_td:
                best_td, best_energy = TD, A_df

        if best_td != float('inf'):
            candidates.append((nbr, best_td, best_energy))

    # --- Part 2: Mobile AUVs ---
    # Only consider AUVs if USE_AUVS flag is True
    if p.get('USE_AUVS', True):
        for auv in auvs:
            if not (auv.is_lc or auv.is_oc):
                continue
            if auv_distance(src.pos, auv.current_pos) > tx_range:
                continue
            if auv.current_pos[2] >= src.depth:
                continue
            if np.dot(dest_pos - src.pos, auv.current_pos - src.pos) <= 0:
                continue

            common_idle = np.where(~src.channel_state)[0]
            if len(common_idle) == 0:
                continue

            dummy_auv = Node(f"AUV-{auv.id}", auv.current_pos, rng)
            best_td = float('inf')
            best_energy = float('inf')

            for idx in common_idle:
                ch_idx = idx % len(channels)
                f_center_khz = channels[ch_idx]
                TD, _, _, A_df = compute_transmission_delay_paper(
                    src, dummy_auv, dest_pos, f_center_khz, lp, bw, pt)
                if TD < best_td:
                    best_td = TD
                    best_energy = A_df + AUV_HOP_ENERGY_PENALTY

            if best_td != float('inf'):
                candidates.append((dummy_auv, best_td, best_energy))

    if not candidates:
        return None, None, None, None

    # Normalize and Select
    max_td = max((c[1]
                 for c in candidates if c[1] != float('inf')), default=1.0)
    max_en = max((c[2]
                 for c in candidates if c[2] != float('inf')), default=1.0)

    best_hop = None
    min_cost = float('inf')

    for (obj, td, en) in candidates:
        cost = (W_time * (td/max_td)) + (W_energy * (en/max_en))
        if cost < min_cost:
            min_cost = cost
            best_hop = (obj, td, None, None)

    return best_hop