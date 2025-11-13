# common.py
# --- Common utilities, parameters, and classes for the OSAR simulation ---

import numpy as np
import os
import sys

# --- FIX: Add path correction to find 'auv2.py' in parent dir ---
# Get the absolute path of this script's directory (e.g., d:/OSAR/ml/simulation)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the 'ml' directory (e.g., d:/OSAR/ml)
ml_dir = os.path.dirname(script_dir)
# Get the path to the project root directory (e.g., d:/OSAR)
project_root = os.path.dirname(ml_dir)
# Add the project root directory (OSAR) to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of fix ---

# --- Import AUV components ---
try:
    from auv2 import (AUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                      AUV_WAYPOINTS, AUV_COVERAGE_RADIUS, DEFAULT_AUV_RELAY_RADIUS,
                      distance as auv_distance)
except ImportError:
    print(f"Error: Could not import 'auv2.py' from root directory: {project_root}")
    print("Please ensure 'auv2.py' (or 'auv.py') exists in that folder.")
    sys.exit()


# ---------- Global Parameters ----------
LP = 512
CONTROL_PACKET_LP = 64
CH_BANDWIDTH = 6e3
DEFAULT_PT_DB = 150.0
AREA = 500.0
TX_RANGE = 250.0
P_BUSY = 0.8
N_NODES = 30
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10
DEFAULT_NUM_PACKETS = 10
DEFAULT_NUM_AUVS = 5
AUV_UPDATE_INTERVAL_S = 0.1

# --- New MC / SVM Parameters ---
DEFAULT_NUM_LCS_TO_SELECT = 3
SVM_UPDATE_PERIOD_S = 5

# --- NEW: Energy Cost Ratio (ECR) Constants ---
E_BIT_TX = 1.0
E_AUV_MOVE_PER_S = 1_000_000.0

# --- Environmental Parameters (Assumed) ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5
EPS = 1e-12

# ---------- Acoustic & Utility Functions ---

def thorp_absorption_db_per_m(f_khz: float) -> float:
    f2 = f_khz**2
    db_per_km = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303
    return db_per_km / 1000.0

def path_loss_linear(d_m: float, f_khz: float, k: float = K_SPREAD) -> float:
    if d_m <= 0: d_m = EPS
    alpha_db_per_m = thorp_absorption_db_per_m(f_khz)
    a_linear_per_m = 10 ** (alpha_db_per_m / 10.0)
    return (d_m ** k) * (a_linear_per_m ** d_m)

def noise_psd_linear(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    f_khz_safe = f_khz + EPS
    Nt_db = 17 - 30 * np.log10(f_khz_safe)
    Ns_db = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f_khz_safe) - 60 * np.log10(f_khz_safe + 0.03)
    Nw_db = 50 + 7.5 * wind + 20 * np.log10(f_khz_safe) - 40 * np.log10(f_khz_safe + 0.4)
    Nth_db = -15 + 20 * np.log10(f_khz_safe)
    Nt_linear, Ns_linear = 10**(Nt_db/10.0), 10**(Ns_db/10.0)
    Nw_linear, Nth_linear = 10**(Nw_db/10.0), 10**(Nth_db/10.0)
    return Nt_linear + Ns_linear + Nw_linear + Nth_linear

def sound_speed(z_m: float, s_ppt: float = SALINITY, T_c: float = TEMP) -> float:
    z_km = z_m / 1000.0
    T = T_c / 10.0
    return (1449.05 + 45.7*T - 5.21*(T**2) + 0.23*(T**3) +
            (1.333 - 0.126*T + 0.009*(T**2))*(s_ppt - 35) +
            16.3*z_km + 0.18*(z_km**2))

def place_nodes_uniform(N, area, rng):
    side = int(np.ceil(N**(1/3)))
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    final_nodes = grid[:N]
    jitter = rng.uniform(-area / (4 * side), area / (4 * side), final_nodes.shape)
    final_nodes = np.clip(final_nodes + jitter, 0, area)
    final_nodes[:, 2] = np.maximum(final_nodes[:, 2], EPS)
    return final_nodes

# ---------- Shared Class Definition ----------

class Node:
    def __init__(self, node_id, pos, channel_state):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = sound_speed(self.depth)

# ---------- Core Routing Logic Functions ----------

def compute_transmission_delay_paper(src, dest, dest_pos_target, f_khz, lp, bw, pt):
    src_pos = np.array(src.pos)
    dest_pos_actual = np.array(dest.pos)
    dest_pos_final_buoy = np.array(dest_pos_target)
    
    vec_id = dest_pos_final_buoy - src_pos
    vec_ij = dest_pos_actual - src_pos
    
    DiD = np.linalg.norm(vec_id)
    if DiD < EPS: DiD = EPS
    
    if np.linalg.norm(vec_ij) < EPS:
        if isinstance(dest, Node) and dest.node_id == 'Buoy': proj_len = DiD
        else: proj_len = EPS
    else:
        proj_len = np.dot(vec_ij, vec_id) / DiD
        
    if proj_len <= EPS: proj_len = EPS
    
    N_hop = max(DiD / proj_len, 1.0)
    
    src_depth = src.depth if isinstance(src.depth, (int, float)) and np.isfinite(src.depth) else AREA
    dest_depth = 0.0 if isinstance(dest, Node) and dest.node_id == 'Buoy' else (dest.depth if isinstance(dest.depth, (int, float)) and np.isfinite(dest.depth) else AREA)
    
    depth_diff = max(src_depth - dest_depth, 0.0)
    mid_depth_m = max((src_depth + dest_depth) / 2.0, 0.0)
    c = sound_speed(mid_depth_m)
    if c < EPS: c = 1500.0
    
    PD_ij = depth_diff / (c + EPS)
    dist_m = max(auv_distance(src_pos, dest_pos_actual), EPS)
    
    A_df = path_loss_linear(dist_m, f_khz)
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    if noise_total < EPS: noise_total = EPS
    
    pt_lin = max(pt, EPS)
    snr_linear = pt_lin / (A_df * noise_total + EPS)
    
    if snr_linear <= 0: r_ij_ch = EPS
    else: r_ij_ch = bw * np.log2(1.0 + snr_linear)
        
    if r_ij_ch <= EPS: return float('inf'), 0.0, 0.0
    
    TD = ((lp / r_ij_ch) + PD_ij) * N_hop
    return TD, r_ij_ch, snr_linear


def select_best_next_hop(src, nodes, dest_pos, tx_range, channels, lp, bw, pt):
    candidates = []
    src_pos = np.array(src.pos)
    dest_pos_target = np.array(dest_pos)
    
    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        
        nbr_pos = np.array(nbr.pos)
        if auv_distance(src_pos, nbr_pos) > tx_range: continue
        
        src_depth = src.depth if isinstance(src.depth, (int, float)) and np.isfinite(src.depth) else AREA
        nbr_depth = nbr.depth if isinstance(nbr.depth, (int, float)) and np.isfinite(nbr.depth) else AREA
        if nbr_depth >= src_depth: continue
        
        if np.dot(dest_pos_target - src_pos, nbr_pos - src_pos) <= 0: continue
        
        src_ch_state = np.array(src.channel_state, dtype=bool)
        nbr_ch_state = np.array(nbr.channel_state, dtype=bool)
        common_idle = np.where(~src_ch_state & ~nbr_ch_state)[0]
        
        if len(common_idle) == 0: continue
        
        best_td_for_nbr, best_ch_for_nbr, best_snr_for_nbr = float('inf'), None, None
        
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, snr_lin = compute_transmission_delay_paper(src, nbr, dest_pos_target, f_center_khz, lp, bw, pt)
            
            if TD < best_td_for_nbr:
                best_td_for_nbr = TD
                best_ch_for_nbr = ch_idx
                best_snr_for_nbr = snr_lin
                
        if best_td_for_nbr != float('inf'):
            candidates.append((nbr, best_td_for_nbr, best_ch_for_nbr, best_snr_for_nbr))
            
    if not candidates: return None, None, None, None
    
    return min(candidates, key=lambda x: x[1])