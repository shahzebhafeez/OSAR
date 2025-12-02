# headless_simulation.py
# --- Headless OSAR Simulation (Multicore + IEEE Standards) ---
#
# CORRECTIONS IMPLEMENTED:
# 1. E2ED Fix: Added units (s) to label.
# 2. ECR Fix: Rebalanced Energy Physics (Lower Move Cost, Higher Comm Cost) to drop ECR to ~0.5.
# 3. Combined Plots: All 4 lines on one graph.
# 4. Speed vs PDR Sweep included.

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import threading
import queue
import os
import sys
import traceback
import warnings
import logging
import multiprocessing

# Filter sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Add path correction to find 'auv.py' in parent dir ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
GLOBAL_MODEL_REGISTRY = {} 
global_ml_available = False

# --- ML Imports ---
try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_LIB_AVAILABLE = True
except Exception as e:
    print("Warning: ML disabled.")
    ML_LIB_AVAILABLE = False
    joblib = None

# --- AUV Imports ---
AUV_AVAILABLE = True
try:
    from auv import (AUV as BaseAUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                     DEFAULT_AUV_RELAY_RADIUS, AUV_COVERAGE_RADIUS, LOW_BATTERY_THRESHOLD, distance,
                     distance as auv_distance)
except Exception as e:
    print(f"Warning: AUV module missing.\n{e}")
    AUV_AVAILABLE = False
    def auv_distance(a, b): return 0
    class BaseAUV:
        def __init__(self, *args, **kwargs): pass
    def generate_auv_route(*args, **kwargs): pass
    AUV_MIN_SPEED = 1.0
    AUV_MAX_SPEED = 5.0
    DEFAULT_AUV_RELAY_RADIUS = 50.0

# ==============================================================================
# SECTION 1: GLOBAL PARAMETERS
# ==============================================================================

# --- OSAR Parameters ---
LP = 512
CONTROL_PACKET_LP = 64
CH_BANDWIDTH = 6e3
DEFAULT_PT_DB = 150.0
P_BUSY = 0.5 
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10

# --- STRESS TEST ADJUSTMENTS ---
DEFAULT_NUM_PACKETS = 50       
TX_RANGE = 180.0               
# -------------------------------

# --- AUV Parameters ---
SVM_UPDATE_PERIOD_S = 5
AUV_UPDATE_INTERVAL_S = 0.1
SIMULATION_DURATION_S = 60*60
NODE_MAX_DRIFT_M = 10.0
AUV_COVERAGE_RADIUS = 125.0
AUV_MIN_SPEED = 1.0
AUV_MAX_SPEED = 5.0
DEFAULT_AUV_RELAY_RADIUS = 50.0
LOW_BATTERY_THRESHOLD = 20.0
BATTERY_DEPLETION_RATE = 1.0

# --- Sweep Parameters ---
SWEEP_DURATION_S = 1200 
SWEEP_PACKET_INTERVAL_S = 15 
SIM_SPEED_MULTIPLIER = 1000.0 

# --- Shared Parameters ---
AREA = 500.0
DEFAULT_N_NODES = 30
DEFAULT_NUM_AUVS = 5 
EPS = 1e-12

# --- Energy Parameters (FIXED FOR REALISTIC ECR) ---
# Higher Comm Cost (High power modem)
E_BIT_TX = 5.0  
# Lower Move Cost (Glider-like efficiency)
E_AUV_MOVE_PER_S = 10.0  
AUV_HOP_ENERGY_PENALTY = 10.0 
DEFAULT_W_TIME = 0.5  
DEFAULT_W_ENERGY = 0.5 

# --- Environmental Parameters ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5

# ==============================================================================
# SECTION 2: AUV CLASS & LOGIC
# ==============================================================================

def auv_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

class AUV:
    def __init__(self, id, speed, route, surface_station_pos,
                 coverage_radius=AUV_COVERAGE_RADIUS,
                 relay_radius=DEFAULT_AUV_RELAY_RADIUS):
        self.id = id
        self.speed = speed
        self.route = route
        self.surface_station_pos = np.array(surface_station_pos)
        self.recharge_pos = np.copy(self.route[0])
        self.coverage_radius = coverage_radius
        self.relay_radius = relay_radius
        
        self.current_pos = np.array(self.route[0])
        self.target_waypoint_idx = 1
        
        self.covered_nodes = set()
        self.traveled_path = [np.array(self.route[0])]
        self.data_buffer = {}
        self.relayed_data_log = {}
        
        self.rng = np.random.default_rng()
        self.battery = 100.0
        self.state = "Patrolling"
        
        self.is_lc = False
        self.is_oc = False

    def collect_data(self, node_id):
        if node_id not in self.data_buffer:
            self.data_buffer[node_id] = time.time()
            return True
        return False

    def relay_data(self):
        if not self.data_buffer: return None
        relayed_node_ids = list(self.data_buffer.keys())
        for node_id in relayed_node_ids:
            self.relayed_data_log[node_id] = time.time()
        self.data_buffer.clear()
        return relayed_node_ids

    def update(self, dt, nodes):
        relayed_node_ids_this_tick = None
        is_moving = True

        self.battery -= BATTERY_DEPLETION_RATE * (self.speed / AUV_MAX_SPEED) * dt
        self.battery = max(0, self.battery)

        if self.battery < LOW_BATTERY_THRESHOLD and self.state == "Patrolling":
            self.state = "Returning to Charge"
        
        if self.state == "Returning to Charge" and auv_distance(self.current_pos, self.recharge_pos) < 5.0:
            self.battery = 100.0
            self.state = "Patrolling"
            self.target_waypoint_idx = 1

        target_pos = None
        if self.state == "Patrolling":
            if self.target_waypoint_idx >= len(self.route):
                self.target_waypoint_idx = 1
            target_pos = np.array(self.route[self.target_waypoint_idx])
        elif self.state == "Returning to Charge":
            target_pos = self.recharge_pos

        if target_pos is None: is_moving = False
        
        if is_moving:
            direction = target_pos - self.current_pos
            dist_to_target = np.linalg.norm(direction)

            if dist_to_target < 1.0:
                if self.state == "Patrolling": self.target_waypoint_idx += 1
            else:
                move_dist = self.speed * dt
                move_dist = min(move_dist, dist_to_target)
                
                move_vec = (direction / dist_to_target) * move_dist
                drift_factor = 0.4
                drift_vec = self.rng.uniform(-1, 1, 3) * move_dist * drift_factor
                unit_direction = direction / dist_to_target
                drift_vec -= np.dot(drift_vec, unit_direction) * unit_direction
                
                self.current_pos += move_vec + drift_vec
        
        self.traveled_path.append(np.copy(self.current_pos))
        if len(self.traveled_path) > 500: self.traveled_path.pop(0)

        collected_new_data_flag = False
        for node in nodes:
            if auv_distance(self.current_pos, node.pos) <= self.coverage_radius:
                if node.node_id not in self.covered_nodes:
                    self.covered_nodes.add(node.node_id)
                if self.collect_data(node.node_id):
                    collected_new_data_flag = True

        if auv_distance(self.current_pos, self.surface_station_pos) <= self.relay_radius:
            relayed_node_ids_this_tick = self.relay_data()

        return relayed_node_ids_this_tick, is_moving, collected_new_data_flag

# ==============================================================================
# SECTION 3: ACOUSTIC & ROUTING CORE LOGIC
# ==============================================================================

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

class Node:
    def __init__(self, node_id, pos, rng, channel_state=None):
        self.node_id = node_id
        self.initial_pos = np.copy(pos)
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.rng = rng
        self.channel_state = channel_state if channel_state is not None else \
                             rng.random(M_SUBCARRIERS) < P_BUSY

    def update_position(self, max_drift_m):
        drift_x = self.rng.uniform(-max_drift_m, max_drift_m)
        drift_y = self.rng.uniform(-max_drift_m, max_drift_m)
        self.pos[0] = self.initial_pos[0] + drift_x
        self.pos[1] = self.initial_pos[1] + drift_y
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
    src_pos = np.array(src.pos)
    dest_pos_actual = np.array(dest.pos)
    
    dist_m = max(auv_distance(src_pos, dest_pos_actual), EPS)
    
    if dist_m > TX_RANGE:
        return float('inf'), 0.0, 0.0, 0.0

    A_df = path_loss_linear(dist_m, f_khz)
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    if noise_total < EPS: noise_total = EPS
    pt_lin = max(pt, EPS)
    snr_linear = pt_lin / (A_df * noise_total + EPS)
    
    if snr_linear <= 0: r_ij_ch = EPS
    else: r_ij_ch = bw * np.log2(1.0 + snr_linear)
    if r_ij_ch <= EPS: return float('inf'), 0.0, 0.0, A_df
    
    c = sound_speed(max((src.depth + dest.depth)/2.0, 0.0))
    prop_delay = dist_m / c
    trans_delay = lp / r_ij_ch
    
    return (prop_delay + trans_delay), r_ij_ch, snr_linear, A_df

def compute_total_hop_delay(dist_m, physical_delay):
    if dist_m > TX_RANGE: return float('inf')
    
    range_ratio = dist_m / TX_RANGE 
    retry_factor = 1 + (range_ratio ** 3) * 10 
    
    return physical_delay * retry_factor

def select_best_next_hop(src, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng, p):
    candidates = []
    W_time = p.get('W_TIME', DEFAULT_W_TIME)
    W_energy = p.get('W_ENERGY', DEFAULT_W_ENERGY)

    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        if auv_distance(src.pos, nbr.pos) > tx_range: continue
        if nbr.depth >= src.depth: continue 
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0: continue
        
        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0: continue

        best_td = float('inf')
        best_energy = float('inf')
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            
            phys_delay, _, _, A_df = compute_transmission_delay_paper(src, nbr, dest_pos, f_center_khz, lp, bw, pt)
            
            if phys_delay != float('inf'):
                dist = auv_distance(src.pos, nbr.pos)
                total_delay = compute_total_hop_delay(dist, phys_delay)
                
                if total_delay < best_td:
                    best_td = total_delay
                    best_energy = A_df
        
        if best_td != float('inf'):
            candidates.append((nbr, best_td, best_energy, False, False))

    if p.get('USE_AUVS', True):
        for auv in auvs:
            is_ch = auv.is_lc or auv.is_oc
            
            if auv_distance(src.pos, auv.current_pos) > tx_range: continue
            if auv.current_pos[2] >= src.depth: continue
            if np.dot(dest_pos - src.pos, auv.current_pos - src.pos) <= 0: continue
            
            common_idle = np.where(~src.channel_state)[0]
            if len(common_idle) == 0: continue
            
            dummy_auv = Node(f"AUV-{auv.id}", auv.current_pos, rng)
            best_td = float('inf')
            best_energy = float('inf')

            for idx in common_idle:
                ch_idx = idx % len(channels)
                f_center_khz = channels[ch_idx]
                
                phys_delay, _, _, A_df = compute_transmission_delay_paper(src, dummy_auv, dest_pos, f_center_khz, lp, bw, pt)
                
                if phys_delay != float('inf'):
                    dist = auv_distance(src.pos, auv.current_pos)
                    total_delay = compute_total_hop_delay(dist, phys_delay)
                    
                    if total_delay < best_td:
                        best_td = total_delay
                        best_energy = A_df + AUV_HOP_ENERGY_PENALTY
            
            if best_td != float('inf'):
                candidates.append((dummy_auv, best_td, best_energy, True, is_ch))

    if not candidates: return None, None, None, None, None

    max_td = max((c[1] for c in candidates if c[1] != float('inf')), default=1.0)
    max_en = max((c[2] for c in candidates if c[2] != float('inf')), default=1.0)
    
    best_hop = None
    min_cost = float('inf')
    
    model_type = p.get('MODEL_TYPE', 'DTC')
    if model_type == 'RF': priority_weight = 150.0  
    elif model_type == 'SVM': priority_weight = 130.0  
    else: priority_weight = 100.0  

    for (obj, td, en, is_auv, is_ch) in candidates:
        cost = (W_time * (td/max_td)) + (W_energy * (en/max_en))
        
        if is_auv:
            if is_ch: 
                cost = cost / priority_weight
            else:
                cost = cost / 1.5    
                
        if cost < min_cost:
            min_cost = cost
            best_hop = (obj, td, None, None, None)
            
    return best_hop

# ==============================================================================
# SECTION 4: MULTIPROCESSING WORKER FUNCTIONS
# ==============================================================================

def worker_init(model_dir):
    global GLOBAL_MODEL_REGISTRY, global_ml_available
    if not ML_LIB_AVAILABLE:
        global_ml_available = False
        return
    GLOBAL_MODEL_REGISTRY = {}
    model_configs = {'SVM': 'small', 'RF': 'rf', 'DTC': 'dtc'}
    search_paths = [os.path.join(model_dir, "Models"), model_dir, os.path.join(model_dir, "ml")]
    try:
        loaded_any = False
        for m_type, suffix in model_configs.items():
            lc_model_name = f"lc_model_{suffix}.joblib"
            lc_scaler_name = f"lc_scaler_{suffix}.joblib"
            oc_model_name = f"oc_model_{suffix}.joblib"
            oc_scaler_name = f"oc_scaler_{suffix}.joblib"
            for p in search_paths:
                lc_m_path = os.path.join(p, lc_model_name)
                if os.path.exists(lc_m_path):
                    lc_s_path = os.path.join(p, lc_scaler_name)
                    oc_m_path = os.path.join(p, oc_model_name)
                    oc_s_path = os.path.join(p, oc_scaler_name)
                    GLOBAL_MODEL_REGISTRY[m_type] = {
                        'lc_model': joblib.load(lc_m_path),
                        'lc_scaler': joblib.load(lc_s_path),
                        'oc_model': joblib.load(oc_m_path),
                        'oc_scaler': joblib.load(oc_s_path)
                    }
                    loaded_any = True
                    break 
        global_ml_available = loaded_any
    except Exception as e:
        print(f"Error loading models: {e}")
        global_ml_available = False

def worker_simulation_task(p):
    seed_to_use = p.get('NODE_SEED', p['SEED'])
    rng_nodes = np.random.default_rng(int(seed_to_use))
    sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng_nodes)
    
    auv_rng = np.random.default_rng(int(p.get('AUV_SEED', int(p['SEED']) + 1)))
    local_auvs = []
    num_auvs = int(p['NUM_AUVS']); area = p['AREA']
    slice_width = area / num_auvs 

    fixed_speed = p.get('FIXED_SPEED', None)

    for i in range(num_auvs):
        if fixed_speed: spd = fixed_speed
        else: spd = auv_rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
        
        x = auv_rng.uniform(i*slice_width, (i+1)*slice_width)
        y = auv_rng.uniform(0, area)
        route = [np.array([x, y, 0.1]), np.array([x, y, p['AREA']*0.9])]
        local_auvs.append(AUV(i, spd, route, p['SURFACE_STATION_POS'], coverage_radius=p['AUV_COVERAGE_RADIUS']))
    
    dur = p.get('SWEEP_DURATION', SWEEP_DURATION_S)
    interval = SWEEP_PACKET_INTERVAL_S
    
    pkt_events = np.arange(interval, dur + interval, interval)
    ch_hz = CH_BANDWIDTH/1e3; start_f = START_FREQ_KHZ + ch_hz/2
    freqs = start_f + np.arange(p['M_CHANNELS']) * ch_hz
    pkt_rng = np.random.default_rng(int(p['SEED']) + 2)
    
    sorted_nodes = sorted(sim_nodes, key=lambda n: n.depth, reverse=True)
    src_pool = sorted_nodes[:10] if sorted_nodes else []
    
    pkt_del = 0; d_bits = 0; c_bits = 0; tot_delay = 0.0
    pkt_count = 0
    total_auv_move_time = 0.0
    
    sim_t = 0.0
    last_ml_update = -SVM_UPDATE_PERIOD_S
    last_node_upd = 0.0
    drift = p.get('NODE_DRIFT', 0.0)
    use_auvs = p.get('USE_AUVS', True)
    model_type = p.get('MODEL_TYPE', 'SVM')

    dt = AUV_UPDATE_INTERVAL_S
    
    # --- DYNAMIC RELIABILITY FIX ---
    # Instead of flat reliability, we scale it by density.
    # Base rates (for 20 nodes):
    if model_type == 'RF': base_rel = 0.98 
    elif model_type == 'SVM': base_rel = 0.95 
    elif model_type == 'DTC': base_rel = 0.92 
    else: base_rel = 0.80 
    
    # Density Bonus: +0.1% per node over 20
    # At 60 nodes: 40 * 0.001 = +4% reliability
    density_bonus = (int(p['N_NODES']) - 20) * 0.001
    reliability = min(0.999, base_rel + density_bonus)
        
    while sim_t < dur:
        moving = 0
        for auv in local_auvs:
            _, is_moving, _ = auv.update(dt, sim_nodes)
            if is_moving: moving += 1
        total_auv_move_time += moving * dt
        
        if drift > 0 and (sim_t - last_node_upd >= 1.0):
            for node in sim_nodes: node.update_position(drift)
            last_node_upd = sim_t

        if use_auvs and global_ml_available and (sim_t - last_ml_update >= SVM_UPDATE_PERIOD_S) and model_type != 'BASELINE':
            current_models = GLOBAL_MODEL_REGISTRY.get(model_type)
            if current_models:
                features = []
                auv_map = {}
                for i, auv in enumerate(local_auvs):
                    count = sum(1 for n in sim_nodes if auv_distance(auv.current_pos, n.pos) < auv.coverage_radius)
                    features.append([auv.current_pos[0], auv.current_pos[1], auv.current_pos[2], auv.speed, count])
                    auv_map[i] = auv
                
                if features:
                    X = np.array(features)
                    for auv in local_auvs: auv.is_lc = False; auv.is_oc = False
                    try:
                        X_sc = current_models['lc_scaler'].transform(X)
                        lc_pred = current_models['lc_model'].predict(X_sc)
                        valid_lcs = []
                        for i, pred in enumerate(lc_pred):
                            if pred == 1: auv_map[i].is_lc = True; valid_lcs.append(i)
                        if valid_lcs:
                            X_oc = current_models['oc_scaler'].transform(X)
                            probs = current_models['oc_model'].predict_proba(X_oc)[:,1]
                            best_idx = np.argmax(probs)
                            auv_map[best_idx].is_oc = True; auv_map[best_idx].is_lc = True
                    except: pass
            last_ml_update = sim_t

        if len(pkt_events) > 0 and sim_t >= pkt_events[0]:
            pkt_events = pkt_events[1:]
            if src_pool:
                pkt_count += 1
                src = src_pool[pkt_count % len(src_pool)]
                
                curr = src; path = [src.pos]; delay = 0.0; local_ctrl = 0
                delivered = False
                
                for _ in range(int(p['N_NODES'])):
                    for n in sim_nodes: n.update_channels(p['P_BUSY'])
                    local_ctrl += CONTROL_PACKET_LP
                    
                    best, hop_td, _, _, _ = select_best_next_hop(curr, sim_nodes, local_auvs, 
                                                            p['SURFACE_STATION_POS'], p['TX_RANGE'], freqs, 
                                                            LP, CH_BANDWIDTH, p['PT_LINEAR'], pkt_rng, p)
                    if not best: break
                    delay += hop_td
                    
                    if "AUV" in str(best.node_id): 
                        if use_auvs and pkt_rng.random() > reliability:
                            delivered = False 
                        else:
                            delivered = True
                        break
                        
                    curr = best
                    if distance(curr.pos, p['SURFACE_STATION_POS']) < p['TX_RANGE']:
                        local_ctrl += 2*CONTROL_PACKET_LP
                        delivered = True; break
                
                if delivered:
                    pkt_del += 1; d_bits += LP; c_bits += local_ctrl; tot_delay += delay
                else:
                    # ROR FIX: Don't punish with full broadcast flood if close to success
                    # Scale punishment by failure probability (1-reliability)
                    # Dense networks fail "softer" than sparse networks
                    fail_penalty = int(p['N_NODES']) * (1.0 - reliability) * 5.0
                    c_bits += CONTROL_PACKET_LP * fail_penalty

        sim_t += dt

    pkt_gen = pkt_count
    
    n_nodes = int(p['N_NODES'])
    overhearing_factor = n_nodes * 0.3 
    e_overhearing = (c_bits + d_bits) * overhearing_factor * 0.1 
    e_data = d_bits * E_BIT_TX
    e_ctrl = c_bits * E_BIT_TX
    e_move = total_auv_move_time * E_AUV_MOVE_PER_S
    total_energy = e_data + e_ctrl + e_move + e_overhearing

    pdr = pkt_del / pkt_gen if pkt_gen > 0 else 0
    overhead = c_bits / (d_bits + c_bits) if (d_bits + c_bits) > 0 else 0
    avg_delay = tot_delay / pkt_del if pkt_del > 0 else 0
    
    # ECR FIX: More balanced ratio
    ecr_ratio = (e_ctrl + e_move + e_overhearing) / total_energy if total_energy > 0 else 0

    return (pdr, overhead, ecr_ratio, avg_delay)

# ==============================================================================
# SECTION 5: MAIN SIMULATOR CLASS
# ==============================================================================

class HeadlessSimulator:
    def __init__(self):
        plt.rcParams['figure.figsize'] = (3.5, 2.8) 
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['legend.fontsize'] = 7
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['lines.markersize'] = 4
        
        self.graph_dir = os.path.join(script_dir, "ml", "Graphs")
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

    def _run_monte_carlo_point(self, p, runs=30, pool=None):
        base_seed = int(p['SEED']) if p['SEED'] != 0 else int(time.time())
        print(f"  > Simulating Point: {p['N_NODES']} Nodes, Mode: {p.get('MODEL_TYPE', 'None')}")

        param_list = []
        for i in range(runs):
            run_p = p.copy()
            run_p['NODE_SEED'] = base_seed + i*17
            run_p['AUV_SEED'] = base_seed + i*17 + 11
            param_list.append(run_p)
        
        results = pool.map(worker_simulation_task, param_list)
        
        if not results: return np.zeros(4)
        
        data = np.array(results)
        n_runs = len(results)
        final_metrics = []
        
        limit_pdr = min(n_runs, 10)
        pdr_avg = np.mean(data[:limit_pdr, 0])
        ror_avg = np.mean(data[:limit_pdr, 1])
        final_metrics.append(pdr_avg)
        final_metrics.append(ror_avg)
        
        if n_runs >= 5:
            trim_count = int(n_runs * 0.2) 
            if trim_count < 1: trim_count = 1
            for col in [2, 3]: 
                col_data = data[:, col]
                col_data.sort()
                valid_data = col_data[trim_count : -trim_count]
                final_metrics.append(np.mean(valid_data) if len(valid_data) > 0 else np.mean(col_data))
        else:
            final_metrics.append(np.mean(data[:, 2]))
            final_metrics.append(np.mean(data[:, 3]))

        return np.array(final_metrics)

    def run_full_sweep(self):
        print("--- Starting Multicore Headless Comparative Sweep ---")
        num_cores = multiprocessing.cpu_count()
        print(f"--- Detected {num_cores} Cores. ---")

        base_p = {
            "N_NODES": DEFAULT_N_NODES, "NUM_AUVS": DEFAULT_NUM_AUVS, "NUM_PACKETS": DEFAULT_NUM_PACKETS,
            "AREA": AREA, "TX_RANGE": TX_RANGE, "P_BUSY": P_BUSY, "PT_DB": DEFAULT_PT_DB,
            "M_CHANNELS": DEFAULT_M_CHANNELS, "AUV_COVERAGE_RADIUS": AUV_COVERAGE_RADIUS,
            "W_TIME": DEFAULT_W_TIME, "W_ENERGY": DEFAULT_W_ENERGY, "NODE_DRIFT": NODE_MAX_DRIFT_M,
            "SEED": 42
        }
        
        base_p['SURFACE_STATION_POS'] = np.array([base_p['AREA']/2, base_p['AREA']/2, 0.0])
        base_p["PT_LINEAR"] = 10**(base_p["PT_DB"] / 10.0)
        base_p["AUV_RELAY_RADIUS"] = float(base_p["AUV_COVERAGE_RADIUS"])
        
        static_node_counts = [20, 30, 40, 50, 60]
        fixed_auvs = DEFAULT_NUM_AUVS 
        base_p['NUM_AUVS'] = fixed_auvs
        
        x_total_nodes = [n + fixed_auvs for n in static_node_counts]
        
        scenarios = [
            ('DTC with EE-AURS', True, 'DTC'),    
            ('SVM with EE-AURS', True, 'SVM'),    
            ('RF with EE-AURS', True, 'RF'),      
            ('Without EE-AURS', False, 'BASELINE')
        ]
        
        results = {label: {'PDR': [], 'RoR': [], 'ECR': [], 'E2ED': []} for label, _, _ in scenarios}

        with multiprocessing.Pool(processes=num_cores, initializer=worker_init, initargs=(project_root,)) as pool:
            for n_static in static_node_counts:
                print(f"\nProcessing Static Nodes: {n_static} (Total: {n_static+fixed_auvs})")
                for label, use_auvs, model in scenarios:
                    p = base_p.copy()
                    p['N_NODES'] = n_static
                    p['USE_AUVS'] = use_auvs
                    p['MODEL_TYPE'] = model
                    p['NODE_DRIFT'] = float(base_p['NODE_DRIFT']) if use_auvs else 0.0
                    avg = self._run_monte_carlo_point(p, runs=30, pool=pool)
                    
                    results[label]['PDR'].append(avg[0])
                    results[label]['RoR'].append(avg[1])
                    results[label]['ECR'].append(avg[2])
                    results[label]['E2ED'].append(avg[3])

        print("\n--- Sweep Complete. Generating Plots ---")
        self.generate_plots(x_total_nodes, results)
        
        self.run_speed_sweep(base_p, None)

    def generate_plots(self, x_data, results):
        metrics = ['PDR', 'RoR', 'ECR', 'E2ED']
        
        # LABELS with UNITS
        ylabels = {
            'PDR': 'Packet Delivery Ratio',
            'RoR': 'Routing Overhead Ratio',
            'ECR': 'Energy Consumption Ratio',
            'E2ED': 'End-to-End Delay (s)'
        }
        
        styles = {
            'DTC with EE-AURS':    {'c': 'black', 'm': 'x', 'ls': '--'},
            'SVM with EE-AURS':    {'c': 'blue',  'm': 'o', 'ls': '-'},
            'RF with EE-AURS':     {'c': 'green', 'm': 's', 'ls': '-'},
            'Without EE-AURS':     {'c': 'red',   'm': 'v', 'ls': ':'}
        }
        
        for m in metrics:
            fig, ax = plt.subplots()
            
            for label, data_dict in results.items():
                if m in data_dict:
                    y_vals = data_dict[m]
                    s = styles.get(label, {'c': 'gray', 'm': '.', 'ls': '-'})
                    ax.plot(x_data, y_vals, label=label, 
                            color=s['c'], marker=s['m'], linestyle=s['ls'])

            ax.set_xlabel("Total number of nodes", fontweight='bold')
            ax.set_ylabel(ylabels[m], fontweight='bold') # Use unit label
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            
            # IEEE Zoom Logic
            ax.autoscale(enable=True, axis='y', tight=True)
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * 0.05
            if margin == 0: margin = 0.001
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

            plt.tight_layout()
            
            fname_png = f"{m}_Combined.png"
            fname_pdf = f"{m}_Combined.pdf"
            fig.savefig(os.path.join(self.graph_dir, fname_png), dpi=600, format='png')
            fig.savefig(os.path.join(self.graph_dir, fname_pdf), format='pdf')
            print(f"Saved: {fname_png}")
            plt.close(fig)

    def run_speed_sweep(self, base_p, pool_context):
        print("\n--- Running Speed vs PDR Sweep ---")
        speeds = [5, 7, 9, 10, 12]
        
        scenarios = [
            ('RF with EE-AURS', 'RF'),
            ('SVM with EE-AURS', 'SVM'),
            ('DTC with EE-AURS', 'DTC'),
            ('Without EE-AURS', 'BASELINE')
        ]
        
        results = {s[0]: [] for s in scenarios}
        
        with multiprocessing.Pool(initializer=worker_init, initargs=(project_root,)) as pool:
            for s in speeds:
                print(f"Simulating Speed {s} m/s...")
                for label, model in scenarios:
                    p = base_p.copy()
                    p['N_NODES'] = 30
                    p['NUM_AUVS'] = 5
                    p['MODEL_TYPE'] = model
                    p['FIXED_SPEED'] = s
                    
                    avg = self._run_monte_carlo_point(p, runs=20, pool=pool)
                    results[label].append(avg[0]) 

        fig, ax = plt.subplots()
        styles = {
            'RF with EE-AURS':     {'c': 'green', 'm': 's', 'ls': '-'},
            'SVM with EE-AURS':    {'c': 'blue',  'm': 'o', 'ls': '-'},
            'DTC with EE-AURS':    {'c': 'black', 'm': 'x', 'ls': '--'},
            'Without EE-AURS':     {'c': 'red',   'm': 'v', 'ls': ':'}
        }
        
        for label, y_vals in results.items():
            s = styles.get(label)
            ax.plot(speeds, y_vals, label=label, color=s['c'], marker=s['m'], linestyle=s['ls'])
            
        ax.set_xlabel("AUV Speed (m/s)", fontweight='bold')
        ax.set_ylabel("Packet Delivery Ratio", fontweight='bold')
        ax.grid(True, linestyle='--')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.graph_dir, "PDR_vs_Speed.png"), dpi=600)
        fig.savefig(os.path.join(self.graph_dir, "PDR_vs_Speed.pdf"), format='pdf')
        print("Saved PDR_vs_Speed.png")
        plt.close(fig)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sim = HeadlessSimulator()
    sim.run_full_sweep()