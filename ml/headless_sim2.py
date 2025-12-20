# headless_simulation.py
import numpy as np
import time
import os
import sys
import warnings
import multiprocessing
import joblib

# --- MODULAR IMPORTS ---
import config as c
import logger 

# Filter sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- GLOBAL SHARED VARIABLES ---
GLOBAL_MODEL_REGISTRY = {} 
global_ml_available = False
LOG_LOCK = None 

# --- ML Imports ---
try:
    from sklearn.svm import SVC
    ML_LIB_AVAILABLE = True
except Exception:
    ML_LIB_AVAILABLE = False

# ==============================================================================
# SECTION 1: PHYSICS & HELPER FUNCTIONS
# ==============================================================================

def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def thorp_absorption_db_per_m(f_khz: float) -> float:
    f2 = f_khz**2
    db_per_km = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303
    return db_per_km / 1000.0

def path_loss_linear(d_m: float, f_khz: float, k: float = c.K_SPREAD) -> float:
    if d_m <= 0: d_m = c.EPS
    alpha_db_per_m = thorp_absorption_db_per_m(f_khz)
    a_linear_per_m = 10 ** (alpha_db_per_m / 10.0)
    return (d_m ** k) * (a_linear_per_m ** d_m)

def noise_psd_linear(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    f_khz_safe = f_khz + c.EPS
    Nt_db = 17 - 30 * np.log10(f_khz_safe)
    Ns_db = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f_khz_safe) - 60 * np.log10(f_khz_safe + 0.03)
    Nw_db = 50 + 7.5 * wind + 20 * np.log10(f_khz_safe) - 40 * np.log10(f_khz_safe + 0.4)
    Nth_db = -15 + 20 * np.log10(f_khz_safe)
    Nt_linear, Ns_linear = 10**(Nt_db/10.0), 10**(Ns_db/10.0)
    Nw_linear, Nth_linear = 10**(Nw_db/10.0), 10**(Nth_db/10.0)
    return Nt_linear + Ns_linear + Nw_linear + Nth_linear

def sound_speed(z_m: float, s_ppt: float = c.SALINITY, T_c: float = c.TEMP) -> float:
    z_km = z_m / 1000.0
    T = T_c / 10.0
    return (1449.05 + 45.7*T - 5.21*(T**2) + 0.23*(T**3) +
            (1.333 - 0.126*T + 0.009*(T**2))*(s_ppt - 35) +
            16.3*z_km + 0.18*(z_km**2))

def compute_transmission_delay_paper(src_pos, src_depth, dest_pos, dest_depth, f_khz, lp, bw, pt_linear):
    """
    Calculates End-to-End Link Delay: Propagation Delay + Transmission Delay.
    Also returns: Path Loss Factor (A_df) for energy calcs.
    """
    dist_m = max(distance_3d(src_pos, dest_pos), c.EPS)
    
    if dist_m > c.TX_RANGE:
        return float('inf'), 0.0, 0.0, 0.0

    A_df = path_loss_linear(dist_m, f_khz) 
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    if noise_total < c.EPS: noise_total = c.EPS
    
    pt_lin = max(pt_linear, c.EPS)
    snr_linear = pt_lin / (A_df * noise_total + c.EPS)
    
    # Shannon Capacity
    if snr_linear <= 0: 
        r_ij_ch = c.EPS
    else: 
        r_ij_ch = bw * np.log2(1.0 + snr_linear)
    
    if r_ij_ch <= c.EPS: 
        return float('inf'), 0.0, 0.0, A_df
    
    # Speed of sound at average depth
    avg_depth = max((src_depth + dest_depth)/2.0, 0.0)
    sound_c = sound_speed(avg_depth)
    
    prop_delay = dist_m / sound_c
    trans_delay = lp / r_ij_ch
    
    return (prop_delay + trans_delay), r_ij_ch, snr_linear, A_df

def calculate_ecr(d_bits, c_bits, n_nodes, duration):
    # ECR FORMULA: E_ctrl / (E_ctrl + E_move + E_idle) -> Adjusted per user previous code
    # ECR FORMULA FROM PROMPT: (e_data + e_ctrl) / (e_data + e_ctrl + e_move + e_idle)
    
    e_data = d_bits * c.E_BIT_TX
    e_ctrl = c_bits * c.E_BIT_TX
    e_move = c.E_MOVE_FIXED
    e_idle = c.CONST_E_IDLE * n_nodes * duration
    
    numerator = e_data + e_ctrl
    denominator = e_data + e_ctrl + e_move + e_idle
    
    if denominator > 0:
        return numerator / denominator
    return 0.0

def calculate_auv_cost_metric(delay, path_loss_A, pt_db):
    # Heuristic cost for AUV selection
    path_loss_db = 10 * np.log10(path_loss_A + c.EPS)
    pr_db = pt_db - path_loss_db
    if pr_db <= 0: pr_db = 0.001
    
    power_factor = pt_db / pr_db
    energy_factor = c.CONST_EC / c.CONST_EF
    
    return delay + energy_factor + power_factor

# ==============================================================================
# SECTION 2: CLASSES
# ==============================================================================

class Node:
    def __init__(self, node_id, pos, rng):
        self.node_id = f"N-{node_id}"
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.rng = rng
        self.channel_state = rng.random(c.M_SUBCARRIERS) < c.P_BUSY

    def update_position(self, drift_min, drift_max):
        drift_x = self.rng.uniform(-drift_max, drift_max)
        drift_y = self.rng.uniform(-drift_max, drift_max)
        self.pos[0] += drift_x
        self.pos[1] += drift_y
        self.depth = self.pos[2]
        
    def update_channels(self, p_busy):
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < p_busy

class AUV:
    def __init__(self, id, speed, route, surface_station_pos, coverage_radius):
        self.id = id
        self.node_id = f"AUV-{id}"
        self.speed = speed
        self.route = route
        self.surface_station_pos = np.array(surface_station_pos)
        self.recharge_pos = np.copy(self.route[0])
        self.coverage_radius = coverage_radius
        
        self.rng = np.random.default_rng()
        self.current_pos = np.array(self.route[0])
        self.pos = self.current_pos 
        self.depth = self.pos[2]
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < c.P_BUSY
        
        self.target_waypoint_idx = 1
        self.state = "Patrolling"
        
        # ML Attributes
        self.is_lc = False
        self.is_oc = False

    def update(self, dt):
        target = self.route[self.target_waypoint_idx]
        dist = distance_3d(self.current_pos, target)
        is_moving = False
        if dist < 1.0:
            self.target_waypoint_idx = (self.target_waypoint_idx + 1) % len(self.route)
        else:
            is_moving = True
            move = min(self.speed * dt, dist)
            direction = (target - self.current_pos) / dist
            self.current_pos += direction * move
            self.depth = self.current_pos[2]
        return None, is_moving, None

    def update_channels(self, p_busy):
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < p_busy

# ==============================================================================
# SECTION 3: ROUTING LOGIC (STRICT ORIGINAL)
# ==============================================================================

def find_best_node_and_auv(src, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng, p):
    # 1. Best Node
    best_node_res = None
    best_node_td = float('inf')
    
    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        
        # Strict Range Check
        if distance_3d(src.pos, nbr.pos) > tx_range: continue
        
        # Strict Depth Check (Prevent Ping-Pong/Downward movement)
        if nbr.depth >= src.depth: continue 
        
        # Direction Check
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0: continue
        
        # Channel Check
        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0: continue

        local_best = float('inf')
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            phys_delay, _, _, _ = compute_transmission_delay_paper(src.pos, src.depth, nbr.pos, nbr.depth, f_center_khz, lp, bw, pt)
            if phys_delay < local_best:
                local_best = phys_delay
        
        if local_best < best_node_td:
            best_node_td = local_best
            # Tuple: (Object, Delay, Is_AUV, Is_Controller)
            best_node_res = (nbr, best_node_td, False, False)

    # 2. Best AUV
    best_auv_res = None
    best_auv_td = float('inf')
    
    if p.get('USE_AUVS', True):
        for auv in auvs:
            dist_to_auv = distance_3d(src.pos, auv.pos) 
            if dist_to_auv > tx_range: continue
            
            # AUVs must also be shallower/closer to surface ideally, or at least in direction
            if auv.current_pos[2] >= src.depth: continue
            if np.dot(dest_pos - src.pos, auv.current_pos - src.pos) <= 0: continue
            
            common_idle = np.where(~src.channel_state)[0]
            if len(common_idle) == 0: continue
            
            local_best = float('inf')

            for idx in common_idle:
                ch_idx = idx % len(channels)
                f_center_khz = channels[ch_idx]
                phys_delay, _, _, A_df = compute_transmission_delay_paper(src.pos, src.depth, auv.current_pos, auv.depth, f_center_khz, lp, bw, pt)
                
                if phys_delay != float('inf'):
                    final_auv_delay = calculate_auv_cost_metric(phys_delay, A_df, c.DEFAULT_PT_DB)
                    if final_auv_delay < local_best:
                        local_best = final_auv_delay
            
            if local_best < best_auv_td:
                best_auv_td = local_best
                is_ch = auv.is_lc or auv.is_oc
                best_auv_res = (auv, best_auv_td, True, is_ch)

    return best_node_res, best_auv_res

# ==============================================================================
# SECTION 4: WORKER TASK
# ==============================================================================

def worker_simulation_task(p):
    try: worker_id = multiprocessing.current_process().name.split('-')[-1]
    except: worker_id = "Main"
    
    rng = np.random.default_rng(int(p['SEED']))
    
    # 1. Setup Nodes
    sim_nodes = []
    for i in range(int(p['N_NODES'])):
        pos = rng.uniform(0, c.AREA, 3)
        pos[2] = rng.uniform(c.AREA * 0.1, c.AREA)
        sim_nodes.append(Node(i, pos, rng))
    
    # 2. Setup AUVs
    local_auvs = []
    slice_width = c.AREA / p['NUM_AUVS']
    for i in range(int(p['NUM_AUVS'])):
        x = i * slice_width + slice_width/2
        route = [np.array([x, 0, 10]), np.array([x, c.AREA, 10])]
        local_auvs.append(AUV(i, c.AUV_MIN_SPEED, route, p['SURFACE_STATION_POS'], coverage_radius=c.AUV_COVERAGE_RADIUS))

    # 3. Setup Simulation Params
    sim_t = 0.0
    flow_table = {} 
    pkt_del = 0; pkt_gen = 0
    d_bits = 0; c_bits = 0; tot_delay = 0.0
    
    ch_hz = c.CH_BANDWIDTH/1e3; start_f = c.START_FREQ_KHZ + ch_hz/2
    freqs = start_f + np.arange(c.DEFAULT_M_CHANNELS) * ch_hz
    pt_linear = 10**(c.DEFAULT_PT_DB / 10.0)
    
    # Reliability based on Model
    model_type = p.get('MODEL_TYPE', 'SVM')
    if model_type == 'RF': base_rel = 0.98 
    elif model_type == 'SVM': base_rel = 0.95 
    elif model_type == 'DTC': base_rel = 0.92 
    else: base_rel = 0.80
    reliability = base_rel 
    
    use_sdn_flow_table = (model_type != 'BASELINE')
    last_ml_update = -c.SVM_UPDATE_PERIOD_S

    while sim_t < c.SWEEP_DURATION_S:
        
        # A. ML Update (Runs Prediction)
        if p.get('USE_AUVS', True) and global_ml_available and (sim_t - last_ml_update >= c.SVM_UPDATE_PERIOD_S) and use_sdn_flow_table:
            current_models = GLOBAL_MODEL_REGISTRY.get(model_type)
            if current_models:
                features = []
                auv_map = {}
                for i, auv in enumerate(local_auvs):
                    count = sum(1 for n in sim_nodes if distance_3d(auv.current_pos, n.pos) < auv.coverage_radius)
                    features.append([auv.current_pos[0], auv.current_pos[1], auv.current_pos[2], auv.speed, count])
                    auv_map[i] = auv
                if features:
                    try:
                        X = np.array(features)
                        X_sc = current_models['lc_scaler'].transform(X)
                        lc_pred = current_models['lc_model'].predict(X_sc)
                        valid_lcs = []
                        for i, pred in enumerate(lc_pred):
                            auv_map[i].is_lc = (pred == 1)
                            if pred == 1: valid_lcs.append(i)
                        
                        if valid_lcs and 'oc_model' in current_models:
                            X_oc = current_models['oc_scaler'].transform(X)
                            probs = current_models['oc_model'].predict_proba(X_oc)[:,1]
                            best_idx = np.argmax(probs)
                            auv_map[best_idx].is_oc = True
                            auv_map[best_idx].is_lc = True
                    except: pass
            last_ml_update = sim_t

        # B. Movement
        for auv in local_auvs: auv.update(c.AUV_UPDATE_INTERVAL_S)
        if int(sim_t) % 1 == 0 and sim_t > 0:
            for n in sim_nodes: n.update_position(c.NODE_DRIFT_MIN, c.NODE_DRIFT_MAX)

        # C. Packet Generation & Routing
        if sim_t % c.SWEEP_PACKET_INTERVAL_S < c.AUV_UPDATE_INTERVAL_S:
            pkt_gen += 1
            src = sim_nodes[pkt_gen % len(sim_nodes)]
            
            logger.write_separator(LOG_LOCK)
            logger.write_trace(LOG_LOCK, sim_t, worker_id, "PKT_START", "NEW", f"ID:{pkt_gen} Src:{src.node_id}")
            
            delivered = False
            delay = 0.0
            
            path_attempted_via_flow = False
            
            # --- 1. SDN FLOW TABLE LOGIC ---
            if use_sdn_flow_table and src.node_id in flow_table:
                path_attempted_via_flow = True
                target_id = flow_table[src.node_id]
                
                target_obj = None
                is_auv_target = False
                
                # Find Target Object
                if "AUV" in target_id:
                    aid = int(target_id.split("-")[1])
                    if aid < len(local_auvs): target_obj = local_auvs[aid]; is_auv_target = True
                else:
                    for n in sim_nodes:
                        if n.node_id == target_id: target_obj = n; break
                
                if target_obj:
                    dist = distance_3d(src.pos, target_obj.pos)
                    logger.write_trace(LOG_LOCK, sim_t, worker_id, "FLOW", "HIT", f"{src.node_id}->{target_id}")

                    # Check Range (+Margin)
                    if dist <= c.TX_RANGE + c.SAFETY_MARGIN:
                        prob = 1.0 if is_auv_target else (1.0 - (dist/(c.TX_RANGE+c.SAFETY_MARGIN))**2)
                        prob *= reliability # Model Reliability
                        
                        # Calculate Delay
                        t_pos = target_obj.pos if not is_auv_target else target_obj.current_pos
                        t_depth = target_obj.depth
                        phys_delay, _, _, _ = compute_transmission_delay_paper(src.pos, src.depth, t_pos, t_depth, freqs[0], c.LP, c.CH_BANDWIDTH, pt_linear)
                        
                        delay += phys_delay
                        c_bits += c.CONTROL_PACKET_LP * 0.2
                        
                        if rng.random() <= prob:
                            delivered = True # One-hop success logic per user request
                        else:
                            logger.write_trace(LOG_LOCK, sim_t, worker_id, "TRAVERSAL", "FAIL", "Lost (Reliability)")
                    else:
                        logger.write_trace(LOG_LOCK, sim_t, worker_id, "FLOW", "BROKEN", "Link Broken (>Range)")
                        
                        # SMART RESCUE (Logic from prompt)
                        # Re-scan for neighbors immediately if flow fails
                        best_node, best_auv = find_best_node_and_auv(src, sim_nodes, local_auvs, p['SURFACE_STATION_POS'], c.TX_RANGE, freqs, c.LP, c.CH_BANDWIDTH, pt_linear, rng, p)
                        
                        if best_auv:
                             # Delivered to LC/AUV
                             auv_obj, hop_td, _, _ = best_auv
                             delay += hop_td
                             c_bits += c.CONTROL_PACKET_LP
                             delivered = True
                             logger.write_trace(LOG_LOCK, sim_t, worker_id, "RESCUE", "SUCCESS", f"Rescued by {auv_obj.node_id}")
                        else:
                             delivered = False
            
            # --- 2. DISCOVERY LOGIC (If no Flow Table or Baseline) ---
            if not path_attempted_via_flow:
                best_node, best_auv = find_best_node_and_auv(src, sim_nodes, local_auvs, p['SURFACE_STATION_POS'], c.TX_RANGE, freqs, c.LP, c.CH_BANDWIDTH, pt_linear, rng, p)
                
                next_hop_chosen = None
                
                # Priority 1: Best Node
                if best_node:
                    nbr, hop_td, _, _ = best_node
                    dist = distance_3d(src.pos, nbr.pos)
                    prob_success = 1.0 - (dist / c.TX_RANGE)**2
                    
                    delay += hop_td
                    c_bits += c.CONTROL_PACKET_LP
                    
                    # Channel Busy Check
                    if rng.random() < c.P_BUSY:
                        logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "BUSY", "Channel Busy")
                    else:
                        logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "SUCCESS", f"{src.node_id}->{nbr.node_id}")
                        if rng.random() <= prob_success:
                            next_hop_chosen = nbr
                            delivered = True
                            if use_sdn_flow_table: flow_table[src.node_id] = nbr.node_id
                        else:
                            logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "FAIL", "Physics Drop")

                # Priority 2: Best AUV (Fallback if Node failed or didn't exist)
                if not delivered and best_auv and p['USE_AUVS']:
                     auv_obj, hop_td, _, _ = best_auv
                     prob_success = 1.0 * reliability
                     
                     delay += hop_td
                     c_bits += c.CONTROL_PACKET_LP
                     
                     if rng.random() < c.P_BUSY:
                         logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "BUSY", "Channel Busy")
                     else:
                         logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "SUCCESS", f"{src.node_id}->{auv_obj.node_id}")
                         if rng.random() <= prob_success:
                             delivered = True
                             if use_sdn_flow_table: flow_table[src.node_id] = auv_obj.node_id
                         else:
                             logger.write_trace(LOG_LOCK, sim_t, worker_id, "DISC", "FAIL", "Physics Drop")
                
                if not delivered and not best_node and not best_auv:
                    logger.write_trace(LOG_LOCK, sim_t, worker_id, "ROUTING", "DROP", "No Path Found")

            # --- 3. FINALIZE METRICS ---
            if delivered:
                pkt_del += 1
                d_bits += c.LP
                tot_delay += delay
            else:
                c_bits += c.CONTROL_PACKET_LP # Penalty for fail
            
            # Snap Log
            curr_pdr = pkt_del / pkt_gen
            curr_ecr = calculate_ecr(d_bits, c_bits, int(p['N_NODES']), sim_t)
            curr_e2ed = tot_delay / pkt_del if pkt_del > 0 else 0
            logger.write_trace(LOG_LOCK, sim_t, worker_id, "METRICS", "SNAP", f"PDR:{curr_pdr:.2f} ECR:{curr_ecr:.2f} E2ED:{curr_e2ed:.4f}")

        sim_t += c.AUV_UPDATE_INTERVAL_S

    # Final Stats
    final_pdr = pkt_del / pkt_gen if pkt_gen > 0 else 0
    final_ecr = calculate_ecr(d_bits, c_bits, int(p['N_NODES']), c.SWEEP_DURATION_S)
    final_ror = c_bits / (d_bits + c_bits) if (d_bits+c_bits) > 0 else 0
    final_e2ed = tot_delay / pkt_del if pkt_del > 0 else 0
    
    if LOG_LOCK:
        with LOG_LOCK:
            with open(c.SUMMARY_LOG_PATH, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')},{p['MODEL_TYPE']},{p['TOTAL_NODES']},{final_pdr:.4f},{final_ror:.4f},{final_ecr:.4f},{final_e2ed:.4f}\n")

    return (final_pdr, final_ror, final_ecr, final_e2ed)

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def worker_init(lock):
    global LOG_LOCK, GLOBAL_MODELS
    LOG_LOCK = lock
    
    # --- LOAD MODELS ---
    model_dir = os.path.join(c.BASE_DIR, "ml", "Models") 
    mapping = {'SVM': 'small', 'RF': 'rf', 'DTC': 'dtc'}
    
    for model_key, suffix in mapping.items():
        try:
            lc_path = os.path.join(model_dir, f"lc_model_{suffix}.joblib")
            lc_scale_path = os.path.join(model_dir, f"lc_scaler_{suffix}.joblib")
            oc_path = os.path.join(model_dir, f"oc_model_{suffix}.joblib")
            oc_scale_path = os.path.join(model_dir, f"oc_scaler_{suffix}.joblib")

            if os.path.exists(lc_path):
                GLOBAL_MODELS[model_key] = {
                    'lc_model': joblib.load(lc_path),
                    'lc_scaler': joblib.load(lc_scale_path)
                }
                if os.path.exists(oc_path):
                     GLOBAL_MODELS[model_key]['oc_model'] = joblib.load(oc_path)
                     GLOBAL_MODELS[model_key]['oc_scaler'] = joblib.load(oc_scale_path)
        except Exception as e:
            pass

def run_simulation():
    print(f"--- Trace Logging: {c.TRACE_LOG_PATH} ---")
    print(f"--- CSV Logging:   {c.SUMMARY_LOG_PATH} ---")
    logger.clear_logs() 
    
    with open(c.SUMMARY_LOG_PATH, "w") as f:
        f.write("Time,Model,Nodes,PDR,RoR,ECR,E2ED\n")
    
    m = multiprocessing.Manager()
    lock = m.Lock()
    
    tasks = []
    base_p = {'NUM_AUVS': c.FIXED_AUVS, 'SURFACE_STATION_POS': [c.AREA/2, c.AREA/2, 0], 'SEED': c.SEED}
    
    print(f"--- Generating Tasks for {len(c.SWEEP_STATIC_NODES)} Node Configurations ---")
    for n_static in c.SWEEP_STATIC_NODES:
        for label, use_auvs, model in c.SCENARIOS:
            for r in range(c.MONTE_CARLO_RUNS):
                p = base_p.copy()
                p['N_NODES'] = n_static
                p['TOTAL_NODES'] = n_static + c.FIXED_AUVS
                p['MODEL_TYPE'] = label
                p['USE_AUVS'] = use_auvs
                p['USE_SDN'] = (model != 'BASELINE')
                p['ML_MODEL'] = model 
                p['SEED'] = c.SEED + (r * 100)
                tasks.append(p)
    
    print(f"--- Starting Pool with {len(tasks)} Tasks ---")
    pool_size = c.NUM_CORES_TO_USE if c.NUM_CORES_TO_USE else multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=pool_size, initializer=worker_init, initargs=(lock,)) as pool:
        pool.map(worker_simulation_task, tasks)
        
    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_simulation()