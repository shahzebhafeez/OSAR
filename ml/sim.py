# sim.py
# --- Headless OSAR Simulation (Multicore + IEEE Standards) ---
#
# TRIAL 12 CONFIGURATION:
# 1. BASELINE: Strict Upward Rule ONLY (No Panic Mode).
# 2. RETRIES: Reduced to 3 (Channel) and 1 (Physics).
# 3. HOP LIMIT: Reduced to 20.
# 4. SDN CONTENTION: Set to 0.3.
# 5. PHYSICS: Hard (Power 4).

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
import joblib
import heapq

# --- MODULAR IMPORTS ---
import config as c
import physics_equations as pe
import logger

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

GLOBAL_MODEL_REGISTRY = {}
global_ml_available = False
LOG_FILE_PATH = c.SUMMARY_LOG_PATH
LOG_LOCK = None

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    ML_LIB_AVAILABLE = True
except Exception as e:
    print("Warning: ML disabled.")
    ML_LIB_AVAILABLE = False
    joblib = None

# ==============================================================================
# SECTION 1: CLASSES
# ==============================================================================


class Node:
    def __init__(self, node_id, pos, rng, channel_state=None):
        self.node_id = f"N-{node_id}"
        self.initial_pos = np.copy(pos)
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.rng = rng
        self.channel_state = channel_state if channel_state is not None else \
            rng.random(c.M_SUBCARRIERS) < c.P_BUSY

    def update_position(self, drift_radius):
        pass

    def update_channels(self, p_busy):
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < p_busy


class AUV:
    def __init__(self, id, speed, route, surface_station_pos,
                 coverage_radius=c.AUV_COVERAGE_RADIUS,
                 relay_radius=c.AUV_RELAY_RADIUS):
        self.id = id
        self.node_id = f"AUV-{id}"
        self.speed = speed
        self.route = route
        self.surface_station_pos = np.array(surface_station_pos)
        self.recharge_pos = np.copy(self.route[0])
        self.coverage_radius = coverage_radius
        self.relay_radius = relay_radius

        self.rng = np.random.default_rng()
        self.current_pos = np.array(self.route[0])
        self.pos = self.current_pos
        self.initial_pos = np.copy(self.route[0])
        self.depth = self.pos[2]
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < c.P_BUSY

        self.target_waypoint_idx = 1
        self.covered_nodes = set()
        self.traveled_path = [np.array(self.route[0])]
        self.data_buffer = {}
        self.relayed_data_log = {}
        self.battery = 100.0
        self.state = "Patrolling"
        self.last_role_change = 0.0

        # ML Roles
        self.is_lc = False
        self.is_oc = False
        self.reliability_score = 0.80

    def collect_data(self, node_id):
        if node_id not in self.data_buffer:
            self.data_buffer[node_id] = time.time()
            return True
        return False

    def relay_data(self):
        if not self.data_buffer:
            return None
        relayed_node_ids = list(self.data_buffer.keys())
        for node_id in relayed_node_ids:
            self.relayed_data_log[node_id] = time.time()
        self.data_buffer.clear()
        return relayed_node_ids

    def update_channels(self, p_busy):
        self.channel_state = self.rng.random(c.M_SUBCARRIERS) < p_busy

    def update(self, dt, nodes):
        relayed_node_ids_this_tick = None
        is_moving = True
        self.battery -= 1.0 * (self.speed / c.AUV_MAX_SPEED) * dt
        self.battery = max(0, self.battery)

        if self.battery < 20.0 and self.state == "Patrolling":
            self.state = "Returning to Charge"

        if self.state == "Returning to Charge" and pe.distance_3d(self.current_pos, self.recharge_pos) < 5.0:
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

        if target_pos is None:
            is_moving = False

        if is_moving:
            direction = target_pos - self.current_pos
            dist_to_target = np.linalg.norm(direction)

            if dist_to_target < 1.0:
                if self.state == "Patrolling":
                    self.target_waypoint_idx += 1
            else:
                move_dist = self.speed * dt
                move_dist = min(move_dist, dist_to_target)
                move_vec = (direction / dist_to_target) * move_dist
                self.current_pos += move_vec
                self.depth = self.current_pos[2]

        self.traveled_path.append(np.copy(self.current_pos))
        if len(self.traveled_path) > 500:
            self.traveled_path.pop(0)

        collected_new_data_flag = False
        for node in nodes:
            if pe.distance_3d(self.current_pos, node.pos) <= self.coverage_radius:
                if node.node_id not in self.covered_nodes:
                    self.covered_nodes.add(node.node_id)
                if self.collect_data(node.node_id):
                    collected_new_data_flag = True

        if pe.distance_3d(self.current_pos, self.surface_station_pos) <= self.relay_radius:
            relayed_node_ids_this_tick = self.relay_data()

        return relayed_node_ids_this_tick, is_moving, collected_new_data_flag

# ==============================================================================
# SECTION 2: SDN PROACTIVE FUNCTIONS
# ==============================================================================


def build_topology_graph(nodes, auvs, surface_pos, tx_range):
    graph = {}
    all_vertices = ["SURFACE"]

    entity_map = {"SURFACE": {'pos': surface_pos, 'depth': 0, 'is_lc': False}}
    for n in nodes:
        all_vertices.append(n.node_id)
        entity_map[n.node_id] = {'pos': n.pos,
                                 'depth': n.depth, 'is_lc': False}
    for a in auvs:
        all_vertices.append(a.node_id)
        entity_map[a.node_id] = {'pos': a.current_pos,
                                 'depth': a.depth, 'is_lc': a.is_lc}

    for v in all_vertices:
        graph[v] = []

    for i in range(len(all_vertices)):
        for j in range(len(all_vertices)):
            if i == j:
                continue

            id1, id2 = all_vertices[i], all_vertices[j]
            e1, e2 = entity_map[id1], entity_map[id2]

            # STRICT UPWARD RULE for SDN Graph
            if e2['depth'] >= e1['depth'] and id2 != "SURFACE":
                continue

            dist = pe.distance_3d(e1['pos'], e2['pos'])

            if dist <= tx_range:
                prob = 1.0 - (dist / tx_range)**4  # Hard Physics
                if prob <= 0.01:
                    prob = 0.01
                cost = 1.0 / prob  # Reliability Cost

                if e2['is_lc']:
                    cost *= 0.5
                if e1['is_lc']:
                    cost *= 0.5

                graph[id1].append((id2, cost))

    return graph


def dijkstra_paths(graph, source_id, dest_id="SURFACE"):
    if source_id not in graph:
        return None
    distances = {node: float('inf') for node in graph}
    distances[source_id] = 0
    previous = {node: None for node in graph}
    pq = [(0, source_id)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > distances[current_node]:
            continue
        if current_node == dest_id:
            break
        if current_node not in graph:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    if distances.get(dest_id, float('inf')) == float('inf'):
        return None
    path = []
    current = dest_id
    while current is not None:
        path.append(current)
        current = previous[current]
    return path[::-1]


def compute_all_paths_proactively(nodes, auvs, surface_pos, tx_range, mc_flow_table):
    graph = build_topology_graph(nodes, auvs, surface_pos, tx_range)
    for node in nodes:
        path = dijkstra_paths(graph, node.node_id)
        if path and len(path) > 1:
            mc_flow_table[node.node_id] = path[1:]
    return mc_flow_table


def invalidate_stale_paths(mc_flow_table, auvs, current_time, invalidate_threshold=2.0):
    to_remove = []
    for src_id, path in mc_flow_table.items():
        for hop in path:
            if "AUV" in hop:
                auv_id = int(hop.split("-")[1])
                if auv_id < len(auvs):
                    auv = auvs[auv_id]
                    last_change = getattr(auv, 'last_role_change', 0.0)
                    if current_time - last_change < invalidate_threshold:
                        to_remove.append(src_id)
                        break
    for src_id in to_remove:
        del mc_flow_table[src_id]
    return mc_flow_table

# ==============================================================================
# SECTION 3: HELPERS (GREEDY - STRICT UPWARD)
# ==============================================================================


def place_nodes_randomly(N, area, rng):
    nodes = []
    n_shallow = int(N * 0.1)
    n_deep = N - n_shallow

    for i in range(n_shallow):
        pos = rng.uniform(0, area, 3)
        pos[2] = rng.uniform(20, 150)
        nodes.append(Node(i, pos, rng))

    for i in range(n_deep):
        pos = rng.uniform(0, area, 3)
        pos[2] = rng.uniform(150, area)
        nodes.append(Node(n_shallow + i, pos, rng))

    return nodes


def find_greedy_next_hop(curr, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng, p, visited):
    """
    Returns sorted list of candidates. STRICT UPWARD ONLY.
    """
    candidates = []
    potential_neighbors = list(
        nodes) + list(auvs) if p.get('USE_AUVS', True) else list(nodes)

    for nbr in potential_neighbors:
        if nbr.node_id == curr.node_id:
            continue
        if nbr.node_id in visited:
            continue

        # [FIX] STRICT UPWARD RULE (No Panic Mode available)
        if nbr.depth >= curr.depth:
            continue

        dist = pe.distance_3d(curr.pos, nbr.pos if not isinstance(
            nbr, AUV) else nbr.current_pos)
        if dist > tx_range:
            continue

        common_idle = np.where(~curr.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0:
            continue

        # --- SCORING ---
        progress = curr.depth - nbr.depth
        prob_success = 1.0 - (dist / tx_range)**4  # Hard Physics
        if prob_success < 0:
            prob_success = 0.0

        score = (prob_success ** 3) * (progress + 200)

        is_auv = isinstance(nbr, AUV)
        if is_auv:
            score *= 5.0
            if nbr.is_lc:
                score *= 2.0

        f_hz = channels[0]
        phys_delay, _, _, _ = pe.compute_transmission_delay_paper(
            curr.pos, curr.depth,
            nbr.pos if not is_auv else nbr.current_pos, nbr.depth,
            f_hz, lp, bw, pt
        )
        candidates.append({'obj': nbr, 'delay': phys_delay,
                          'is_auv': is_auv, 'score': score})

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

# ==============================================================================
# SECTION 4: WORKER SIMULATION
# ==============================================================================


def worker_init(model_dir, lock):
    global GLOBAL_MODEL_REGISTRY, global_ml_available, LOG_LOCK
    LOG_LOCK = lock
    if not ML_LIB_AVAILABLE:
        global_ml_available = False
        return
    GLOBAL_MODEL_REGISTRY = {}
    model_configs = {'SVM': 'small', 'RF': 'rf', 'DTC': 'dtc'}
    search_paths = [os.path.join(model_dir, "Models"),
                    model_dir, os.path.join(model_dir, "ml")]
    try:
        loaded_any = False
        for m_type, suffix in model_configs.items():
            lc_model_name = f"lc_model_{suffix}.joblib"
            lc_scaler_name = f"lc_scaler_{suffix}.joblib"
            for p in search_paths:
                lc_m_path = os.path.join(p, lc_model_name)
                lc_s_path = os.path.join(p, lc_scaler_name)
                if os.path.exists(lc_m_path):
                    GLOBAL_MODEL_REGISTRY[m_type] = {
                        'lc_model': joblib.load(lc_m_path),
                        'lc_scaler': joblib.load(lc_s_path)
                    }
                    loaded_any = True
                    break
        global_ml_available = loaded_any
    except Exception as e:
        print(f"Error loading models: {e}")
        global_ml_available = False


def worker_simulation_task(p):
    try:
        worker_id = multiprocessing.current_process().name.split('-')[-1]
    except:
        worker_id = "Main"

    rng_nodes = np.random.default_rng(int(p.get('NODE_SEED', p['SEED'])))
    sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng_nodes)

    auv_rng = np.random.default_rng(int(p.get('AUV_SEED', int(p['SEED']) + 1)))
    local_auvs = []
    num_auvs = int(p['NUM_AUVS'])
    area = p['AREA']
    slice_width = area / num_auvs

    fixed_speed = p.get('FIXED_SPEED', None)
    for i in range(num_auvs):
        spd = fixed_speed if fixed_speed else auv_rng.uniform(
            c.AUV_MIN_SPEED, c.AUV_MAX_SPEED)
        x = auv_rng.uniform(i*slice_width, (i+1)*slice_width)
        y = auv_rng.uniform(0, area)
        route = [np.array([x, y, 0.1]), np.array([x, y, p['AREA']*0.9])]
        local_auvs.append(AUV(
            i, spd, route, p['SURFACE_STATION_POS'], coverage_radius=c.AUV_COVERAGE_RADIUS))

    dur = p.get('SWEEP_DURATION', c.SWEEP_DURATION_S)
    interval = c.SWEEP_PACKET_INTERVAL_S
    pkt_events = np.arange(interval, dur + interval, interval)
    ch_hz = c.CH_BANDWIDTH/1e3
    start_f = c.START_FREQ_KHZ + ch_hz/2
    freqs = start_f + np.arange(p['M_CHANNELS']) * ch_hz
    pkt_rng = np.random.default_rng(int(p['SEED']) + 2)

    sorted_nodes = sorted(sim_nodes, key=lambda n: n.depth, reverse=True)
    src_pool = sorted_nodes[:10] if sorted_nodes else []
    mc_flow_table = {}

    pkt_del = 0
    d_bits = 0
    c_bits = 0
    tot_delay = 0.0
    pkt_count = 0
    sim_t = 0.0
    last_ml_update = -c.SVM_UPDATE_PERIOD_S
    use_auvs = p.get('USE_AUVS', True)
    model_type = p.get('MODEL_TYPE', 'SVM')
    dt = c.AUV_UPDATE_INTERVAL_S
    use_sdn = (model_type != 'BASELINE')
    pt_linear = 10**(c.DEFAULT_PT_DB / 10.0)

    while sim_t < dur:
        if use_auvs and global_ml_available and use_sdn:
            if sim_t - last_ml_update >= c.SVM_UPDATE_PERIOD_S:
                current_models = GLOBAL_MODEL_REGISTRY.get(model_type)
                if current_models:
                    features = []
                    auv_map = {}
                    for i, auv in enumerate(local_auvs):
                        count = sum(1 for n in sim_nodes if pe.distance_3d(
                            auv.current_pos, n.pos) < auv.coverage_radius)
                        features.append(
                            [auv.current_pos[0], auv.current_pos[1], auv.current_pos[2], auv.speed, count])
                        auv_map[i] = auv
                    if features:
                        try:
                            X = np.array(features)
                            X_sc = current_models['lc_scaler'].transform(X)
                            lc_pred = current_models['lc_model'].predict(X_sc)
                            for i, pred in enumerate(lc_pred):
                                auv = auv_map[i]
                                old_role = auv.is_lc
                                auv.is_lc = (pred == 1)
                                auv.reliability_score = 0.98 if model_type == 'RF' else 0.95 if pred == 1 else 0.70
                                if old_role != auv.is_lc:
                                    auv.last_role_change = sim_t
                        except:
                            pass
                last_ml_update = sim_t
                mc_flow_table = invalidate_stale_paths(
                    mc_flow_table, local_auvs, sim_t)
                mc_flow_table = compute_all_paths_proactively(
                    sim_nodes, local_auvs, p['SURFACE_STATION_POS'], c.TX_RANGE, mc_flow_table)

        for auv in local_auvs:
            auv.update(dt, sim_nodes)
        auv.update_channels(c.P_BUSY)

        if len(pkt_events) > 0 and sim_t >= pkt_events[0]:
            pkt_events = pkt_events[1:]
            if src_pool:
                pkt_count += 1
                src = src_pool[pkt_count % len(src_pool)]
                curr = src
                logger.write_separator(LOG_LOCK)
                logger.write_trace(LOG_LOCK, sim_t, worker_id, "PKT_START",
                                   "NEW", f"ID:{pkt_count} Src:{src.node_id}")

                is_delivered = False
                packet_delay = 0.0
                path_steps = []

                # --- SDN MODE ---
                if use_sdn:
                    if src.node_id in mc_flow_table:
                        path_steps = mc_flow_table[src.node_id]
                        logger.write_trace(
                            LOG_LOCK, sim_t, worker_id, "FLOW", "HIT", f"Path: {len(path_steps)}")
                    else:
                        path_steps = dijkstra_paths(build_topology_graph(
                            sim_nodes, local_auvs, p['SURFACE_STATION_POS'], c.TX_RANGE), src.node_id)
                        if path_steps and len(path_steps) > 1:
                            path_steps = path_steps[1:]
                            mc_flow_table[src.node_id] = path_steps
                            packet_delay += 0.1 * len(path_steps)
                            c_bits += c.CONTROL_PACKET_LP * len(path_steps)

                    if path_steps:
                        path_broken = False
                        curr = src
                        for next_id in path_steps:
                            target = next(
                                (n for n in sim_nodes if n.node_id == next_id), None)
                            if not target:
                                if "SURFACE" in next_id:
                                    is_delivered = True
                                    break
                                aid = int(next_id.split("-")[1])
                                if aid < len(local_auvs):
                                    target = local_auvs[aid]

                            if not target:
                                path_broken = True
                                break

                            # [FIX] SDN Contention (0.3x)
                            if pkt_rng.random() < (c.P_BUSY * 0.3):
                                path_broken = True
                                logger.write_trace(
                                    LOG_LOCK, sim_t, worker_id, "FLOW", "DROP", "Priority Channel Busy")
                                del mc_flow_table[src.node_id]
                                break

                            t_pos = target.pos if not "AUV" in next_id else target.current_pos
                            dist = pe.distance_3d(curr.pos, t_pos)

                            if dist > c.TX_RANGE + c.SAFETY_MARGIN:
                                path_broken = True
                                del mc_flow_table[src.node_id]
                                break

                            pd, _, _, _ = pe.compute_transmission_delay_paper(
                                curr.pos, curr.depth, t_pos, target.depth, freqs[0], c.LP, c.CH_BANDWIDTH, pt_linear)

                            if pd == float('inf'):
                                path_broken = True
                                del mc_flow_table[src.node_id]
                                break

                            packet_delay += pd
                            c_bits += 32
                            curr = target

                        if not path_broken and is_delivered:
                            if packet_delay > 3.0:
                                is_delivered = False
                                logger.write_trace(
                                    LOG_LOCK, sim_t, worker_id, "DELIVERY", "FAIL", "Expired")
                            else:
                                logger.write_trace(
                                    LOG_LOCK, sim_t, worker_id, "DELIVERY", "SUCCESS", "SDN")
                        elif path_broken:
                            packet_delay += 0.5
                            is_delivered = False

                # --- BASELINE / RECOVERY ---
                if not is_delivered:
                    hops = 0
                    visited = {curr.node_id}

                    while hops < 20:  # [FIX] Reduced Hop Limit
                        hops += 1

                        is_lc_hit = False
                        if "AUV" in curr.node_id:
                            aid = int(curr.node_id.split("-")[1])
                            if aid < len(local_auvs) and local_auvs[aid].is_lc:
                                is_lc_hit = True

                        if curr.depth < 10.0 or is_lc_hit:
                            is_delivered = True
                            if use_sdn:
                                packet_delay += 0.2
                                c_bits += c.CONTROL_PACKET_LP
                            break

                        step_success = False

                        # [FIX] Strict Upward Only
                        candidates = find_greedy_next_hop(curr, sim_nodes, local_auvs, p['SURFACE_STATION_POS'],
                                                          c.TX_RANGE, freqs, c.LP, c.CH_BANDWIDTH, pt_linear, pkt_rng, p, visited)

                        for cand in candidates:
                            next_obj = cand['obj']
                            is_auv_hop = cand['is_auv']

                            # [FIX] Reduced Channel Retries (3)
                            channel_open = False
                            for _ in range(3):
                                if pkt_rng.random() >= c.P_BUSY:
                                    channel_open = True
                                    break
                            if not channel_open:
                                continue

                            dist = pe.distance_3d(
                                curr.pos, next_obj.pos if not is_auv_hop else next_obj.current_pos)
                            # [FIX] Hard Physics (Power 4)
                            prob_succ = 1.0 - (dist/c.TX_RANGE)**4
                            if is_auv_hop:
                                prob_succ = 1.0

                            # [FIX] Single Shot Physics (No ARQ)
                            if pkt_rng.random() <= prob_succ:
                                curr = next_obj
                                visited.add(curr.node_id)
                                packet_delay += cand['delay']
                                c_bits += c.CONTROL_PACKET_LP
                                logger.write_trace(
                                    LOG_LOCK, sim_t, worker_id, "DISC", "SUCCESS", f"Hop {curr.node_id}")
                                step_success = True
                                if is_auv_hop:
                                    is_delivered = True
                                break

                        if not step_success:
                            logger.write_trace(
                                LOG_LOCK, sim_t, worker_id, "ROUTING", "DROP", "Fail")
                            break

                        if is_delivered:
                            break

                    if is_delivered and packet_delay > 3.0:
                        is_delivered = False

                if is_delivered:
                    pkt_del += 1
                    d_bits += c.LP
                    tot_delay += packet_delay
                else:
                    c_bits += c.CONTROL_PACKET_LP

                curr_pdr = pkt_del / pkt_count
                logger.write_trace(LOG_LOCK, sim_t, worker_id,
                                   "METRICS", "SNAP", f"PDR:{curr_pdr:.2f}")

        sim_t += dt

    pkt_gen = pkt_count
    final_pdr = pkt_del / pkt_gen if pkt_gen > 0 else 0
    final_ecr = pe.calculate_ecr(
        d_bits, c_bits, int(p['N_NODES']), c.SWEEP_DURATION_S)
    final_ror = c_bits / (d_bits + c_bits) if (d_bits+c_bits) > 0 else 0
    final_e2ed = tot_delay / pkt_del if pkt_del > 0 else 0

    if LOG_LOCK:
        try:
            with LOG_LOCK:
                with open(LOG_FILE_PATH, "a") as f:
                    f.write(
                        f"{time.strftime('%H:%M:%S')},{model_type},{p['N_NODES']},{final_pdr:.4f},{final_ror:.4f},{final_ecr:.4f},{final_e2ed:.4f}\n")
        except:
            pass

    return (final_pdr, final_ror, final_ecr, final_e2ed)


class HeadlessSimulator:
    def __init__(self):
        self.graph_dir = os.path.join(script_dir, "ml", "Graphs")
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        try:
            with open(LOG_FILE_PATH, "w") as f:
                f.write("Time,Model,Nodes,PDR,RoR,ECR,E2ED\n")
        except:
            pass

    def _run_monte_carlo_point(self, p, runs=30, pool=None):
        base_seed = int(p['SEED']) if p['SEED'] != 0 else int(time.time())
        print(
            f"  > Simulating Point: {p['N_NODES']} Nodes, Mode: {p.get('MODEL_TYPE', 'None')}")
        param_list = []
        for i in range(runs):
            run_p = p.copy()
            run_p['NODE_SEED'] = base_seed + i*17
            run_p['AUV_SEED'] = base_seed + i*17 + 11
            param_list.append(run_p)

        results = pool.map(worker_simulation_task, param_list)
        if not results:
            return np.zeros(4)

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
            if trim_count < 1:
                trim_count = 1
            for col in [2, 3]:
                col_data = data[:, col]
                col_data.sort()
                valid_data = col_data[trim_count: -trim_count]
                final_metrics.append(np.mean(valid_data) if len(
                    valid_data) > 0 else np.mean(col_data))
        else:
            final_metrics.append(np.mean(data[:, 2]))
            final_metrics.append(np.mean(data[:, 3]))
        return np.array(final_metrics)

    def run_full_sweep(self):
        print("--- Starting Multicore Headless Comparative Sweep ---")
        num_cores = multiprocessing.cpu_count()
        print(f"--- Detected {num_cores} Cores. ---")

        base_p = {
            "N_NODES": c.DEFAULT_N_NODES, "NUM_AUVS": c.DEFAULT_NUM_AUVS, "NUM_PACKETS": c.DEFAULT_NUM_PACKETS,
            "AREA": c.AREA, "TX_RANGE": c.TX_RANGE, "P_BUSY": c.P_BUSY, "PT_DB": c.DEFAULT_PT_DB,
            "M_CHANNELS": c.DEFAULT_M_CHANNELS, "AUV_COVERAGE_RADIUS": c.AUV_COVERAGE_RADIUS,
            "W_TIME": c.DEFAULT_W_TIME, "W_ENERGY": c.DEFAULT_W_ENERGY, "NODE_DRIFT": c.NODE_MAX_DRIFT_M,
            "SEED": 42
        }
        base_p['SURFACE_STATION_POS'] = np.array(
            [base_p['AREA']/2, base_p['AREA']/2, 0.0])
        base_p["PT_LINEAR"] = 10**(base_p["PT_DB"] / 10.0)
        base_p["AUV_RELAY_RADIUS"] = float(base_p["AUV_COVERAGE_RADIUS"])

        static_node_counts = c.SWEEP_STATIC_NODES
        fixed_auvs = c.FIXED_AUVS
        base_p['NUM_AUVS'] = fixed_auvs

        x_total_nodes = [n + fixed_auvs for n in static_node_counts]

        scenarios = c.SCENARIOS
        results = {label: {'PDR': [], 'RoR': [], 'ECR': [], 'E2ED': []}
                   for label, _, _ in scenarios}

        m = multiprocessing.Manager()
        lock = m.Lock()

        with multiprocessing.Pool(processes=num_cores, initializer=worker_init, initargs=(project_root, lock)) as pool:
            for n_static in static_node_counts:
                print(
                    f"\nProcessing Static Nodes: {n_static} (Total: {n_static+fixed_auvs})")
                for label, use_auvs, model in scenarios:
                    p = base_p.copy()
                    p['N_NODES'] = n_static
                    p['USE_AUVS'] = use_auvs
                    p['MODEL_TYPE'] = model
                    p['NODE_DRIFT'] = float(
                        base_p['NODE_DRIFT']) if use_auvs else 0.0
                    avg = self._run_monte_carlo_point(
                        p, runs=c.MONTE_CARLO_RUNS, pool=pool)

                    results[label]['PDR'].append(avg[0])
                    results[label]['RoR'].append(avg[1])
                    results[label]['ECR'].append(avg[2])
                    results[label]['E2ED'].append(avg[3])

        print("\n--- Sweep Complete. Generating Plots ---")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sim = HeadlessSimulator()
    sim.run_full_sweep()
