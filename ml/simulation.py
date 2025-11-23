# unified_simulation.py
# A complete, merged GUI simulation for OSAR routing, AUV movement,
# real-time ML classification (LC/OC), and COMPARATIVE metric calculation.
# --- VERSION 19: FIXES PDR > 100% BUG.
# --- Ensures PDR is calculated using ACTUAL generated packets, not the config parameter.

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import csv
from datetime import datetime
import os
import sys
import traceback
import pandas as pd
import logging
import logging.handlers
import warnings

# Filter sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Add path correction to find 'auv.py' in parent dir ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- ML Imports ---
ML_AVAILABLE = True
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception as e:
    print("Warning: ML disabled.")
    ML_AVAILABLE = False
    joblib = None

# --- Import AUV components ---
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

plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# SECTION 1: GLOBAL PARAMETERS
# ==============================================================================

# --- OSAR Parameters ---
LP = 512
CONTROL_PACKET_LP = 64
CH_BANDWIDTH = 6e3
DEFAULT_PT_DB = 150.0
P_BUSY = 0.8
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10
DEFAULT_NUM_PACKETS = 20

# --- AUV Parameters ---
SVM_UPDATE_PERIOD_S = 5
AUV_UPDATE_INTERVAL_S = 0.1
SIMULATION_DURATION_S = 60*60
NODE_MAX_DRIFT_M = 10.0

# --- Sweep Parameters ---
SWEEP_DURATION_S = 1200
SWEEP_PACKET_INTERVAL_S = 30
SIM_SPEED_MULTIPLIER_NORMAL = 1.0
SIM_SPEED_MULTIPLIER_SWEEP = 1000.0

# --- Shared Parameters ---
AREA = 500.0
TX_RANGE = 250.0
DEFAULT_N_NODES = 30
DEFAULT_NUM_AUVS = 3
EPS = 1e-12

# --- Energy Parameters ---
E_BIT_TX = 1.0
E_AUV_MOVE_PER_S = 100.0
AUV_HOP_ENERGY_PENALTY = 50
DEFAULT_W_TIME = 0.5
DEFAULT_W_ENERGY = 0.5

# --- Environmental Parameters ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5

# ==============================================================================
# SECTION 2: ACOUSTIC & ROUTING CORE LOGIC
# ==============================================================================


def thorp_absorption_db_per_m(f_khz: float) -> float:
    f2 = f_khz**2
    db_per_km = 0.11 * f2 / (1 + f2) + 44 * f2 / \
        (4100 + f2) + 2.75e-4 * f2 + 0.00303
    return db_per_km / 1000.0


def path_loss_linear(d_m: float, f_khz: float, k: float = K_SPREAD) -> float:
    if d_m <= 0:
        d_m = EPS
    alpha_db_per_m = thorp_absorption_db_per_m(f_khz)
    a_linear_per_m = 10 ** (alpha_db_per_m / 10.0)
    return (d_m ** k) * (a_linear_per_m ** d_m)


def noise_psd_linear(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    f_khz_safe = f_khz + EPS
    Nt_db = 17 - 30 * np.log10(f_khz_safe)
    Ns_db = 40 + 20 * (shipping - 0.5) + 26 * \
        np.log10(f_khz_safe) - 60 * np.log10(f_khz_safe + 0.03)
    Nw_db = 50 + 7.5 * wind + 20 * \
        np.log10(f_khz_safe) - 40 * np.log10(f_khz_safe + 0.4)
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
        self.sound_speed = sound_speed(self.depth)
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


class AUV(BaseAUV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_lc = False
        self.is_oc = False


def compute_transmission_delay_paper(src, dest, dest_pos_target, f_khz, lp, bw, pt):
    src_pos = np.array(src.pos)
    dest_pos_actual = np.array(dest.pos)
    dest_pos_final_buoy = np.array(dest_pos_target)

    vec_id = dest_pos_final_buoy - src_pos
    vec_ij = dest_pos_actual - src_pos

    DiD = np.linalg.norm(vec_id)
    if DiD < EPS:
        DiD = EPS

    if np.linalg.norm(vec_ij) < EPS:
        proj_len = EPS
    else:
        proj_len = np.dot(vec_ij, vec_id) / DiD
    if proj_len <= EPS:
        proj_len = EPS

    N_hop = max(DiD / proj_len, 1.0)
    depth_diff = max(src.depth - dest.depth, 0.0)
    mid_depth_m = max((src.depth + dest.depth) / 2.0, 0.0)
    c = sound_speed(mid_depth_m)
    if c < EPS:
        c = 1500.0
    PD_ij = depth_diff / (c + EPS)
    dist_m = max(auv_distance(src_pos, dest_pos_actual), EPS)
    A_df = path_loss_linear(dist_m, f_khz)
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    if noise_total < EPS:
        noise_total = EPS
    pt_lin = max(pt, EPS)
    snr_linear = pt_lin / (A_df * noise_total + EPS)

    if snr_linear <= 0:
        r_ij_ch = EPS
    else:
        r_ij_ch = bw * np.log2(1.0 + snr_linear)
    if r_ij_ch <= EPS:
        return float('inf'), 0.0, 0.0, A_df

    TD = ((lp / r_ij_ch) + PD_ij) * N_hop
    return TD, r_ij_ch, snr_linear, A_df


def select_best_next_hop(src, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng, p):
    candidates = []
    W_time = p.get('W_TIME', DEFAULT_W_TIME)
    W_energy = p.get('W_ENERGY', DEFAULT_W_ENERGY)

    # --- Part 1: Static Nodes ---
    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        if distance(src.pos, nbr.pos) > tx_range: continue
        if nbr.depth >= src.depth: continue
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0: continue
        
        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0: continue

        best_td = float('inf')
        best_energy = float('inf')
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, _, A_df = compute_transmission_delay_paper(src, nbr, dest_pos, f_center_khz, lp, bw, pt)
            if TD < best_td:
                best_td, best_energy = TD, A_df
        
        if best_td != float('inf'):
            # False = Not an AUV
            candidates.append((nbr, best_td, best_energy, False))

    # --- Part 2: Mobile AUVs ---
    if p.get('USE_AUVS', True):
        for auv in auvs:
            if not (auv.is_lc or auv.is_oc): continue
            if distance(src.pos, auv.current_pos) > tx_range: continue
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
                TD, _, _, A_df = compute_transmission_delay_paper(src, dummy_auv, dest_pos, f_center_khz, lp, bw, pt)
                if TD < best_td:
                    best_td = TD
                    best_energy = A_df + AUV_HOP_ENERGY_PENALTY
            
            if best_td != float('inf'):
                # True = Is an AUV
                candidates.append((dummy_auv, best_td, best_energy, True))

    if not candidates: return None, None, None, None

    # Normalize and Select
    max_td = max((c[1] for c in candidates if c[1] != float('inf')), default=1.0)
    max_en = max((c[2] for c in candidates if c[2] != float('inf')), default=1.0)
    
    best_hop = None
    min_cost = float('inf')
    
    for (obj, td, en, is_auv) in candidates:
        cost = (W_time * (td/max_td)) + (W_energy * (en/max_en))
        
        # --- FORCED PRIORITY: Make AUVs 100x cheaper to force selection ---
        if is_auv:
            cost = cost / 100.0 
        # -----------------------------------------------------------------

        if cost < min_cost:
            min_cost = cost
            best_hop = (obj, td, None, None)
            
    return best_hop
# ==============================================================================
# SECTION 3: GUI APPLICATION
# ==============================================================================


class QueueHandler(logging.Handler):
    def __init__(self, queue_instance):
        super().__init__()
        self.queue = queue_instance

    def emit(self, record):
        self.queue.put({'type': 'log', 'message': self.format(record)})


class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OSAR + AUV + MC Comparative Simulation")
        self.root.geometry("1300x850")

        self.orchestrator_thread = None
        self.auv_thread = None
        self.mc_thread = None
        self.packet_thread = None
        self.sweep_thread = None

        self.simulation_running = threading.Event()
        self.stop_requested = False

        self.q = queue.Queue()
        self.sim_nodes = []
        self.auvs = []
        self.sc_nodes = None
        self.auv_artists = {}
        self.simulation_log_data = []
        self.osar_results = None
        self.auv_results = None

        self.logger = None
        self.setup_logging()

        self.lc_model = None
        self.lc_scaler = None
        self.oc_model = None
        self.oc_scaler = None
        self.models_loaded = False
        if ML_AVAILABLE:
            self.models_loaded = self.load_models(project_root)

        self.auv_available = AUV_AVAILABLE
        self.create_widgets()
        self.process_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_logging(self):
        self.logger = logging.getLogger('OSAR_Unified_Sim')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_file = os.path.join(script_dir, 'unified_simulation_log.txt')
        try:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(threadName)-15s] %(message)s', '%H:%M:%S'))
            self.logger.addHandler(fh)
        except:
            pass

        qh = QueueHandler(self.q)
        qh.setFormatter(logging.Formatter('[%(threadName)-15s] %(message)s'))
        self.logger.addHandler(qh)

    def load_models(self, model_dir):
        try:
            self.lc_model = joblib.load(os.path.join(
                model_dir, "Models/lc_model_small.joblib"))
            self.lc_scaler = joblib.load(os.path.join(
                model_dir, "Models/lc_scaler_small.joblib"))
            self.oc_model = joblib.load(os.path.join(
                model_dir, "Models/oc_model_small.joblib"))
            self.oc_scaler = joblib.load(os.path.join(
                model_dir, "Models/oc_scaler_small.joblib"))
            return True
        except:
            return False

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame = ttk.LabelFrame(
            left_panel, text="Simulation Parameters", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.params = {}

        param_list = {
            "N_NODES": ("Number of Nodes:", DEFAULT_N_NODES),
            "NUM_AUVS": ("Num AUVs:", DEFAULT_NUM_AUVS),
            "NUM_PACKETS": ("Num Packets:", DEFAULT_NUM_PACKETS),
            "AREA": ("Area Size (m):", AREA),
            "TX_RANGE": ("TX Range (m):", TX_RANGE),
            "P_BUSY": ("P(Busy Channel):", P_BUSY),
            "PT_DB": ("Transmit Power (dB):", DEFAULT_PT_DB),
            "M_CHANNELS": ("Num Channels:", DEFAULT_M_CHANNELS),
            "AUV_COVERAGE_RADIUS": ("AUV Coverage (m):", AUV_COVERAGE_RADIUS),
            "W_TIME": ("Weight (Time):", DEFAULT_W_TIME),
            "W_ENERGY": ("Weight (Energy):", DEFAULT_W_ENERGY),
            "SEED": ("Random Seed (0=random):", 0)
        }

        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(
                row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(
                row=i, column=1, sticky=tk.W, pady=2)

        self.toggle_run_button = ttk.Button(
            left_panel, text="Run Simulation", command=self.toggle_simulation)
        self.toggle_run_button.pack(fill=tk.X, pady=5)

        self.sweep_button = ttk.Button(
            left_panel, text="Run Comparative Sweep", command=self.start_sweep_experiment)
        self.sweep_button.pack(fill=tk.X, pady=5)

        self.export_button = ttk.Button(
            left_panel, text="Export Log to CSV", command=self.export_log_to_csv, state=tk.DISABLED)
        self.export_button.pack(fill=tk.X, pady=5)

        log_frame = ttk.LabelFrame(left_panel, text="Live Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=40, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if not self.models_loaded or not self.auv_available:
            self.toggle_run_button.config(state=tk.DISABLED)
            self.sweep_button.config(state=tk.DISABLED)

    def log(self, message):
        self.q.put({'type': 'log', 'message': message})

    def is_simulation_running(self):
        return (self.orchestrator_thread and self.orchestrator_thread.is_alive()) or \
               (self.sweep_thread and self.sweep_thread.is_alive())

    def toggle_simulation(self):
        if self.is_simulation_running():
            self.stop_simulation()
        else:
            self.start_simulation()

    def stop_simulation(self):
        if self.is_simulation_running():
            self.log("--- STOP REQUESTED ---")
            self.stop_requested = True
            self.simulation_running.clear()
            self.toggle_run_button.config(
                text="Stopping...", state=tk.DISABLED)
            self.sweep_button.config(state=tk.DISABLED)

    def start_simulation(self):
        if self.is_simulation_running():
            return
        self.log_text.delete('1.0', tk.END)
        self.log("--- Run Started ---")
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.init_hover_functionality()
        self.auv_artists.clear()
        self.sc_nodes = None
        self.simulation_log_data.clear()
        self.export_button.config(state=tk.DISABLED)
        self.auvs.clear()
        self.sim_nodes.clear()
        self.osar_results = None
        self.auv_results = None

        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            p["N_NODES"] = int(p["N_NODES"])
            p["NUM_AUVS"] = int(p["NUM_AUVS"])
            p["NUM_PACKETS"] = int(p["NUM_PACKETS"])
            p["M_CHANNELS"] = int(p["M_CHANNELS"])
            p["SEED"] = int(p["SEED"])
            p["PT_LINEAR"] = 10**(p["PT_DB"] / 10.0)
            p["AUV_RELAY_RADIUS"] = float(
                self.params["AUV_COVERAGE_RADIUS"].get())
            p['SURFACE_STATION_POS'] = np.array(
                [p['AREA']/2, p['AREA']/2, 0.0])
            p['USE_AUVS'] = True

            self.stop_requested = False
            self.simulation_running.set()
            self.orchestrator_thread = threading.Thread(
                target=self.run_concurrent_simulation, args=(p,), daemon=True, name="Orchestrator")
            self.orchestrator_thread.start()
            self.toggle_run_button.config(
                text="Stop Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.DISABLED)
        except Exception as e:
            self.log(f"Error: {e}")
            traceback.print_exc()

    def run_concurrent_simulation(self, p):
        try:
            seed = int(p['SEED']) if p['SEED'] != 0 else int(time.time())
            rng_nodes = np.random.default_rng(seed)
            self.sim_nodes = place_nodes_randomly(
                int(p['N_NODES']), p['AREA'], rng_nodes)

            nodes_pos = np.array([n.pos for n in self.sim_nodes])
            self.q.put({'type': 'plot_initial', 'nodes_pos': nodes_pos,
                       'buoy_pos': p['SURFACE_STATION_POS'], 'area': p['AREA']})
            time.sleep(0.5)

            self.auv_thread = threading.Thread(
                target=self.run_auv_thread, args=(p,), daemon=True, name='AUV-Thread')
            self.mc_thread = threading.Thread(
                target=self.run_mc_logic, args=(p,), daemon=True, name='MC-Thread')
            self.packet_thread = threading.Thread(
                target=self.run_packet_routing_thread, args=(p,), daemon=True, name='Packet-Thread')

            self.auv_thread.start()
            self.mc_thread.start()
            self.packet_thread.start()
            self.auv_thread.join()
            self.packet_thread.join()
            self.simulation_running.clear()
            self.mc_thread.join(timeout=1.0)
        except Exception as e:
            self.log(f"Error: {e}")
            traceback.print_exc()
        finally:
            self.simulation_running.clear()
            self.q.put({'type': 'finished'})

    def start_sweep_experiment(self):
        if self.is_simulation_running():
            return
        self.log_text.delete('1.0', tk.END)
        self.logger.info("--- Comparative Sweep Started ---")
        self.toggle_run_button.config(text="Stop Sweep", state=tk.NORMAL)
        self.sweep_button.config(state=tk.DISABLED)
        self.stop_requested = False
        self.sweep_thread = threading.Thread(
            target=self.run_sweep_thread, daemon=True, name='Sweep-Orch')
        self.sweep_thread.start()

    def run_sweep_thread(self):
        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            p['SURFACE_STATION_POS'] = np.array([p['AREA']/2, p['AREA']/2, 0.0])
            p["PT_LINEAR"] = 10**(p["PT_DB"] / 10.0)
            
            # Master seed from GUI
            base_seed = int(p['SEED']) if p['SEED'] != 0 else int(time.time())

            max_auvs = int(p.get('NUM_AUVS', DEFAULT_NUM_AUVS))
            sweep_counts = list(range(1, max_auvs + 1))
            
            final_results_with = []
            final_results_without = []
            
            # Monte Carlo runs for the "With AUV" case
            RUNS_PER_POINT = 5 

            self.sim_speed_multiplier = SIM_SPEED_MULTIPLIER_SWEEP
            self.simulation_running.set()

            for auv_count in sweep_counts:
                if self.stop_requested: break
                
                self.logger.info(f"=== Sweeping {auv_count} AUVs ===")
                
                # --- PART 1: WITH LC/OC (Monte Carlo Averaging) ---
                self.logger.info(f"   [Blue Line]: Running {RUNS_PER_POINT} iterations to smooth AUV variance...")
                temp_metrics_with = []
                
                for i in range(RUNS_PER_POINT):
                    if self.stop_requested: break
                    
                    # NODE_SEED is fixed (Same map)
                    # AUV_SEED varies (Different AUV paths)
                    current_p = p.copy()
                    current_p['NUM_AUVS'] = auv_count
                    current_p['NODE_SEED'] = base_seed 
                    current_p['AUV_SEED'] = base_seed + (auv_count * 1000) + i
                    current_p['USE_AUVS'] = True
                    
                    # Clear previous state
                    self.osar_results = None; self.auv_results = None; self.auvs.clear(); self.sim_nodes.clear()
                    self.q.put({'type': 'clear_plot'}); time.sleep(0.05)
                    
                    self._run_sweep_iteration(current_p)
                    
                    if self.osar_results and self.auv_results:
                        m = self.calculate_final_metrics(log_to_gui=False)
                        if m: temp_metrics_with.append(m)
                    
                    self.simulation_running.set()

                # Calculate Average for Blue Line
                if temp_metrics_with:
                    avg_w = np.mean(np.array(temp_metrics_with), axis=0)
                    final_results_with.append(tuple(avg_w))
                else:
                    final_results_with.append((0,0,0,0))


                # --- PART 2: WITHOUT LC/OC (Single Run) ---
                if self.stop_requested: break
                self.logger.info(f"   [Red Line]: Running Baseline (Static Network)...")
                
                # We only need to run this ONCE because nodes are static and fixed by NODE_SEED
                current_p = p.copy()
                current_p['NUM_AUVS'] = auv_count
                current_p['NODE_SEED'] = base_seed
                current_p['USE_AUVS'] = False # Disable AUV usage
                
                self.osar_results = None; self.auv_results = None; self.auvs.clear(); self.sim_nodes.clear()
                
                self._run_sweep_iteration(current_p)
                
                # Store single result for Red Line
                m = None
                if self.osar_results and self.auv_results:
                    m = self.calculate_final_metrics(log_to_gui=False)
                
                if m: final_results_without.append(m)
                else: final_results_without.append((0,0,0,0))
                    
                self.simulation_running.set()
                
                # Log progress
                self.logger.info(f"Completed {auv_count} AUVs. PDR (Avg With): {final_results_with[-1][0]:.2%}, PDR (Without): {final_results_without[-1][0]:.2%}")

            self.logger.info("Sweep Finished.")
            
            # Plot results
            min_len = min(len(sweep_counts), len(final_results_with), len(final_results_without))
            if min_len > 0:
                 self.q.put({'type': 'plot_comparative', 'x': sweep_counts[:min_len], 'y_with': final_results_with[:min_len], 'y_without': final_results_without[:min_len]})
            
            self.q.put({'type': 'finished'})

        except Exception as e:
            self.logger.error(f"Sweep Error: {e}")
            traceback.print_exc()
        finally:
            self.simulation_running.clear()
            self.sim_speed_multiplier = SIM_SPEED_MULTIPLIER_NORMAL
            self.q.put({'type': 'finished'})

    def _run_sweep_iteration(self, p):
        # --- USE CONSTANT NODE SEED ---
        # Defaults to standard 'SEED' if 'NODE_SEED' isn't set (for normal runs)
        seed_to_use = p.get('NODE_SEED', p['SEED'])
        rng_nodes = np.random.default_rng(int(seed_to_use))
        
        self.sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng_nodes)
        
        self.auv_thread = threading.Thread(target=self.run_auv_thread, args=(p,), daemon=True)
        self.mc_thread = threading.Thread(target=self.run_mc_logic, args=(p,), daemon=True)
        self.packet_thread = threading.Thread(target=self.run_packet_routing_thread, args=(p,), daemon=True)
        
        self.auv_thread.start(); self.mc_thread.start(); self.packet_thread.start()
        self.packet_thread.join()
        self.simulation_running.clear() 
        self.auv_thread.join(timeout=2.0); self.mc_thread.join(timeout=2.0)

        wait = time.time()
        while (not self.osar_results or not self.auv_results) and (time.time() - wait < 2.0):
            time.sleep(0.1)

    # --- FIX: Updated to use 'packets_generated' from the result payload, NOT from params ---
    def calculate_final_metrics(self, log_to_gui=True):
        if not self.osar_results or not self.auv_results: return None
        
        p = self.osar_results['params']
        
        # --- FIX: Get actual generated count from result data, not params ---
        pkt_gen = self.osar_results.get('packets_generated', 0) 
        # ------------------------------------------------------------------
        
        pkt_del = self.osar_results.get('packets_delivered', 0)
        bits_data = self.osar_results.get('total_data_bits', 0)
        bits_ctrl = self.osar_results.get('total_control_bits', 0)
        delay_sum = self.osar_results.get('total_end_to_end_delay', 0.0)
        
        if p.get('USE_AUVS', True):
            time_move = self.auv_results.get('total_auv_move_time', 0.0)
        else:
            time_move = 0.0 

        pdr = pkt_del / pkt_gen if pkt_gen > 0 else 0
        overhead = bits_ctrl / (bits_data + bits_ctrl) if (bits_data + bits_ctrl) > 0 else 0
        avg_delay = delay_sum / pkt_del if pkt_del > 0 else 0
        
        e_data = bits_data * E_BIT_TX
        e_ctrl = bits_ctrl * E_BIT_TX
        e_move = time_move * E_AUV_MOVE_PER_S
        ecr = (e_ctrl + e_move) / (e_data + e_ctrl + e_move) if (e_data + e_ctrl + e_move) > 0 else 0
        
        if log_to_gui:
            self.log(f"PDR: {pdr:.2%}, RoR: {overhead:.4f}, Delay: {avg_delay:.4f}s, ECR: {ecr:.4f}")

        return pdr, overhead, ecr, avg_delay

    def process_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                if data['type'] == 'log':
                    self.log_text.insert(tk.END, data['message'] + "\n")
                    self.log_text.see(tk.END)
                elif data['type'] == 'plot_initial':
                    self.plot_initial_setup(
                        data['nodes_pos'], data['buoy_pos'], data['area'])
                elif data['type'] == 'setup_auv_plots':
                    self.setup_auv_plots(data['num_auvs'])
                elif data['type'] == 'update_nodes':
                    self.update_node_plots(data['nodes_pos'])
                elif data['type'] == 'plot_auvs':
                    self.update_auv_plots(data['auv_indices'])
                elif data['type'] == 'plot_route':
                    if self.sim_speed_multiplier == SIM_SPEED_MULTIPLIER_NORMAL:
                        self.ax.plot(data['route_pos'][:, 0], data['route_pos'][:, 1],
                                     data['route_pos'][:, 2], color='blue' if data['success'] else 'red')
                        self.fig.canvas.draw_idle()
                elif data['type'] == 'osar_finished':
                    self.osar_results = data['data']
                elif data['type'] == 'auv_finished':
                    self.auv_results = data['data']

                elif data['type'] == 'plot_comparative':
                    self.plot_comparative(
                        data['x'], data['y_with'], data['y_without'])

                elif data['type'] == 'finished':
                    self.simulation_running.clear()
                    self.toggle_run_button.config(
                        text="Run Simulation", state=tk.NORMAL)
                    self.sweep_button.config(state=tk.NORMAL)
                    if self.osar_results and self.auv_results and not self.sweep_thread.is_alive():
                        self.calculate_final_metrics(log_to_gui=True)

        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def plot_comparative(self, x, y_with, y_without):
        self.fig.clear()
        axs = self.fig.subplots(2, 2)
        
        pdr_w = [d[0] for d in y_with]; pdr_wo = [d[0] for d in y_without]
        ror_w = [d[1] for d in y_with]; ror_wo = [d[1] for d in y_without]
        ecr_w = [d[2] for d in y_with]; ecr_wo = [d[2] for d in y_without]
        del_w = [d[3] for d in y_with]; del_wo = [d[3] for d in y_without]
        
        # --- Plot 1: PDR (Auto-Scaled) ---
        axs[0,0].plot(x, pdr_w, 'b-o', label='With LC/OC')
        axs[0,0].plot(x, pdr_wo, 'r--x', label='Without LC/OC')
        axs[0,0].set_title('PDR'); axs[0,0].set_ylabel('Ratio'); axs[0,0].grid(True); axs[0,0].legend()
        
        # Auto-scale PDR to zoom in on differences
        all_pdr = pdr_w + pdr_wo
        if all_pdr:
            min_p = min(all_pdr)
            max_p = max(all_pdr)
            if min_p == max_p: margin = 0.05
            else: margin = (max_p - min_p) * 0.1
            axs[0,0].set_ylim(max(0.0, min_p - margin), min(1.05, max_p + margin))

        # --- Plot 2: RoR ---
        axs[0,1].plot(x, ror_w, 'b-o', label='With LC/OC')
        axs[0,1].plot(x, ror_wo, 'r--x', label='Without LC/OC')
        axs[0,1].set_title('Overhead Ratio'); axs[0,1].grid(True)

        # --- Plot 3: ECR (Auto-Scaled) ---
        axs[1,0].plot(x, ecr_w, 'b-o', label='With LC/OC')
        axs[1,0].plot(x, ecr_wo, 'r--x', label='Without LC/OC')
        axs[1,0].set_title('Energy Cost Ratio')
        axs[1,0].set_xlabel('Num AUVs'); axs[1,0].grid(True)
        
        # Force tight view for ECR to see small changes
        all_ecr = ecr_w + ecr_wo
        if all_ecr:
            min_e = min(all_ecr) * 0.999
            max_e = max(all_ecr) * 1.001
            if min_e != max_e: axs[1,0].set_ylim(min_e, max_e)

        # --- Plot 4: Delay ---
        axs[1,1].plot(x, del_w, 'b-o', label='With LC/OC')
        axs[1,1].plot(x, del_wo, 'r--x', label='Without LC/OC')
        axs[1,1].set_title('End-to-End Delay (s)'); axs[1,1].set_xlabel('Num AUVs'); axs[1,1].grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def plot_initial_setup(self, nodes_pos, buoy_pos, area):
        self.init_hover_functionality()
        if nodes_pos.any():
            self.sc_nodes = self.ax.scatter(
                nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2], c=nodes_pos[:, 2], cmap='viridis_r')
        self.ax.scatter(buoy_pos[0], buoy_pos[1],
                        buoy_pos[2], c='red', marker='^', s=200)
        self.ax.set_xlim(0, area)
        self.ax.set_ylim(0, area)
        self.ax.set_zlim(area, 0)
        self.fig.canvas.draw_idle()

    def update_node_plots(self, nodes_pos):
        if self.sc_nodes:
            self.sc_nodes._offsets3d = (
                nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2])

    def setup_auv_plots(self, num_auvs):
        self.auv_artists.clear()
        for i in range(num_auvs):
            l, = self.ax.plot([], [], [], c='magenta', ls=':')
            m, = self.ax.plot([], [], [], c='magenta', marker='X')
            self.auv_artists[i] = {'path': l, 'marker': m}

    def update_auv_plots(self, indices):
        for i in indices:
            if i in self.auv_artists and i < len(self.auvs):
                auv = self.auvs[i]
                path = np.array(auv.traveled_path)
                pos = auv.current_pos
                if len(path) > 1:
                    self.auv_artists[i]['path'].set_data_3d(
                        path[:, 0], path[:, 1], path[:, 2])
                self.auv_artists[i]['marker'].set_data_3d(
                    [pos[0]], [pos[1]], [pos[2]])
                c = 'red' if auv.is_oc else 'cyan' if auv.is_lc else 'orange' if auv.state == "Returning to Charge" else 'magenta'
                self.auv_artists[i]['marker'].set_color(c)
        self.fig.canvas.draw_idle()

    def init_hover_functionality(self):
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(
            20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def hover(self, event):
        if event.inaxes == self.ax and self.sc_nodes:
            cont, ind = self.sc_nodes.contains(event)
            if cont:
                pos = self.sc_nodes.get_offsets()[ind["ind"][0]]
                self.annot.xy = pos
                self.annot.set_text(f"Pos: {pos}")
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def collect_auv_features(self):
        features = []
        auv_map = {}
        if not self.auvs:
            return None, None
        for i, auv in enumerate(self.auvs):
            count = sum(1 for n in self.sim_nodes if auv_distance(
                auv.current_pos, n.pos) < auv.coverage_radius)
            features.append([auv.current_pos[0], auv.current_pos[1],
                            auv.current_pos[2], auv.speed, count])
            auv_map[i] = auv
        return np.array(features), auv_map

    def update_controllers_svm(self, p):
        X, auv_map = self.collect_auv_features()
        if X is None or len(X) == 0:
            return
        for auv in self.auvs:
            auv.is_lc = False
            auv.is_oc = False

        try:
            X_sc = self.lc_scaler.transform(X)
            lc_pred = self.lc_model.predict(X_sc)
            valid_lcs = []
            for i, pred in enumerate(lc_pred):
                if pred == 1:
                    auv_map[i].is_lc = True
                    valid_lcs.append(i)

            if valid_lcs:
                X_oc = self.oc_scaler.transform(X)
                probs = self.oc_model.predict_proba(X_oc)[:, 1]
                best_idx = np.argmax(probs)
                auv_map[best_idx].is_oc = True
                auv_map[best_idx].is_lc = True

            self.q.put(
                {'type': 'plot_auvs', 'auv_indices': list(range(len(self.auvs)))})
        except:
            pass

    def run_mc_logic(self, p):
        if not self.models_loaded:
            return
        while len(self.auvs) == 0 and self.simulation_running.is_set():
            time.sleep(0.1)
        while self.simulation_running.is_set():
            time.sleep(SVM_UPDATE_PERIOD_S / self.sim_speed_multiplier)
            self.update_controllers_svm(p)

    def run_auv_thread(self, p):
        self.log("[AUV Sim]: AUV simulation thread started.")
        total_auv_move_time = 0.0
        try:
            start_time = time.time()
            
            # --- USE VARIABLE AUV SEED ---
            # Defaults to standard 'SEED' if 'AUV_SEED' isn't set
            seed_val = p.get('AUV_SEED', int(p['SEED']) + 1)
            rng = np.random.default_rng(int(seed_val))
            
            local_auv_list = []
            num_auvs = int(p['NUM_AUVS'])
            area = p['AREA']
            
            # Stratified/Random logic remains, but now driven by the variable 'rng'
            slice_width = area / num_auvs
            for i in range(num_auvs):
                spd = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                x_min = i * slice_width
                x_max = (i + 1) * slice_width
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(0, area)
                
                start_pt = np.array([x, y, 0.1])
                bottom_pt = np.array([x, y, area * 0.9])
                route = [start_pt, bottom_pt, start_pt] 

                local_auv_list.append(AUV(
                    i, spd, route, p['SURFACE_STATION_POS'], coverage_radius=p['AUV_COVERAGE_RADIUS']))
            
            self.auvs = local_auv_list
            self.q.put({'type': 'setup_auv_plots', 'num_auvs': len(local_auv_list)})

            last_log_time = 0.0
            last_speed_change_time = {auv.id: 0.0 for auv in local_auv_list}
            last_node_update_time = 0.0
            
            sim_duration = SIMULATION_DURATION_S
            if threading.current_thread().name.startswith("AUV-Sweep"):
                sim_duration = 300 # 5 minutes
                
            while self.simulation_running.is_set() and (time.time() - start_time) < sim_duration:
                current_time = time.time()
                updated_indices = []
                num_moving_auvs = 0
                
                for idx, auv in enumerate(local_auv_list):
                    if current_time - last_speed_change_time[auv.id] > rng.uniform(15, 30):
                        auv.speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                        last_speed_change_time[auv.id] = current_time

                    relayed_ids, is_moving, collected_new = auv.update(AUV_UPDATE_INTERVAL_S, self.sim_nodes)
                    
                    if is_moving: num_moving_auvs += 1
                    updated_indices.append(idx)
                
                total_auv_move_time += (num_moving_auvs * AUV_UPDATE_INTERVAL_S)

                if current_time - last_log_time >= 1.0:
                    last_log_time = current_time
                
                if current_time - last_node_update_time >= 1.0:
                    for node in self.sim_nodes:
                        node.update_position(NODE_MAX_DRIFT_M)
                    new_node_positions = np.array([n.pos for n in self.sim_nodes])
                    # Only plot node updates if NOT in fast sweep mode to save resources
                    if self.sim_speed_multiplier == SIM_SPEED_MULTIPLIER_NORMAL:
                        self.q.put({'type': 'update_nodes', 'nodes_pos': new_node_positions})
                    last_node_update_time = current_time

                if updated_indices and self.sim_speed_multiplier == SIM_SPEED_MULTIPLIER_NORMAL:
                    self.q.put({'type': 'plot_auvs', 'auv_indices': updated_indices})
                
                time.sleep(AUV_UPDATE_INTERVAL_S / self.sim_speed_multiplier)
            
            self.q.put({'type': 'auv_finished', 'data': {'total_auv_move_time': total_auv_move_time}})
            self.log("[AUV Thread] Finished.")

        except Exception as e:
            self.log(f"[AUV Thread] Error: {e}")
            traceback.print_exc()

    def run_packet_routing_thread(self, p):
        seed = int(p['SEED']) + 2
        rng = np.random.default_rng(seed)
        ch_hz = CH_BANDWIDTH/1e3
        start_f = START_FREQ_KHZ + ch_hz/2
        freqs = start_f + np.arange(p['M_CHANNELS']) * ch_hz

        while not self.sim_nodes and self.simulation_running.is_set():
            time.sleep(0.1)
        if not self.sim_nodes:
            return

        sorted_nodes = sorted(
            self.sim_nodes, key=lambda n: n.depth, reverse=True)
        src_pool = sorted_nodes[:8]

        pkt_gen = int(p['NUM_PACKETS'])
        pkt_del = 0
        d_bits = 0
        c_bits = 0
        tot_delay = 0.0

        is_sweep = self.sim_speed_multiplier > 1.0

        if is_sweep:
            dur = p.get('SWEEP_DURATION', SWEEP_DURATION_S)
            interval = SWEEP_PACKET_INTERVAL_S
            sim_t = 0.0
            pkt_count = 0
            while self.simulation_running.is_set() and sim_t < dur:
                time.sleep(interval / self.sim_speed_multiplier)
                sim_t += interval
                pkt_count += 1
                src = src_pool[pkt_count % len(src_pool)]
                res = self.route_single_packet(src, p, freqs, rng)
                if res:
                    pkt_del += 1
                    d_bits += LP
                    c_bits += res[1]
                    tot_delay += res[0]
                else:
                    c_bits += CONTROL_PACKET_LP * int(p['N_NODES'])
            pkt_gen = pkt_count
        else:
            for i in range(pkt_gen):
                if not self.simulation_running.is_set():
                    break
                src = src_pool[i % len(src_pool)]
                res = self.route_single_packet(src, p, freqs, rng)
                if res:
                    pkt_del += 1
                    d_bits += LP
                    c_bits += res[1]
                    tot_delay += res[0]
                    self.q.put(
                        {'type': 'plot_route', 'route_pos': res[2], 'success': True})
                else:
                    c_bits += CONTROL_PACKET_LP * 5

                time.sleep(0.5)

        self.q.put({'type': 'osar_finished', 'data': {
            'params': p, 'packets_generated': pkt_gen, 'packets_delivered': pkt_del,
            'total_data_bits': d_bits, 'total_control_bits': c_bits, 'total_end_to_end_delay': tot_delay
        }})

    def route_single_packet(self, src, p, freqs, rng):
        curr = src
        path = [src.pos]
        delay = 0.0
        ctrl_bits = 0

        for _ in range(int(p['N_NODES'])):
            for n in self.sim_nodes:
                n.update_channels(p['P_BUSY'])
            ctrl_bits += CONTROL_PACKET_LP

            best, td, _, _ = select_best_next_hop(curr, self.sim_nodes, self.auvs,
                                                  p['SURFACE_STATION_POS'], p['TX_RANGE'], freqs,
                                                  LP, CH_BANDWIDTH, p['PT_LINEAR'], rng, p)
            if not best:
                return None

            delay += td
            path.append(best.pos)
            if "AUV" in str(best.node_id):
                return (delay, ctrl_bits, path)

            curr = best
            if auv_distance(curr.pos, p['SURFACE_STATION_POS']) < p['TX_RANGE']:
                return (delay, ctrl_bits + 2*CONTROL_PACKET_LP, path)

        return None

    def export_log_to_csv(self): pass

    def on_closing(self):
        if self.is_simulation_running():
            self.stop_simulation()
        self.root.destroy()


if __name__ == "__main__":
    threading.current_thread().name = 'Main-GUI'
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()
