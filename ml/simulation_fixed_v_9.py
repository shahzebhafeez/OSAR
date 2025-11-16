# unified_simulation.py
# A complete, merged GUI simulation for OSAR routing, AUV movement,
# real-time ML classification (LC/OC), and full metric calculation.

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
import traceback  # For detailed error logging
import pandas as pd
import logging  # <-- For logging
import logging.handlers  # <-- For logging

# --- Add path correction to find 'auv.py' in parent dir ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of path fix ---

# --- ML Imports (robust) ---
ML_AVAILABLE = True
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception as e:
    print("Warning: scikit-learn or joblib not available. ML functionality disabled.")
    print("Install with: pip install scikit-learn joblib")
    ML_AVAILABLE = False
    joblib = None

# --- Import AUV components (robust) ---
AUV_AVAILABLE = True
try:
    from auv import (AUV as BaseAUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                      DEFAULT_AUV_RELAY_RADIUS,AUV_COVERAGE_RADIUS,LOW_BATTERY_THRESHOLD,distance,
                      distance as auv_distance) # Use auv.py's distance
except Exception as e:
    print(
        f"Warning: Could not import 'auv' (auv.py) from: {project_root}. AUV functionality disabled.\n{e}")
    AUV_AVAILABLE = False

    def auv_distance(a, b): # Placeholder
        return np.linalg.norm(np.array(a) - np.array(b))

    class BaseAUV: # Placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AUV module not available")

    def generate_auv_route(*args, **kwargs):
        raise RuntimeError("AUV module not available")
    
    AUV_MIN_SPEED = 1.0
    AUV_MAX_SPEED = 5.0
    DEFAULT_AUV_RELAY_RADIUS = 50.0

# --- Use a professional, publication-quality plot style ---
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# SECTION 1: GLOBAL PARAMETERS (MERGED)
# ==============================================================================

# --- OSAR Parameters ---
LP = 512  # Data packet size in bits (64 bytes)
CONTROL_PACKET_LP = 64 # Control packet size in bits (8 bytes)
CH_BANDWIDTH = 6e3  # 6 kHz bandwidth per channel
DEFAULT_PT_DB = 150.0 # Default transmit power in dB re uPa
P_BUSY = 0.8
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10
DEFAULT_NUM_PACKETS = 10

# --- AUV Parameters ---
SVM_UPDATE_PERIOD_S = 5 # How often the MC thread re-classifies AUVs
AUV_UPDATE_INTERVAL_S = 0.1 # From auv.py
SIMULATION_DURATION_S = 60*60 # From auv.py (AUVs run for 1 hour)
NODE_MAX_DRIFT_M = 10.0 # From auv.py

# --- Shared Parameters ---
AREA = 500.0
TX_RANGE = 250.0 # Max comms range for Node->Node and Node->AUV
DEFAULT_N_NODES = 30
DEFAULT_NUM_AUVS = 3
EPS = 1e-12

# --- NEW: Energy Cost Ratio (ECR) Constants ---
E_BIT_TX = 1.0 # Energy cost per bit transmitted
E_AUV_MOVE_PER_S = 1_000_000.0 # Energy cost per second of AUV movement

# --- Environmental Parameters (Assumed) ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5

# ==============================================================================
# SECTION 2: ACOUSTIC & ROUTING CORE LOGIC
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

# --- Merged Node Class ---
class Node:
    """Represents a sensor node that can drift and has channel states."""
    def __init__(self, node_id, pos, rng, channel_state=None):
        self.node_id = node_id
        self.initial_pos = np.copy(pos) # Anchor point
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.rng = rng
        self.sound_speed = sound_speed(self.depth)
        self.channel_state = channel_state if channel_state is not None else \
                             rng.random(M_SUBCARRIERS) < P_BUSY

    def update_position(self, max_drift_m):
        """Applies a random horizontal drift relative to the anchor point."""
        drift_x = self.rng.uniform(-max_drift_m, max_drift_m)
        drift_y = self.rng.uniform(-max_drift_m, max_drift_m)
        
        self.pos[0] = self.initial_pos[0] + drift_x
        self.pos[1] = self.initial_pos[1] + drift_y
        self.depth = self.pos[2] # Z position remains anchored

    def update_channels(self, p_busy):
        """Updates the node's channel busy status."""
        self.channel_state = self.rng.random(M_SUBCARRIERS) < p_busy

def place_nodes_randomly(N, area, rng):
    nodes = []
    for i in range(N):
        pos = rng.uniform(0, area, 3)
        pos[2] = rng.uniform(area * 0.1, area) # Start 10% down
        nodes.append(Node(i, pos, rng)) # Pass rng to Node
    return nodes

# --- Merged AUV Class ---
class AUV(BaseAUV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_lc = False
        self.is_oc = False
# --- End of Merged Class ---

def compute_transmission_delay_paper(src, dest, dest_pos_target, f_khz, lp, bw, pt):
    """Calculates hop delay from any 'src' (Node) to any 'dest' (Node or Dummy AUV Node)."""
    src_pos = np.array(src.pos)
    dest_pos_actual = np.array(dest.pos)
    dest_pos_final_buoy = np.array(dest_pos_target)
    
    vec_id = dest_pos_final_buoy - src_pos
    vec_ij = dest_pos_actual - src_pos
    
    DiD = np.linalg.norm(vec_id)
    if DiD < EPS: DiD = EPS
    
    if np.linalg.norm(vec_ij) < EPS:
         # If dest is the same as src, proj_len is 0
         proj_len = EPS
    else:
         # Project ij vector onto id vector
         proj_len = np.dot(vec_ij, vec_id) / DiD
         
    if proj_len <= EPS: proj_len = EPS # Only allow forward progress
    
    N_hop = max(DiD / proj_len, 1.0)
    
    src_depth = src.depth
    dest_depth = dest.depth
    
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

# --- MODIFIED: The new "Smart" routing function ---
def select_best_next_hop(src, nodes, auvs, dest_pos, tx_range, channels, lp, bw, pt, rng):
    """
    Finds the single best next hop, considering BOTH static nodes and mobile AUVs (LC/OC).
    """
    candidates = [] # Will hold tuples of (target_object, td, ch, snr)
    
    # --- Part 1: Check all Static Nodes ---
    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        if auv_distance(src.pos, nbr.pos) > tx_range: continue
        if nbr.depth >= src.depth: continue
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0: continue
        
        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0: continue

        best_td_for_nbr = float('inf')
        best_ch_for_nbr = None
        best_snr_for_nbr = None
        
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, snr_lin = compute_transmission_delay_paper(src, nbr, dest_pos, f_center_khz, lp, bw, pt)
            if TD < best_td_for_nbr:
                best_td_for_nbr = TD
                best_ch_for_nbr = ch_idx
                best_snr_for_nbr = snr_lin
        
        if best_td_for_nbr != float('inf'):
            candidates.append((nbr, best_td_for_nbr, best_ch_for_nbr, best_snr_for_nbr))

    # --- Part 2: Check all Mobile AUVs ---
    for auv in auvs:
        if not (auv.is_lc or auv.is_oc): continue
        if auv_distance(src.pos, auv.current_pos) > tx_range: continue
        if auv.current_pos[2] >= src.depth: continue
        if np.dot(dest_pos - src.pos, auv.current_pos - src.pos) <= 0: continue
        
        common_idle = np.where(~src.channel_state)[0]
        if len(common_idle) == 0: continue
        
        # Create a "dummy" Node object to represent the AUV
        dummy_auv_node = Node(f"AUV-{auv.id}", auv.current_pos, rng)
        dummy_auv_node.channel_state = np.array([False]*M_SUBCARRIERS) # Assume AUV is always idle
        
        best_td_for_auv = float('inf')
        best_ch_for_auv = None
        best_snr_for_auv = None

        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, snr_lin = compute_transmission_delay_paper(src, dummy_auv_node, dest_pos, f_center_khz, lp, bw, pt)
            if TD < best_td_for_auv:
                best_td_for_auv = TD
                best_ch_for_auv = ch_idx
                best_snr_for_auv = snr_lin
        
        if best_td_for_auv != float('inf'):
            candidates.append((dummy_auv_node, best_td_for_auv, best_ch_for_auv, best_snr_for_auv))

    # --- Part 3: Find the best overall candidate ---
    if not candidates: 
        return None, None, None, None
        
    best_hop, best_td, best_ch, best_snr = min(candidates, key=lambda x: x[1])
    return best_hop, best_td, best_ch, best_snr
# --- End of New Function ---


# ==============================================================================
# SECTION 3: GUI APPLICATION
# ==============================================================================

class QueueHandler(logging.Handler):
    def __init__(self, queue_instance):
        super().__init__()
        self.queue = queue_instance

    def emit(self, record):
        msg = self.format(record)
        self.queue.put({'type': 'log', 'message': msg})

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OSAR + AUV + MC Unified Simulation")
        self.root.geometry("1300x850")
        
        self.orchestrator_thread = None
        self.auv_thread = None
        self.mc_thread = None
        self.packet_thread = None
        self.sweep_thread = None
        
        self.simulation_running = threading.Event()
        self.q = queue.Queue()
        self.sim_nodes = []
        self.auvs = []
        self.sc_nodes = None
        self.auv_artists = {}
        self.simulation_log_data = [] 
        
        self.auv_results = None
        self.osar_results = None
        
        self.logger = None 
        self.setup_logging()

        self.lc_model = None
        self.lc_scaler = None
        self.oc_model = None
        self.oc_scaler = None
        self.models_loaded = False
        if ML_AVAILABLE:
            self.models_loaded = self.load_models(project_root)
        else:
            self.logger.warning("ML not available; models will not be loaded.")
        
        self.auv_available = AUV_AVAILABLE

        self.create_widgets()
        self.process_queue()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_logging(self):
        """Configures the logger to send messages to the GUI queue and a file."""
        self.logger = logging.getLogger('OSAR_Unified_Sim')
        self.logger.setLevel(logging.DEBUG) 
        self.logger.propagate = False
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 1. File Handler
        log_file_path = ""
        try:
            log_file_path = os.path.join(script_dir, 'unified_simulation_log.txt')
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(threadName)-19s] [%(levelname)-5s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"CRITICAL: Failed to create log file at {log_file_path}. Error: {e}")

        # 2. Queue Handler
        gui_formatter = logging.Formatter(
            '[%(threadName)-19s] %(message)s'
        )
        queue_handler = QueueHandler(self.q)
        queue_handler.setFormatter(gui_formatter)
        self.logger.addHandler(queue_handler)
        
        self.logger.info(f"Logging configured. Log file: {log_file_path if log_file_path else 'FILE FAILED'}")


    def load_models(self, model_dir):
        model_files = {
            "lc_model": os.path.join(model_dir, "lc_model_small.joblib"),
            "lc_scaler": os.path.join(model_dir, "lc_scaler_small.joblib"),
            "oc_model": os.path.join(model_dir, "oc_model_small.joblib"),
            "oc_scaler": os.path.join(model_dir, "oc_scaler_small.joblib")
        }
        try:
            self.logger.info(f"Attempting to load models from: {model_dir}")
            self.lc_model = joblib.load(model_files["lc_model"])
            self.lc_scaler = joblib.load(model_files["lc_scaler"])
            self.oc_model = joblib.load(model_files["oc_model"])
            self.oc_scaler = joblib.load(model_files["oc_scaler"])
            self.logger.info("Successfully loaded all SVM models and scalers.")
            return True
        except FileNotFoundError as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.error(
                f"Please ensure models (e.g., 'lc_model_small.joblib') exist in: {model_dir}")
            return False
        except Exception as e:
            self.logger.critical(f"An error occurred loading models: {e}", exc_info=True)
            return False

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding="10")
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
            "SEED": ("Random Seed (0=random):", 0)
        }
        
        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        self.toggle_run_button = ttk.Button(left_panel, text="Run Simulation", command=self.toggle_simulation)
        self.toggle_run_button.pack(fill=tk.X, pady=5)
        
        self.sweep_button = ttk.Button(left_panel, text="Run AUV Sweep Plot", command=self.start_sweep_experiment)
        self.sweep_button.pack(fill=tk.X, pady=5)
        
        if not self.models_loaded or not self.auv_available:
            self.toggle_run_button.config(text="Models/AUV Not Found", state=tk.DISABLED)
            self.sweep_button.config(text="Models/AUV Not Found", state=tk.DISABLED)
        
        self.export_button = ttk.Button(left_panel, text="Export Log to CSV", command=self.export_log_to_csv, state=tk.DISABLED)
        self.export_button.pack(fill=tk.X, pady=5)

        log_frame = ttk.LabelFrame(left_panel, text="Live Log & Data", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=40, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        if not self.models_loaded:
            self.logger.error(
                f"Error: SVM models not found in {project_root}\nPlease run 'data_collector.py' and 'train_model.py'.")
        if not self.auv_available:
            self.logger.error(
                "Error: AUV components (auv.py) not found in project root. AUV simulation disabled.")

    def init_hover_functionality(self):
        """Initializes the annotation box for hovering."""
        try:
            self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="yellow", alpha=0.8),
                                          arrowprops=dict(arrowstyle="->"))
            self.annot.set_visible(False)
            self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        except Exception as e:
            self.logger.warning(f"Could not init hover functionality: {e}")

    def update_annot(self, ind):
        """Updates the annotation text when hovering over a node."""
        try:
            node_index = ind["ind"][0]
            if node_index >= len(self.sim_nodes): return
            
            node = self.sim_nodes[node_index]
            
            # Use self.sc_nodes (the scatter plot for nodes) to get position
            offsets = self.sc_nodes.get_offsets()
            if node_index >= len(offsets): return
            
            pos = offsets[node_index]
            self.annot.xy = pos
            text = (f"ID: {node.node_id}\nPos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                    f"Depth: {node.depth:.1f} m")
            self.annot.set_text(text)
        except Exception as e:
            self.logger.warning(f"Error in update_annot: {e}")

    def hover(self, event):
        """Handles the mouse hover event to show/hide the annotation."""
        vis = self.annot.get_visible()
        try:
            # Check if the hover is over the node scatter plot
            if event.inaxes == self.ax and self.sc_nodes:
                cont, ind = self.sc_nodes.contains(event)
                if cont:
                    self.update_annot(ind)
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                elif vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
        except Exception as e:
             self.logger.warning(f"Error in hover event: {e}")

    def log(self, message):
        """Thread-safe logging to the GUI."""
        self.q.put({'type': 'log', 'message': message})

    def is_simulation_running(self):
        orch_alive = self.orchestrator_thread and self.orchestrator_thread.is_alive()
        sweep_alive = self.sweep_thread and self.sweep_thread.is_alive()
        return orch_alive or sweep_alive

    def toggle_simulation(self):
        if self.is_simulation_running():
            self.stop_simulation()
        else:
            self.start_simulation()

    def stop_simulation(self):
        if self.is_simulation_running():
            self.log("--- SIMULATION STOP REQUESTED BY USER ---")
            self.simulation_running.clear()
            self.toggle_run_button.config(text="Stopping...", state=tk.DISABLED)
            self.sweep_button.config(state=tk.DISABLED)

    def start_simulation(self):
        if self.is_simulation_running(): return
        
        self.log_text.delete('1.0', tk.END)
        self.log("--- New Unified Simulation Run Started ---")
        
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
            p["AUV_RELAY_RADIUS"] = float(self.params["AUV_COVERAGE_RADIUS"].get())
            p['SURFACE_STATION_POS'] = np.array([p['AREA']/2, p['AREA']/2, 0.0])
            
            if not self.models_loaded:
                self.log("Error: Cannot start, ML models not loaded.")
                return
            if not self.auv_available:
                self.log("Error: Cannot start, AUV module not available.")
                return
            
            self.simulation_running.set()
            self.orchestrator_thread = threading.Thread(target=self.run_concurrent_simulation, args=(p,), daemon=True, name="Orchestrator")
            self.orchestrator_thread.start()
            
            self.toggle_run_button.config(text="Stop Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.DISABLED)
            
        except ValueError: self.log("Error: Please enter valid numbers.")
        except Exception as e:
            self.log(f"Error starting sim: {e}")
            traceback.print_exc()
            self.toggle_run_button.config(text="Run AUV Simulation", state=tk.NORMAL)

    def run_concurrent_simulation(self, p):
        """
        Orchestrator: Creates nodes, then starts AUV, MC, and Packet threads
        to run concurrently.
        """
        try:
            self.log(f"Creating {int(p['N_NODES'])} sensor nodes...")
            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng_nodes = np.random.default_rng(seed)
            self.sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng_nodes)
            self.log(f"Created {len(self.sim_nodes)} nodes.")
            
            nodes_pos = np.array([n.pos for n in self.sim_nodes]) if self.sim_nodes else np.array([])
            self.q.put({'type': 'plot_initial', 'nodes_pos': nodes_pos, 'buoy_pos': p['SURFACE_STATION_POS'], 'area': p['AREA']})
            time.sleep(0.5) 

            self.log("\n--- Starting AUV + MC + Packet Threads (Concurrent) ---")
            self.auv_thread = threading.Thread(
                target=self.run_auv_thread, args=(p,), daemon=True, name='AUV-Thread')
            self.mc_thread = threading.Thread(
                target=self.run_mc_logic, args=(p,), daemon=True, name='MC-Thread')
            self.packet_thread = threading.Thread(
                target=self.run_packet_routing_thread, args=(p,), daemon=True, name='Packet-Routing-Thread')

            self.auv_thread.start()
            self.mc_thread.start()
            self.packet_thread.start()

            self.auv_thread.join()
            self.packet_thread.join()
            
            if not self.simulation_running.is_set():
                 self.log("Simulation stopped by user.")
                 return

            self.log("--- AUV and Packet Threads Finished ---")

            self.log("--- Stopping MC Controller ---")
            self.simulation_running.clear()
            self.mc_thread.join(timeout=2.0)

            self.log("\n--- All Simulations Finished ---")

        except Exception as e:
            self.log(f"Orchestrator Error: {e}")
            traceback.print_exc()
        finally:
            self.simulation_running.clear()
            self.log("[Orchestrator Thread] Finished.")
            self.q.put({'type': 'finished'}) 
    
    def start_sweep_experiment(self):
        if self.is_simulation_running():
            self.logger.warning("Simulation is already running.")
            return

        self.log_text.delete('1.0', tk.END)
        self.logger.info("--- New AUV Sweep Run Started ---")

        self.toggle_run_button.config(text="Stop Sweep", state=tk.NORMAL)
        self.sweep_button.config(state=tk.DISABLED)
        
        self.sweep_thread = threading.Thread(
            target=self.run_sweep_thread, daemon=True, name='Sweep-Orchestrator')
        self.sweep_thread.start()

    def run_sweep_thread(self):
        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            # --- FIX: Add the missing SURFACE_STATION_POS key ---
            self.surface_station_pos = np.array([p['AREA']/2, p['AREA']/2, 0.0])
            p['SURFACE_STATION_POS'] = self.surface_station_pos
            # --- NEW FIX: Add the missing PT_LINEAR key ---
            p["PT_LINEAR"] = 10**(p["PT_DB"] / 10.0)
            # --- End of Fix ---
            # --- End of Fix ---
            max_auvs = int(p.get('NUM_AUVS', DEFAULT_NUM_AUVS))
            if max_auvs <= 0:
                self.logger.error("Cannot run sweep with 0 AUVs. Set 'Num AUVs' > 0.")
                self.q.put({'type': 'finished'})
                return
            
            sweep_auv_counts = list(range(1, max_auvs + 1))
            self.logger.info(f"--- Starting AUV Sweep (1 to {max_auvs} AUVs) ---")
            experiment_results = []
            
            self.simulation_running.set()

            for auv_count in sweep_auv_counts:
                if not self.simulation_running.is_set():
                    self.logger.warning(f"AUV Sweep cancelled by user at {auv_count} AUVs.")
                    break
                if not ML_AVAILABLE or not AUV_AVAILABLE:
                    self.logger.error("Sweep cancelled: ML or AUV module missing.")
                    break 

                self.logger.info(f"\n--- EXPERIMENT: Running for {auv_count} AUV(s) ---")

                current_p = p.copy()
                current_p['NUM_AUVS'] = auv_count
                self.params['NUM_AUVS'].set(str(auv_count)) # Update GUI

                self.osar_results = None
                self.auv_results = None
                self.auvs.clear()
                self.sim_nodes.clear()
                self.q.put({'type': 'clear_plot'})
                time.sleep(0.1)

                # --- 1. Create sensor nodes ---
                self.log(f"[Sweep]: Creating {int(current_p['N_NODES'])} sensor nodes...")
                seed = current_p['SEED'] if current_p['SEED'] != 0 else int(time.time())
                rng_nodes = np.random.default_rng(seed)
                self.sim_nodes = place_nodes_randomly(int(current_p['N_NODES']), current_p['AREA'], rng_nodes)
                self.log(f"[Sweep]: Created {len(self.sim_nodes)} nodes.")
                
                # --- 2. Run All Threads Concurrently ---
                self.log("[Sweep]: Starting AUV + MC + Packet Threads...")
                self.auv_thread = threading.Thread(
                    target=self.run_auv_thread, args=(current_p,), daemon=True, name=f'AUV-Sweep-{auv_count}')
                self.mc_thread = threading.Thread(
                    target=self.run_mc_logic, args=(current_p,), daemon=True, name=f'MC-Sweep-{auv_count}')
                self.packet_thread = threading.Thread(
                    target=self.run_packet_routing_thread, args=(current_p,), daemon=True, name=f'Pkt-Sweep-{auv_count}')

                self.auv_thread.start()
                self.mc_thread.start()
                self.packet_thread.start()

                # --- 3. Wait for AUV and Packet threads ---
                # For a sweep, we run AUVs for a *limited time*, not 1 hour
                # We wait for packets to finish, then stop AUVs
                self.packet_thread.join()
                self.log("[Sweep]: Packet routing finished.")
                
                # Stop the AUV and MC threads
                # (We can't wait an hour for AUVs in a sweep)
                self.simulation_running.clear()
                self.auv_thread.join(timeout=5.0)
                self.mc_thread.join(timeout=5.0)
                self.log("[Sweep]: AUV and MC threads stopped.")
                
                self.log(f"[Sweep]: Run for {auv_count} AUVs finished. Collecting results...")

                # --- 4. Collect Results ---
                wait_start_time = time.time()
                while (not self.osar_results or not self.auv_results) and (time.time() - wait_start_time < 10.0):
                    time.sleep(0.2)
                
                if not self.osar_results or not self.auv_results:
                    self.logger.error(f"--- EXPERIMENT ERROR: Timeout waiting for results for {auv_count} AUVs ---")
                    continue

                pdr, ror, ecr = self.calculate_final_metrics(log_to_gui=False)

                if pdr is not None and ror is not None and ecr is not None:
                    experiment_results.append(
                        {'auvs': auv_count, 'pdr': pdr, 'ror': ror, 'ecr': ecr})
                    self.logger.info(f"  -> Results: PDR={pdr:.3f}, RoR={ror:.3f}, ECR={ecr:.3f}")
                else:
                    self.logger.error(f"--- EXPERIMENT ERROR: Failed to calculate metrics for {auv_count} AUVs ---")
                
                self.simulation_running.set() # Re-set the flag for the next loop iteration

            # --- Experiment Finished ---
            self.logger.info("\n--- EXPERIMENT SWEEP FINISHED ---")
            if experiment_results:
                self.q.put({'type': 'plot_experiment_results',
                            'data': experiment_results})
            else:
                self.logger.warning("No results to plot.")
                self.q.put({'type': 'finished'}) 

        except Exception as e:
            self.logger.critical(f"[Sweep Thread] FATAL ERROR: {e}", exc_info=True)
        finally:
            self.simulation_running.clear()
            self.q.put({'type': 'finished'})
            self.logger.info("[Sweep Thread] Finished.")

    def calculate_final_metrics(self, log_to_gui=True):
        if not self.osar_results:
            if log_to_gui: self.logger.error("Error: Missing OSAR results, cannot calculate metrics.")
            return None, None, None
        if not self.auv_results:
            if log_to_gui: self.logger.error("Error: Missing AUV results, cannot calculate metrics.")
            return None, None, None

        if log_to_gui:
            self.logger.info("\n\n--- FINAL SIMULATION STATS ---")

        p = self.osar_results['params']
        packets_generated = p.get('NUM_PACKETS', 0)
        packets_delivered = self.osar_results.get('packets_delivered', 0)
        total_data_bits = self.osar_results.get('total_data_bits', 0)
        total_control_bits = self.osar_results.get('total_control_bits', 0)
        total_end_to_end_delay = self.osar_results.get('total_end_to_end_delay', 0.0)
        
        total_auv_move_time = self.auv_results.get('total_auv_move_time', 0.0)

        # --- PDR ---
        pdr = packets_delivered / packets_generated if packets_generated > 0 else 0
        if log_to_gui:
            self.logger.info(
                f"Packet Delivery Ratio (PDR): {pdr:.2%} ({packets_delivered}/{packets_generated})")

        # --- RoR ---
        total_bits = total_data_bits + total_control_bits
        overhead_ratio = total_control_bits / total_bits if total_bits > 0 else 0
        if log_to_gui:
            self.logger.info(f"Overhead Ratio (RoR) (by bytes): {overhead_ratio:.4f}")

        # --- E2E Delay ---
        avg_delay = total_end_to_end_delay / packets_delivered if packets_delivered > 0 else float('inf')
        if log_to_gui:
            self.logger.info(
                f"Average End-to-End Delay (for delivered packets): {avg_delay:.4f} s")

        # --- ECR ---
        energy_data = total_data_bits * E_BIT_TX
        energy_control = total_control_bits * E_BIT_TX
        energy_movement = total_auv_move_time * E_AUV_MOVE_PER_S
        energy_management = energy_control + energy_movement
        total_energy = energy_data + energy_management
        ecr = energy_management / total_energy if total_energy > 0 else 0

        if log_to_gui:
            self.logger.info(f"Energy Cost Ratio (ECR): {ecr:.4f}")
            self.logger.info(
                f" (Energy_Data: {energy_data:,.0f}, Energy_Control: {energy_control:,.0f}, Energy_Movement: {energy_movement:,.0f})")

        return pdr, overhead_ratio, ecr

    def process_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                if data['type'] == 'log': self.log_text.insert(tk.END, data['message'] + "\n"); self.log_text.see(tk.END)
                elif data['type'] == 'plot_initial': self.plot_initial_setup(data['nodes_pos'], data['buoy_pos'], data['area'])
                elif data['type'] == 'setup_auv_plots': self.setup_auv_plots(data['num_auvs'])
                elif data['type'] == 'update_nodes': self.update_node_plots(data['nodes_pos'])
                elif data['type'] == 'plot_auvs': self.update_auv_plots(data['auv_indices'])
                
                elif data['type'] == 'plot_route':
                    route_pos = np.array(data['route_pos'])
                    color = "mediumblue" if data['success'] else 'crimson'
                    style = '-' if data['success'] else '--'
                    self.ax.plot(route_pos[:, 0], route_pos[:, 1], route_pos[:, 2],
                                 color=color, linewidth=2.5, marker='o', linestyle=style,
                                 markersize=6, markerfacecolor='orange', zorder=5)
                    self.fig.canvas.draw_idle()
                
                elif data['type'] == 'osar_finished':
                    self.osar_results = data['data']
                    self.logger.debug("[Queue]: OSAR results received.")
                    if self.auv_results and not self.is_simulation_running():
                        self.calculate_final_metrics(log_to_gui=True)
                        
                elif data['type'] == 'auv_finished':
                    self.auv_results = data['data']
                    self.logger.debug("[Queue]: AUV results received.")
                    if self.osar_results and not self.is_simulation_running():
                        self.calculate_final_metrics(log_to_gui=True)
                
                elif data['type'] == 'plot_experiment_results':
                    self.plot_sweep_results(data['data'])
                
                elif data['type'] == 'finished':
                    self.simulation_running.clear()
                    self.log("\n--- Simulation Finished ---")
                    # Check for results before enabling metrics
                    if self.osar_results and self.auv_results:
                         self.calculate_final_metrics(log_to_gui=True)
                    if self.simulation_log_data:
                        self.export_button.config(state=tk.NORMAL)
                    
        except queue.Empty: pass
        except Exception as e:
            print(f"Error in process_queue: {e}")
            traceback.print_exc()
            self.log(f"Error processing GUI updates: {e}")
            self.simulation_running.clear()

        if not self.is_simulation_running():
            if self.toggle_run_button['state'] != tk.NORMAL:
                self.toggle_run_button.config(text="Run AUV Simulation", state=tk.NORMAL)
                self.sweep_button.config(text="Run AUV Sweep Plot", state=tk.NORMAL)
                if self.simulation_log_data:
                    self.export_button.config(state=tk.NORMAL)
        
        self.root.after(100, self.process_queue)

    def plot_initial_setup(self, nodes_pos, buoy_pos, area):
        self.init_hover_functionality()
        
        if nodes_pos.any():
            self.sc_nodes = self.ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2],
                                         c=nodes_pos[:, 2], cmap="viridis_r", s=50, alpha=0.9,
                                         edgecolors='black', linewidth=0.6, label="Sensor Nodes")
        self.ax.scatter(buoy_pos[0], buoy_pos[1], buoy_pos[2], c="red", s=180, marker="^",
                        label="Surface Buoy", edgecolors='black', linewidth=1.2, depthshade=False)
        
        self.ax.scatter([], [], [], c='magenta', s=150, marker='X', label="AUV (Normal)")
        self.ax.scatter([], [], [], c='orange', s=150, marker='X', label="AUV (Charging)")
        self.ax.scatter([], [], [], c='cyan', s=250, marker='X', label="AUV (LC)")
        self.ax.scatter([], [], [], c='red', s=300, marker='X', label="AUV (OC)")
        
        self.ax.set_xlabel("X (m)", fontweight='bold')
        self.ax.set_ylabel("Y (m)", fontweight='bold')
        self.ax.set_zlabel("Depth (m)", fontweight='bold')
        self.ax.set_title("AUV Path & MC Classification Simulation", fontsize=14, fontweight='bold')
        self.ax.invert_zaxis()
        self.ax.set_xlim(0, area)
        self.ax.set_ylim(0, area)
        self.ax.set_zlim(area, 0)
        self.ax.legend(loc='upper left')
        self.ax.view_init(elev=25, azim=-75)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def update_node_plots(self, nodes_pos):
        if self.sc_nodes and len(nodes_pos) > 0:
            self.sc_nodes._offsets3d = (nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2])

    def setup_auv_plots(self, num_auvs):
        self.auv_artists.clear()
        for i in range(num_auvs):
            path_line, = self.ax.plot([], [], [], 
                                      color='magenta', linestyle=':', 
                                      linewidth=1.5, zorder=9, alpha=0.7)
            marker_point, = self.ax.plot([], [], [], 
                                         color='magenta', markersize=15, marker='X',
                                         zorder=10, markeredgecolor='black')
            
            self.auv_artists[i] = {
                'path': path_line,
                'marker': marker_point
            }
        self.fig.canvas.draw_idle()

    def update_auv_plots(self, auv_indices):
        for idx in auv_indices:
            if idx in self.auv_artists and idx < len(self.auvs):
                auv = self.auvs[idx]
                artists = self.auv_artists[idx]
                
                path_data = np.array(auv.traveled_path)
                if path_data.shape[0] > 1:
                    artists['path'].set_data_3d(path_data[:, 0], path_data[:, 1], path_data[:, 2])
                
                pos = auv.current_pos
                artists['marker'].set_data_3d([pos[0]], [pos[1]], [pos[2]])
                
                color = 'magenta' # Default
                if auv.state == "Returning to Charge":
                    color = 'orange'
                
                if auv.is_lc:
                    color = 'cyan'
                if auv.is_oc:
                    color = 'red'
                
                artists['path'].set_color(color)
                artists['marker'].set_color(color)
                
        self.fig.canvas.draw_idle()
        
    def plot_sweep_results(self, results):
        self.fig.clear() 
        
        auv_counts = [r['auvs'] for r in results]
        pdr_data = [r['pdr'] for r in results]
        ror_data = [r['ror'] for r in results]
        ecr_data = [r['ecr'] for r in results]

        ax1 = self.fig.add_subplot(311)
        ax1.plot(auv_counts, pdr_data, marker='o', color='blue', label='PDR')
        ax1.set_title('Simulation Metrics vs. Number of AUVs', fontsize=14, fontweight='bold')
        ax1.set_ylabel("Packet Delivery Ratio (PDR)", fontweight='bold')
        if pdr_data:
            ax1.set_ylim(0, max(1.1, max(pdr_data) * 1.1))
        ax1.set_xticks(auv_counts)
        ax1.grid(True)
        
        ax2 = self.fig.add_subplot(312, sharex=ax1)
        ax2.plot(auv_counts, ror_data, marker='s', color='green', label='RoR')
        ax2.set_ylabel("Overhead Ratio (RoR)", fontweight='bold')
        ax2.grid(True)

        ax3 = self.fig.add_subplot(313, sharex=ax1)
        ax3.plot(auv_counts, ecr_data, marker='^', color='red', label='ECR')
        ax3.set_ylabel("Energy Cost Ratio (ECR)", fontweight='bold')
        ax3.set_xlabel("Number of AUVs", fontweight='bold')
        ax3.grid(True)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.canvas.draw_idle()

    def export_log_to_csv(self):
        if not self.simulation_log_data:
            self.log("No data to export.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                                                title="Save AUV Log As...")
        if not filepath: return
        
        headers = ['Timestamp', 'AUV_ID', 'Speed_m/s', 'Position_X', 'Position_Y', 'Position_Z', 
                   'Battery_Percentage', 'AUV_State', 'Covered_Nodes_in_Radius']
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                sorted_data = sorted(self.simulation_log_data, key=lambda row: (row[0], row[1])) # Sort by time, then ID
                writer.writerows(sorted_data)
            self.log(f"Log successfully exported to {filepath}")
        except IOError as e:
            self.log(f"Error exporting file: {e}")

    def collect_auv_features(self):
        features = []
        auv_map = {}

        current_auv_list = list(self.auvs)
        if not self.sim_nodes or not current_auv_list:
            return None, None
        
        for i, auv in enumerate(current_auv_list):
            count = 0
            for node in self.sim_nodes:
                if auv_distance(auv.current_pos, node.pos) < auv.coverage_radius:
                    count += 1

            features.append([
                auv.current_pos[0],
                auv.current_pos[1],
                auv.current_pos[2],
                auv.speed,
                count 
            ])
            auv_map[i] = auv

        return np.array(features), auv_map

    def update_controllers_svm(self, p):
        self.logger.debug("[MC]: Waking up to update controllers...")
        X_features, auv_map = self.collect_auv_features()

        if X_features is None or X_features.shape[0] == 0:
            self.logger.debug("[MC]: AUVs/Nodes not ready. Skipping update.")
            return
        
        for auv in self.auvs:
            auv.is_lc = False
            auv.is_oc = False

        import numpy as np
        import pandas as pd

        def safe_transform(scaler, X):
            try:
                if hasattr(scaler, "feature_names_in_"):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=scaler.feature_names_in_)
                elif isinstance(X, pd.DataFrame):
                    X = X.values
                return scaler.transform(X)
            except Exception as e:
                self.logger.warning(f"[MC]: Scaler transform error: {e}")
                return X

        def safe_predict(model, X):
            try:
                if hasattr(model, "feature_names_in_"):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=model.feature_names_in_)
                elif isinstance(X, pd.DataFrame):
                    X = X.values
                return model.predict(X)
            except Exception as e:
                self.logger.warning(f"[MC]: Model predict error: {e}")
                return np.zeros(len(X))

        X_scaled_lc = safe_transform(self.lc_scaler, X_features)
        lc_predictions = safe_predict(self.lc_model, X_scaled_lc)

        selected_lc_ids = set()
        for i, pred in enumerate(lc_predictions):
            if pred == 1:
                auv = auv_map[i]
                auv.is_lc = True
                selected_lc_ids.add(auv.id)

        X_scaled_oc = safe_transform(self.oc_scaler, X_features)
        oc_predictions = safe_predict(self.oc_model, X_scaled_oc)

        selected_oc_id = None
        if np.sum(oc_predictions) > 0:
            try:
                oc_probabilities = self.oc_model.predict_proba(X_scaled_oc)[:, 1]
            except Exception:
                oc_probabilities = np.zeros_like(oc_predictions, dtype=float)
                self.logger.warning(
                    "[MC]: OC model has no predict_proba(). Using zeros.")

            best_oc_index = -1
            max_prob = -1
            for i, pred in enumerate(oc_predictions):
                if pred == 1 and oc_probabilities[i] > max_prob:
                    max_prob = oc_probabilities[i]
                    best_oc_index = i

            if best_oc_index != -1:
                auv = auv_map[best_oc_index]
                auv.is_oc = True
                auv.is_lc = True 
                selected_oc_id = auv.id
                if auv.id not in selected_lc_ids:
                    selected_lc_ids.add(auv.id)

        self.logger.info(f"[MC]: LCs updated: {selected_lc_ids or 'None'}")
        self.logger.info(f"[MC]: OC updated: {selected_oc_id or 'None'}")

    def run_mc_logic(self, p):
        if not self.models_loaded:
            self.log("[MC Sim]: Models not loaded. MC thread exiting.")
            return

        self.log("[MC Sim]: MC logic thread started (real-time loop).")
        try:
            while (len(self.auvs) == 0 or len(self.sim_nodes) == 0) and self.simulation_running.is_set():
                self.log("[MC Sim]: Waiting for nodes and AUVs...")
                time.sleep(0.5)

            while self.simulation_running.is_set():
                self.log("[MC Sim]: Loop start, sleeping...")
                for _ in range(SVM_UPDATE_PERIOD_S * 10):
                    if not self.simulation_running.is_set():
                        self.log("[MC Sim]: simulation_running is False, breaking sleep.")
                        break
                    time.sleep(0.1)
                
                if not self.simulation_running.is_set():
                    self.log("[MC Sim]: simulation_running is False, breaking loop.")
                    break
                    
                self.update_controllers_svm(p)
            
            self.log("[MC Sim]: Main loop exited.")

        except Exception as e:
            self.log(f"[MC Thread] Error: {e}")
            traceback.print_exc()
        finally:
            self.log("[MC Thread] Finished.")
            
    def run_auv_thread(self, p):
        self.log("[AUV Sim]: AUV simulation thread started.")
        total_auv_move_time = 0.0
        try:
            start_time = time.time()
            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng = np.random.default_rng(seed + 1)
            
            local_auv_list = []
            for i in range(int(p['NUM_AUVS'])):
                speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                route = generate_auv_route(p['AREA'], p['SURFACE_STATION_POS'], i)
                auv = AUV(i, speed, route, p['SURFACE_STATION_POS'], coverage_radius=p['AUV_COVERAGE_RADIUS'])
                local_auv_list.append(auv)
                self.log(f"[AUV {i}]: Created. Speed: {speed:.1f} m/s, Radius: {auv.coverage_radius} m")
            self.auvs = local_auv_list
            
            self.q.put({'type': 'setup_auv_plots', 'num_auvs': len(local_auv_list)})

            last_log_time = time.time()
            last_speed_change_time = {auv.id: time.time() for auv in local_auv_list}
            last_node_update_time = time.time()
            
            sim_duration = SIMULATION_DURATION_S
            if threading.current_thread().name.startswith("AUV-Sweep"):
                # Run for a shorter, fixed time during sweeps
                sim_duration = 300 # 5 minutes
                self.log(f"[AUV Sim]: Sweep mode detected. Running for {sim_duration}s.")

            while self.simulation_running.is_set() and (time.time() - start_time) < sim_duration:
                current_time = time.time()
                updated_indices = []
                num_moving_auvs = 0
                
                for idx, auv in enumerate(local_auv_list):
                    if current_time - last_speed_change_time[auv.id] > rng.uniform(15, 30):
                        auv.speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                        self.log(f"[AUV {auv.id}]: Speed changed to {auv.speed:.1f} m/s")
                        last_speed_change_time[auv.id] = current_time

                    relayed_ids, is_moving, collected_new = auv.update(AUV_UPDATE_INTERVAL_S, self.sim_nodes)
                    
                    if is_moving:
                        num_moving_auvs += 1

                    if relayed_ids:
                        self.log(f"[AUV {auv.id}]: Relayed data for nodes {relayed_ids}.")
                    if collected_new:
                        self.log(f"[AUV {auv.id}]: Collected new data from node(s).")
                    
                    updated_indices.append(idx)
                
                total_auv_move_time += (num_moving_auvs * AUV_UPDATE_INTERVAL_S)

                if current_time - last_log_time >= 1.0:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for auv in local_auv_list:
                        nodes_in_radius = [node.node_id for node in self.sim_nodes if distance(auv.current_pos, node.pos) <= auv.coverage_radius]
                        nodes_str = ', '.join(map(str, sorted(nodes_in_radius)))

                        pos = auv.current_pos
                        log_entry = [timestamp, auv.id, f"{auv.speed:.2f}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                                     f"{auv.battery:.1f}", auv.state, nodes_str]
                        self.simulation_log_data.append(log_entry)
                        
                        if auv.state == "Returning to Charge" and auv.battery < LOW_BATTERY_THRESHOLD:
                             pass
                        elif auv.state == "Patrolling" and auv.battery == 100.0 and (current_time - last_speed_change_time.get(auv.id, 0) < 1.5):
                             self.log(f"[AUV {auv.id}]: Recharged. Resuming patrol.")
                             
                    last_log_time = current_time
                
                if current_time - last_node_update_time >= 1.0:
                    for node in self.sim_nodes:
                        node.update_position(NODE_MAX_DRIFT_M)
                    new_node_positions = np.array([n.pos for n in self.sim_nodes])
                    self.q.put({'type': 'update_nodes', 'nodes_pos': new_node_positions})
                    last_node_update_time = current_time

                if updated_indices:
                    self.q.put({'type': 'plot_auvs', 'auv_indices': updated_indices})
                
                time.sleep(AUV_UPDATE_INTERVAL_S)
            
            if not self.simulation_running.is_set():
                self.log("[Sim]: Simulation stopped by user.")
            else:
                self.log(f"[Sim]: Simulation duration ({sim_duration}s) finished.")

        except Exception as e:
            self.log(f"[AUV Thread] Error: {e}")
            traceback.print_exc()
        finally:
            self.log("\n--- AUV FINAL STATS ---")
            if self.auvs:
                for auv in self.auvs:
                    self.log(f"--- AUV {auv.id} Report ---")
                    self.log(f"  Nodes Detected: {sorted(list(auv.covered_nodes)) or 'None'}")
                    relayed = sorted(list(auv.relayed_data_log.keys()))
                    self.log(f"  Data Relayed (IDs): {relayed or 'None'}")
            
            self.q.put({'type': 'auv_finished', 'data': {
                'total_auv_move_time': total_auv_move_time
            }})
            self.log("[AUV Thread] Finished.")
    
    # --- NEW: run_packet_routing_thread (Adapted from osar_gui.py) ---
    def run_packet_routing_thread(self, p):
        self.log("[Packet Sim]: Packet routing thread started.")
        seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
        rng = np.random.default_rng(seed + 2) # Use different seed
        
        ch_bw_khz = CH_BANDWIDTH / 1000.0
        start_center_freq = START_FREQ_KHZ + (ch_bw_khz / 2.0)
        channels_khz = start_center_freq + np.arange(p["M_CHANNELS"]) * ch_bw_khz
        
        if not self.sim_nodes:
            self.log("[Packet Sim]: No nodes to simulate.")
            return

        dest_pos = p['SURFACE_STATION_POS']
        src = max(self.sim_nodes, key=lambda n: n.depth)
        self.log(f"[Packet Sim]: Source: Node {src.node_id}, Depth={src.depth:.1f} m")
        
        packets_generated = int(p['NUM_PACKETS'])
        packets_delivered = 0
        total_data_bits = 0
        total_control_bits = 0
        total_end_to_end_delay = 0.0

        for i in range(packets_generated):
            if not self.simulation_running.is_set():
                self.log("[Packet Sim]: Simulation stopped.")
                break
                
            self.log(f"\n--- [Packet Sim] Routing Packet {i+1}/{packets_generated} ---")
            current, visited_pos = src, [src.pos]
            packet_delay, packet_delivered_flag = 0.0, False

            for hop_count in range(int(p['N_NODES']) + 5):
                if not self.simulation_running.is_set(): break
                
                # Update all node channel states for this hop attempt
                for node in self.sim_nodes: node.update_channels(p['P_BUSY'])

                # 1. Account for the "request" control packet for EVERY hop attempt.
                total_control_bits += CONTROL_PACKET_LP
                
                # --- MODIFIED: Call the new "smart" hop selection ---
                best_nbr, TD, _, _ = select_best_next_hop(current, self.sim_nodes, self.auvs, 
                                                          dest_pos, p['TX_RANGE'], channels_khz, 
                                                          LP, CH_BANDWIDTH, p['PT_LINEAR'], rng)
                # --- End of Change ---
                
                if best_nbr is None:
                    self.log(f"  ⚠️ Node {current.node_id} is stuck. Dropping packet.")
                    break # Failed hop, control packet was already counted
                
                # 2. If successful, account for "reply" and "data"
                total_control_bits += CONTROL_PACKET_LP
                total_data_bits += LP
                packet_delay += TD
                visited_pos.append(best_nbr.pos)
                self.log(f"  {current.node_id} → {best_nbr.node_id} | Hop Delay: {TD:.4f}s")
                
                # --- NEW: Check hop type ---
                if "AUV" in str(best_nbr.node_id):
                    # Packet was delivered to an AUV (LC or OC)
                    self.log(f"  ✅ Packet delivered to {best_nbr.node_id}.")
                    packets_delivered += 1
                    total_end_to_end_delay += packet_delay
                    packet_delivered_flag = True
                    break # Packet is delivered
                else:
                    # It's a normal node, continue hopping
                    current = best_nbr
                # --- End of Change ---

                # Check if this new node can reach the buoy
                if auv_distance(current.pos, dest_pos) < p['TX_RANGE'] or current.depth <= 5.0:
                    buoy_node = Node('Buoy', dest_pos, rng, channel_state=np.array([False]*M_SUBCARRIERS))
                    final_td, _, _ = compute_transmission_delay_paper(current, buoy_node, dest_pos, channels_khz[0], LP, CH_BANDWIDTH, p['PT_LINEAR'])
                    
                    total_data_bits += LP
                    total_control_bits += 2 * CONTROL_PACKET_LP # Final hop also has RTS/CTS
                    packet_delay += final_td
                    visited_pos.append(dest_pos)
                    self.log(f"  {current.node_id} → Buoy | Final Hop Delay: {final_td:.4f}s")
                    
                    packets_delivered += 1
                    total_end_to_end_delay += packet_delay
                    packet_delivered_flag = True
                    break
            
            self.q.put({'type': 'plot_route', 'route_pos': visited_pos, 'success': packet_delivered_flag})
            time.sleep(0.5) # Slow down packet routing for visualization

        # --- Send final results ---
        self.q.put({'type': 'osar_finished', 'data': {
            'params': p,
            'packets_generated': packets_generated,
            'packets_delivered': packets_delivered,
            'total_data_bits': total_data_bits,
            'total_control_bits': total_control_bits,
            'total_end_to_end_delay': total_end_to_end_delay
        }})
        self.log("[Packet Sim] Finished.")
            
    def on_closing(self):
        print("--- Application Closing ---")
        if self.is_simulation_running():
            print("Attempting to stop threads...")
            self.stop_simulation() # Signal threads to stop
            self.root.destroy() # Close window
        else:
            self.root.destroy()

if __name__ == "__main__":
    threading.current_thread().name = 'Main-GUI' # Name the main thread
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop() 
