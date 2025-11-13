# simulation.py (fixed v11 - Final)
# A graphical user interface for the OSAR simulation.
# --- Fixed: Sequential execution, Robust file/GUI logging, Sweep plots ---
# --- Added: Increased AUV speed, Run/Stop toggle button ---

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
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
    from auv2 import (AUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                      AUV_WAYPOINTS, AUV_COVERAGE_RADIUS, DEFAULT_AUV_RELAY_RADIUS,
                      distance as auv_distance)
except Exception as e:
    print(
        f"Warning: Could not import 'auv2' (auv2.py) from: {project_root}. AUV functionality disabled.\n{e}")
    AUV_AVAILABLE = False
    # Provide minimal placeholders so module can be imported; these will raise if used.

    def auv_distance(a, b):
        raise RuntimeError("AUV module not available")

    class AUV:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AUV module not available")

    def generate_auv_route(*args, **kwargs):
        raise RuntimeError("AUV module not available")
    
    # --- MODIFIED: Default AUV Speeds (if auv2.py is missing) ---
    AUV_MIN_SPEED = 10.0  # Was 0.1
    AUV_MAX_SPEED = 20.0 # Was 0.5
    # --- End of Change ---
    
    AUV_WAYPOINTS = 5
    AUV_COVERAGE_RADIUS = 20.0
    DEFAULT_AUV_RELAY_RADIUS = 50.0

# --- Use a publication-quality plot style ---
plt.style.use('seaborn-v0_8-whitegrid')

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

# ---------- Acoustic & Utility Functions ----------


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


def place_nodes_uniform(N, area, rng):
    side = int(np.ceil(N**(1/3)))
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    final_nodes = grid[:N]
    jitter = rng.uniform(-area / (4 * side), area /
                         (4 * side), final_nodes.shape)
    final_nodes = np.clip(final_nodes + jitter, 0, area)
    final_nodes[:, 2] = np.maximum(final_nodes[:, 2], EPS)
    return final_nodes


class Node:
    def __init__(self, node_id, pos, channel_state):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = sound_speed(self.depth)


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
        if isinstance(dest, Node) and dest.node_id == 'Buoy':
            proj_len = DiD
        else:
            proj_len = EPS
    else:
        proj_len = np.dot(vec_ij, vec_id) / DiD
    if proj_len <= EPS:
        proj_len = EPS
    N_hop = max(DiD / proj_len, 1.0)

    src_depth = src.depth if isinstance(
        src.depth, (int, float)) and np.isfinite(src.depth) else AREA
    dest_depth = 0.0 if isinstance(dest, Node) and dest.node_id == 'Buoy' else (
        dest.depth if isinstance(dest.depth, (int, float)) and np.isfinite(dest.depth) else AREA)

    depth_diff = max(src_depth - dest_depth, 0.0)
    mid_depth_m = max((src_depth + dest_depth) / 2.0, 0.0)
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
        return float('inf'), 0.0, 0.0
    TD = ((lp / r_ij_ch) + PD_ij) * N_hop
    return TD, r_ij_ch, snr_linear


def select_best_next_hop(src, nodes, dest_pos, tx_range, channels, lp, bw, pt):
    candidates = []
    src_pos = np.array(src.pos)
    dest_pos_target = np.array(dest_pos)
    for nbr in nodes:
        if nbr.node_id == src.node_id:
            continue
        nbr_pos = np.array(nbr.pos)
        if auv_distance(src_pos, nbr_pos) > tx_range:
            continue

        src_depth = src.depth if isinstance(
            src.depth, (int, float)) and np.isfinite(src.depth) else AREA
        nbr_depth = nbr.depth if isinstance(
            nbr.depth, (int, float)) and np.isfinite(nbr.depth) else AREA
        if nbr_depth >= src_depth:
            continue

        if np.dot(dest_pos_target - src_pos, nbr_pos - src_pos) <= 0:
            continue
        src_ch_state = np.array(src.channel_state, dtype=bool)
        nbr_ch_state = np.array(nbr.channel_state, dtype=bool)
        common_idle = np.where(~src_ch_state & ~nbr_ch_state)[0]
        if len(common_idle) == 0:
            continue
        best_td_for_nbr, best_ch_for_nbr, best_snr_for_nbr = float(
            'inf'), None, None
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, snr_lin = compute_transmission_delay_paper(
                src, nbr, dest_pos_target, f_center_khz, lp, bw, pt)
            if TD < best_td_for_nbr:
                best_td_for_nbr = TD
                best_ch_for_nbr = ch_idx
                best_snr_for_nbr = snr_lin
        if best_td_for_nbr != float('inf'):
            candidates.append(
                (nbr, best_td_for_nbr, best_ch_for_nbr, best_snr_for_nbr))
    if not candidates:
        return None, None, None, None
    return min(candidates, key=lambda x: x[1])


# ---------- NEW: QueueHandler for logging ----------
class QueueHandler(logging.Handler):
    """
    A custom logging handler that puts formatted log messages onto a queue.
    """
    def __init__(self, queue_instance):
        super().__init__()
        self.queue = queue_instance

    def emit(self, record):
        """
        Formats the log record and puts it on the queue
        in the existing {'type': 'log', ...} format.
        """
        msg = self.format(record)
        self.queue.put({'type': 'log', 'message': msg})

# ---------- Main GUI Application ----------
class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OSAR Underwater Routing Simulation")
        self.root.geometry("1300x850")
        self.sim_thread = None
        self.auv_thread = None
        self.mc_thread = None
        self.simulation_running = threading.Event()
        self.q = queue.Queue()  # This queue is now used for ALL messages (log + GUI)
        self.sim_nodes = []
        self.auvs = []
        self.sc = None
        self.route_plots = []
        self.surface_station_pos = np.array([AREA/2, AREA/2, 0.0])
        self.osar_results = None
        self.auv_results = None
        self.mc_finished = False
        
        # --- NEW: Setup Logger ---
        self.logger = None # Will be initialized in setup_logging
        self.setup_logging()
        # -------------------------

        self.lc_model = None
        self.lc_scaler = None
        self.oc_model = None
        self.oc_scaler = None
        # --- Load models only if ML_AVAILABLE ---
        self.models_loaded = False
        if ML_AVAILABLE:
            self.models_loaded = self.load_models(project_root)
        else:
            self.logger.warning("ML not available; models will not be loaded.")

        # AUV availability flag
        self.auv_available = AUV_AVAILABLE

        self.create_widgets()
        self.init_hover_functionality()
        self.process_queue()

    def setup_logging(self):
        """Configures the logger to send messages to the GUI queue and a file."""
        self.logger = logging.getLogger('OSAR_Sim')
        self.logger.setLevel(logging.DEBUG)  # Log everything from DEBUG level up

        # 1. File Handler (writes to simulation_log.txt)
        log_file_path = ""
        try:
            # script_dir is a global variable defined at the top of the file
            log_file_path = os.path.join(script_dir, 'simulation_log.txt')
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(threadName)-18s] [%(levelname)-5s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"CRITICAL: Failed to create log file at {log_file_path}. Error: {e}")

        # 2. Queue Handler (writes to the GUI)
        gui_formatter = logging.Formatter(
            '[%(threadName)-18s] %(message)s'
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
            # If joblib isn't available this will raise
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
        controls_frame = ttk.LabelFrame(
            left_panel, text="Simulation Parameters", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.params = {}
        param_list = {
            "N_NODES": ("Number of Nodes:", N_NODES),
            "AREA": ("Area Size (m):", AREA),
            "TX_RANGE": ("TX Range (m):", TX_RANGE),
            "P_BUSY": ("P(Busy Channel):", P_BUSY),
            "PT_DB": ("Transmit Power (dB):", DEFAULT_PT_DB),
            "M_CHANNELS": ("Num Channels:", DEFAULT_M_CHANNELS),
            "NUM_PACKETS": ("Num Packets to Send:", DEFAULT_NUM_PACKETS),
            "NUM_AUVS": ("Num AUVs:", DEFAULT_NUM_AUVS),
            "AUV_RELAY_RADIUS": ("AUV Relay Radius(m):", DEFAULT_AUV_RELAY_RADIUS),
            "NUM_LCS_TO_SELECT": ("(Fallback) Num LCs:", DEFAULT_NUM_LCS_TO_SELECT),
            "SEED": ("Random Seed (0=random):", 0)
        }
        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(
                row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(
                row=i, column=1, sticky=tk.W, pady=2)

        # --- MODIFIED: Renamed button and changed command ---
        self.toggle_run_button = ttk.Button(
            left_panel, text="Run Simulation", command=self.toggle_simulation)
        self.toggle_run_button.pack(fill=tk.X, pady=5)
        # --- End of Change ---
        
        self.sweep_button = ttk.Button(
            left_panel, text="Run AUV Sweep Plot", command=self.start_sweep_experiment)
        self.sweep_button.pack(fill=tk.X, pady=5)

        # Disable run button if essential components are missing
        if not self.models_loaded or not self.auv_available:
            self.toggle_run_button.config(
                text="Models/AUV Not Found", state=tk.DISABLED)
            self.sweep_button.config(
                text="Models/AUV Not Found", state=tk.DISABLED)

        log_frame = ttk.LabelFrame(left_panel, text="Live Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=40, height=20, font=("Consolas", 9)) # Use monospaced font
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
                "Error: AUV components (auv2.py) not found in project root. AUV simulation disabled.")

    def init_hover_functionality(self):
        try:
            self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                         bbox=dict(boxstyle="round",
                                                   fc="yellow", alpha=0.8),
                                         arrowprops=dict(arrowstyle="->"))
            self.annot.set_visible(False)
            self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        except Exception as e:
            self.logger.warning(f"Could not init hover functionality: {e}")


    def update_annot(self, ind):
        if not self.sim_nodes or not hasattr(self.sc, 'get_offsets'):
            return
        try:
            node_index = ind["ind"][0]
            if node_index >= len(self.sim_nodes):
                return
            node = self.sim_nodes[node_index]
            offsets = self.sc.get_offsets()
            if node_index >= len(offsets):
                return
            pos = offsets[node_index]
            self.annot.xy = pos
            text = (f"ID: {node.node_id}\nPos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                    f"Depth: {node.depth:.1f} m")
            self.annot.set_text(text)
        except IndexError:
            self.annot.set_visible(False)
        except Exception as e:
            self.logger.warning(f"Error in update_annot: {e}")

    def hover(self, event):
        vis = self.annot.get_visible()
        try:
            if event.inaxes == self.ax and self.sc and hasattr(self.sc, 'contains'):
                cont, ind = self.sc.contains(event)
                if cont:
                    self.update_annot(ind)
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                elif vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
        except Exception as e:
             self.logger.warning(f"Error in hover event: {e}")


    # --- OLD self.log() method is REMOVED ---

    def run_sequential_simulation(self, p):
        """
        Orchestrator thread to run simulation phases sequentially.
        Phase 1: OSAR
        Phase 2: AUV
        Phase 3: MC (runs once at the end)
        """
        try:
            # --- 1. Run OSAR Simulation ---
            self.logger.info("--- [1/3] Starting OSAR Simulation (Static Routing) ---")
            self.sim_thread = threading.Thread(
                target=self.run_simulation_thread, args=(p,), daemon=True, name='OSAR-Thread')
            self.sim_thread.start()
            self.sim_thread.join()  # Wait for OSAR to finish
            
            # Check if user stopped during OSAR
            if not self.simulation_running.is_set():
                 self.logger.warning("Simulation stopped during OSAR phase.")
                 return

            self.logger.info("--- OSAR Simulation Finished ---")
            time.sleep(0.5)

            # --- 2. Run AUV Simulation ---
            if not self.sim_nodes:
                self.logger.warning("--- Error: OSAR failed to create nodes. AUV/MC sim skipped. ---")
                return  # Exit orchestrator thread

            self.logger.info("\n--- [2/3] Starting AUV Simulation (Mobile Relays) ---")
            self.auv_thread = threading.Thread(
                target=self.run_auv_thread, args=(p,), daemon=True, name='AUV-Thread')
            self.auv_thread.start()
            self.auv_thread.join()  # Wait for AUVs to finish
            
            # Check if user stopped during AUV
            if not self.simulation_running.is_set():
                 self.logger.warning("Simulation stopped during AUV phase.")
                 return

            self.logger.info("--- AUV Simulation Finished ---")
            
            # --- 3. Run MC (Post-Run Classification) ---
            self.logger.info("\n--- [3/3] Starting MC Classification (Post-Run) ---")
            self.mc_thread = threading.Thread(
                target=self.run_mc_logic, args=(p,), daemon=True, name='MC-Thread')
            self.mc_thread.start()
            self.mc_thread.join() # Wait for MC to finish its single run
            
            self.logger.info("--- MC Classification Finished ---")
            self.logger.info("\n--- All Simulations Finished ---")

        except Exception as e:
            self.logger.critical(f"[Orchestrator] Error: {e}", exc_info=True)
        finally:
            self.simulation_running.clear() # Clear at the very end
            self.logger.info("[Orchestrator Thread] Finished.")
            self.q.put({'type': 'force_enable_buttons'}) # Final failsafe
    
    # --- NEW: Helper function to check if sim is running ---
    def is_simulation_running(self):
        orch_alive = hasattr(
            self, 'orchestrator_thread') and self.orchestrator_thread and self.orchestrator_thread.is_alive()
        sweep_alive = hasattr(
            self, 'sweep_thread') and self.sweep_thread and self.sweep_thread.is_alive()
        return orch_alive or sweep_alive

    # --- NEW: Toggle function for the main button ---
    def toggle_simulation(self):
        if self.is_simulation_running():
            self.stop_simulation()
        else:
            self.start_simulation()

    # --- NEW: Stop function ---
    def stop_simulation(self):
        if self.is_simulation_running():
            self.logger.warning("--- SIMULATION STOP REQUESTED BY USER ---")
            self.simulation_running.clear()  # Signal all threads to stop
            
            # Update buttons to show stopping state
            self.toggle_run_button.config(text="Stopping...", state=tk.DISABLED)
            self.sweep_button.config(text="Stopping...", state=tk.DISABLED)

    def start_simulation(self):
        if self.is_simulation_running():
            self.logger.warning("Simulation is already running.")
            return

        self.simulation_running.clear()
        time.sleep(0.05)

        self.log_text.delete('1.0', tk.END) # Clear GUI log
        self.logger.info("--- New Simulation Run Started ---")

        # Clear plots
        try:
            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.init_hover_functionality()
            self.route_plots.clear()
            for auv in self.auvs:
                if getattr(auv, 'path_artist', None):
                    try: auv.path_artist.remove()
                    except ValueError: pass
                if getattr(auv, 'marker_artist', None):
                    try: auv.marker_artist.remove()
                    except ValueError: pass
        except Exception as e:
            self.logger.warning(f"Error clearing plot: {e}")

        self.auvs.clear()
        self.sim_nodes.clear()
        self.osar_results = None
        self.auv_results = None
        self.mc_finished = False

        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            p["N_NODES"] = int(p["N_NODES"])
            p["M_CHANNELS"] = int(p["M_CHANNELS"])
            p["NUM_PACKETS"] = int(p["NUM_PACKETS"])
            p["NUM_AUVS"] = int(p["NUM_AUVS"])
            p["NUM_LCS_TO_SELECT"] = int(p["NUM_LCS_TO_SELECT"])
            p["SEED"] = int(p["SEED"])
            p["AUV_RELAY_RADIUS"] = float(p["AUV_RELAY_RADIUS"])
            p["PT_LINEAR"] = 10**(p["PT_DB"] / 10.0)

            self.surface_station_pos = np.array(
                [p['AREA']/2, p['AREA']/2, 0.0])
            p['SURFACE_STATION_POS'] = self.surface_station_pos

            if not self.auv_available:
                self.logger.error("Cannot start: AUV components missing.")
                return
            if not self.models_loaded:
                self.logger.error("Cannot start: Models not loaded.")
                return
            if p["N_NODES"] == 0:
                self.logger.error("Error: 'Number of Nodes' must be greater than 0.")
                return

            self.simulation_running.set()  # Set the flag for the new run

            self.orchestrator_thread = threading.Thread(
                target=self.run_sequential_simulation, args=(p,), daemon=True, name='Orchestrator')
            self.orchestrator_thread.start()

            # --- MODIFIED: Update button text/state ---
            self.toggle_run_button.config(text="Stop Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.DISABLED) 
            # --- End of Change ---

        except ValueError:
            self.logger.error("Error: Please enter valid numbers for all parameters.")
            self.run_button.config(state=tk.NORMAL)
        except Exception as e:
            self.logger.critical(f"Error starting simulation: {e}", exc_info=True)
            self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.NORMAL)
            self.simulation_running.clear()  # Ensure flag is clear on error

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
        packets_generated = p.get('NUM_PACKETS', DEFAULT_NUM_PACKETS)
        packets_delivered = self.osar_results['packets_delivered']
        total_data_bits = self.osar_results['total_data_bits']
        total_control_bits = self.osar_results['total_control_bits']
        total_end_to_end_delay = self.osar_results['total_end_to_end_delay']
        total_auv_move_time = self.auv_results.get('total_auv_move_time', 0.0)

        pdr = packets_delivered / packets_generated if packets_generated > 0 else 0
        if log_to_gui:
            self.logger.info(
                f"Packet Delivery Ratio (PDR): {pdr:.2%} ({packets_delivered}/{packets_generated})")

        total_bits = total_data_bits + total_control_bits
        overhead_ratio = total_control_bits / total_bits if total_bits > 0 else 0
        if log_to_gui:
            self.logger.info(f"Overhead Ratio (RoR) (by bytes): {overhead_ratio:.4f}")

        avg_delay = total_end_to_end_delay / \
            packets_delivered if packets_delivered > 0 else float('inf')
        if log_to_gui:
            self.logger.info(
                f"Average End-to-End Delay (for delivered packets): {avg_delay:.4f} s")

        energy_data = total_data_bits * E_BIT_TX
        energy_control = total_control_bits * E_BIT_TX
        energy_movement = total_auv_move_time * E_AUV_MOVE_PER_S
        total_energy = energy_data + energy_control + energy_movement
        energy_management = energy_control + energy_movement
        ecr = energy_management / total_energy if total_energy > 0 else 0

        if log_to_gui:
            self.logger.info(f"Energy Cost Ratio (ECR): {ecr:.4f}")
            self.logger.info(
                f" (Energy_Data: {energy_data:,.0f}, Energy_Control: {energy_control:,.0f}, Energy_Movement: {energy_movement:,.0f})")
            # Re-enable buttons only when called for the GUI
            self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.NORMAL)

        return pdr, overhead_ratio, ecr  # Return the calculated values

    def start_sweep_experiment(self):
        if self.is_simulation_running():
            self.logger.warning("Simulation is already running.")
            return

        self.log_text.delete('1.0', tk.END) # Clear GUI log
        self.logger.info("--- New AUV Sweep Run Started ---")

        # --- MODIFIED: Update button text/state ---
        self.toggle_run_button.config(text="Stop Sweep", state=tk.NORMAL)
        self.sweep_button.config(state=tk.DISABLED)
        # --- End of Change ---

        self.sweep_thread = threading.Thread(
            target=self.run_sweep_thread, daemon=True, name='Sweep-Orchestrator')
        self.sweep_thread.start()

    def run_sweep_thread(self):
        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            max_auvs = int(p.get('NUM_AUVS', DEFAULT_NUM_AUVS))
            if max_auvs <= 0:
                self.logger.error("Cannot run sweep with 0 AUVs. Set 'Num AUVs' > 0.")
                self.q.put({'type': 'force_enable_buttons'})
                return
            
            sweep_auv_counts = list(range(1, max_auvs + 1))

            self.logger.info(f"--- Starting AUV Sweep (1 to {max_auvs} AUVs) ---")

            experiment_results = []
            
            self.simulation_running.set()  # Set the flag for the *entire* sweep

            for auv_count in sweep_auv_counts:
                # Check if user stopped the sweep
                if not self.simulation_running.is_set():
                    self.logger.warning(f"AUV Sweep cancelled by user at {auv_count} AUVs.")
                    break

                if not ML_AVAILABLE or not AUV_AVAILABLE:
                    self.logger.error("Sweep cancelled: ML or AUV module missing.")
                    break  # Safety check

                self.logger.info(f"\n--- EXPERIMENT: Running for {auv_count} AUV(s) ---")

                current_p = p.copy()
                current_p['NUM_AUVS'] = auv_count
                self.params['NUM_AUVS'].set(str(auv_count)) # Update GUI

                self.osar_results = None
                self.auv_results = None
                self.mc_finished = False
                self.auvs.clear()
                self.sim_nodes.clear()
                self.q.put({'type': 'clear_plot'}) # Request main thread to clear plot
                time.sleep(0.1) # Give queue time to process

                # --- 1. Run OSAR ---
                self.logger.debug(f"[Sweep]: Running OSAR for {auv_count} AUVs...")
                self.sim_thread = threading.Thread(
                    target=self.run_simulation_thread, args=(current_p,), daemon=True, name=f'OSAR-Sweep-{auv_count}')
                self.sim_thread.start()
                self.sim_thread.join()  # Wait for OSAR to finish
                if not self.simulation_running.is_set(): continue

                # --- 2. Run AUV ---
                self.logger.debug(f"[Sweep]: Running AUV for {auv_count} AUVs...")
                self.auv_thread = threading.Thread(
                    target=self.run_auv_thread, args=(current_p,), daemon=True, name=f'AUV-Sweep-{auv_count}')
                self.auv_thread.start()
                self.auv_thread.join()  # Wait for AUVs to finish
                if not self.simulation_running.is_set(): continue

                # --- 3. Run MC ---
                self.logger.debug(f"[Sweep]: Running MC for {auv_count} AUVs...")
                self.mc_thread = threading.Thread(
                    target=self.run_mc_logic, args=(current_p,), daemon=True, name=f'MC-Sweep-{auv_count}')
                self.mc_thread.start()
                self.mc_thread.join()
                
                self.logger.info(f"[Sweep]: Run for {auv_count} AUVs finished. Collecting results...")

                # --- 4. Collect Results ---
                wait_start_time = time.time()
                while (not self.osar_results or not self.auv_results) and (time.time() - wait_start_time < 10.0):
                    time.sleep(0.2)  # Poll for results
                
                if not self.osar_results or not self.auv_results:
                    self.logger.error(f"--- EXPERIMENT ERROR: Timeout waiting for results for {auv_count} AUVs ---")
                    continue  # Skip this run

                pdr, ror, ecr = self.calculate_final_metrics(log_to_gui=False)

                if pdr is not None and ror is not None and ecr is not None:
                    experiment_results.append(
                        {'auvs': auv_count, 'pdr': pdr, 'ror': ror, 'ecr': ecr})
                    self.logger.info(f"  -> Results: PDR={pdr:.3f}, RoR={ror:.3f}, ECR={ecr:.3f}")
                else:
                    self.logger.error(f"--- EXPERIMENT ERROR: Failed to calculate metrics for {auv_count} AUVs ---")

            # --- Experiment Finished ---
            self.logger.info("\n--- EXPERIMENT SWEEP FINISHED ---")
            if experiment_results:
                self.q.put({'type': 'plot_experiment_results',
                            'data': experiment_results})
            else:
                self.logger.warning("No results to plot.")
                self.q.put({'type': 'force_enable_buttons'}) # Re-enable buttons

        except Exception as e:
            self.logger.critical(f"[Sweep Thread] FATAL ERROR: {e}", exc_info=True)
        finally:
            self.simulation_running.clear()
            self.q.put({'type': 'force_enable_buttons'})
            self.logger.info("[Sweep Thread] Finished.")

    def process_queue(self):
        try:
            # Handle all messages in the queue
            while True:
                data = self.q.get_nowait()

                # --- NEW: Simplified Log Handler ---
                if data['type'] == 'log':
                    self.log_text.insert(tk.END, data['message'] + "\n")
                    self.log_text.see(tk.END)

                elif data['type'] == 'plot_nodes':
                    self.fig.clear()
                    self.ax = self.fig.add_subplot(111, projection="3d")
                    self.init_hover_functionality()
                    
                    self.route_plots.clear()
                    for auv in self.auvs:
                        if getattr(auv, 'path_artist', None):
                            try: auv.path_artist.remove()
                            except ValueError: pass
                        if getattr(auv, 'marker_artist', None):
                            try: auv.marker_artist.remove()
                            except ValueError: pass

                    self.sc = self.ax.scatter(data['pos'][:, 0], data['pos'][:, 1], data['pos'][:, 2],
                                             c=data['depths'], cmap="viridis_r", s=50, alpha=0.9,
                                             edgecolors='black', linewidth=0.6, label="Sensor Nodes")
                    self.ax.scatter(data['src'][0], data['src'][1], data['src'][2], c='lime', s=180, marker="o",
                                    label="Source Node", edgecolors='black', linewidth=1.2, depthshade=False)
                    self.ax.scatter(self.surface_station_pos[0], self.surface_station_pos[1], self.surface_station_pos[2],
                                    c="red", s=180, marker="^",
                                    label="Surface Buoy (MC)", edgecolors='black', linewidth=1.2, depthshade=False)
                    self.ax.scatter([], [], [], c='magenta',
                                    s=150, marker='x', label="AUV (Normal)")
                    self.ax.scatter([], [], [], c='cyan', s=250,
                                    marker='X', label="AUV (LC)")
                    self.ax.scatter([], [], [], c='red', s=300,
                                    marker='X', label="AUV (OC)")

                    self.ax.set_xlabel("X (m)", fontweight='bold')
                    self.ax.set_ylabel("Y (m)", fontweight='bold')
                    self.ax.set_zlabel("Depth (m)", fontweight='bold')
                    self.ax.set_title(
                        "OSAR Underwater Network Topology", fontsize=14, fontweight='bold')
                    self.ax.invert_zaxis()
                    self.ax.set_xlim(0, AREA)
                    self.ax.set_ylim(0, AREA)
                    self.ax.set_zlim(AREA, 0)
                    self.ax.legend(loc='upper left')
                    self.ax.view_init(elev=25, azim=-75)
                    self.fig.tight_layout()
                    self.fig.canvas.draw_idle()

                elif data['type'] == 'plot_route':
                    if not hasattr(self.ax, 'plot'): continue # Guard if axes are 2D
                    route_pos = np.array(data['route_pos'])
                    color = "mediumblue" if data['success'] else 'crimson'
                    style = '-' if data['success'] else '--'
                    route_plot_list = self.ax.plot(route_pos[:, 0], route_pos[:, 1], route_pos[:, 2],
                                                  color=color, linewidth=2.5, marker='o', linestyle=style,
                                                  markersize=6, markerfacecolor='orange', zorder=5)
                    if route_plot_list:
                        self.route_plots.append(route_plot_list[0])
                    self.fig.canvas.draw_idle()

                elif data['type'] == 'plot_auvs':
                    if not hasattr(self.ax, 'plot'): continue # Guard if axes are 2D
                    for auv in self.auvs:
                        if getattr(auv, 'path_artist', None):
                            try: auv.path_artist.remove()
                            except ValueError: pass
                        if getattr(auv, 'marker_artist', None):
                            try: auv.marker_artist.remove()
                            except ValueError: pass
                    
                    for auv in self.auvs:
                        pos = auv.current_pos
                        path_data = np.array(auv.traveled_path)
                        if auv.is_oc:
                            color = 'red'; marker = 'X'; size = 300; path_style = '-'; path_width = 2.0
                        elif auv.is_lc:
                            color = 'cyan'; marker = 'X'; size = 250; path_style = '--'; path_width = 1.5
                        else:
                            color = 'magenta'; marker = 'x'; size = 150; path_style = ':'; path_width = 1.0

                        if path_data.shape[0] > 1:
                            path_artist_list = self.ax.plot(path_data[:, 0], path_data[:, 1], path_data[:, 2],
                                                           color=color, linestyle=path_style, linewidth=path_width, zorder=9)
                            if path_artist_list:
                                auv.path_artist = path_artist_list[0]

                        marker_artist = self.ax.scatter(pos[0], pos[1], pos[2],
                                                       c=color, marker=marker, s=size,
                                                       depthshade=False, zorder=10, edgecolors='black', linewidth=0.5)
                        auv.marker_artist = marker_artist
                    self.fig.canvas.draw_idle()

                elif data['type'] == 'clear_plot':
                    self.fig.clear() # Clear 2D or 3D
                    self.ax = self.fig.add_subplot(111, projection="3d") # Put back 3D
                    self.ax.set_title("Waiting for next run...", fontsize=12)
                    self.init_hover_functionality() # Re-link hover
                    self.fig.canvas.draw_idle()

                elif data['type'] == 'plot_experiment_results':
                    results = data['data']
                    self.fig.clear() # Clear the entire figure
                    
                    auv_counts = [r['auvs'] for r in results]
                    pdr_data = [r['pdr'] for r in results]
                    ror_data = [r['ror'] for r in results]
                    ecr_data = [r['ecr'] for r in results]

                    ax1 = self.fig.add_subplot(311)
                    ax1.plot(auv_counts, pdr_data, marker='o', color='blue', label='PDR')
                    ax1.set_title('Simulation Metrics vs. Number of AUVs', fontsize=14, fontweight='bold')
                    ax1.set_ylabel("Packet Delivery Ratio (PDR)", fontweight='bold')
                    if pdr_data:
                        ax1.set_ylim(0, max(1.1, max(pdr_data)*1.1))
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
                    
                    self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
                    self.sweep_button.config(state=tk.NORMAL)

                elif data['type'] == 'osar_finished':
                    self.osar_results = data['data']
                    self.logger.debug("[Queue]: OSAR results received.")
                    if self.auv_results and not self.is_simulation_running():
                        self.calculate_final_metrics(log_to_gui=True)
                        self.osar_results = None 
                        self.auv_results = None
                        
                elif data['type'] == 'auv_finished':
                    self.auv_results = data['data']
                    self.logger.debug("[Queue]: AUV results received.")
                    if self.osar_results and not self.is_simulation_running():
                        self.calculate_final_metrics(log_to_gui=True)
                        self.osar_results = None
                        self.auv_results = None

                elif data['type'] == 'mc_finished':
                    self.mc_finished = True
                    self.logger.debug("[MC Sim]: Finished.")
                
                elif data['type'] == 'force_enable_buttons':
                    self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
                    self.sweep_button.config(text="Run AUV Sweep Plot", state=tk.NORMAL)

        except queue.Empty:
            pass  # No new messages
        except Exception as e:
            print(f"CRITICAL ERROR in process_queue: {e}")
            traceback.print_exc()
            self.simulation_running.clear()
            self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
            self.sweep_button.config(state=tk.NORMAL)

        # --- Rescheduling logic ---
        # Check if any main orchestrator thread is alive
        if self.is_simulation_running():
            self.root.after(100, self.process_queue)
        else:
            # All threads are dead.
            if self.toggle_run_button['state'] != tk.NORMAL:
                
                if self.osar_results and self.auv_results:
                    self.logger.info("...Calculating final metrics (cleanup)...")
                    self.calculate_final_metrics(log_to_gui=True)
                else:
                    self.logger.info("Simulation ended or was stopped.")
                
                # Final cleanup to re-enable buttons
                self.toggle_run_button.config(text="Run Simulation", state=tk.NORMAL)
                self.sweep_button.config(text="Run AUV Sweep Plot", state=tk.NORMAL)
                self.osar_results = None 
                self.auv_results = None

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

        # --- Handle feature/scaler compatibility ---
        import numpy as np
        import pandas as pd

        def safe_transform(scaler, X):
            """Ensure X matches the scaler’s expected input type."""
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
            """Ensure X matches the model’s expected input type."""
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

        # --- LC prediction ---
        X_scaled_lc = safe_transform(self.lc_scaler, X_features)
        lc_predictions = safe_predict(self.lc_model, X_scaled_lc)

        selected_lc_ids = set()
        for i, pred in enumerate(lc_predictions):
            if pred == 1:
                auv = auv_map[i]
                auv.is_lc = True
                selected_lc_ids.add(auv.id)

        # --- OC prediction ---
        X_scaled_oc = safe_transform(self.oc_scaler, X_features)
        oc_predictions = safe_predict(self.oc_model, X_scaled_oc)

        selected_oc_id = None
        if np.sum(oc_predictions) > 0:
            try:
                oc_probabilities = self.oc_model.predict_proba(X_scaled_oc)[
                    :, 1]
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

        # --- Logging ---
        self.logger.info(f"[MC]: LCs updated: {selected_lc_ids or 'None'}")
        self.logger.info(f"[MC]: OC updated: {selected_oc_id or 'None'}")

        # --- Refresh visualization ---
        self.q.put(
            {'type': 'plot_auvs', 'auv_indices': list(range(len(self.auvs)))})

    def run_mc_logic(self, p):
        # --- MODIFIED: This thread now runs only ONCE ---
        if not self.models_loaded:
            self.logger.error("[MC Sim]: Models not loaded. MC thread exiting.")
            self.q.put({'type': 'mc_finished', 'data': {
                        'reason': 'models_missing'}})
            return

        self.logger.info("[MC Sim]: MC logic thread started (single run).")
        try:
            # Wait for nodes and AUVs to be ready (they should be, but good to check)
            wait_start = time.time()
            while (len(self.auvs) == 0 or len(self.sim_nodes) == 0) and (time.time() - wait_start < 5.0):
                self.logger.debug("[MC Sim]: Waiting for nodes and AUVs...")
                time.sleep(0.5)
            
            if len(self.auvs) == 0 or len(self.sim_nodes) == 0:
                self.logger.error("[MC Sim]: Nodes or AUVs not found. Exiting.")
                self.q.put({'type': 'mc_finished', 'data': {'reason': 'nodes_or_auvs_missing'}})
                return

            # --- Run the classification ONCE ---
            self.logger.info("[MC Sim]: Classifying AUVs at their final positions.")
            self.update_controllers_svm(p)
            # --- End of single run ---

        except Exception as e:
            self.logger.critical(f"[MC Thread] Error: {e}", exc_info=True)
        finally:
            self.q.put({'type': 'mc_finished', 'data': {}})
            self.logger.info("[MC Thread] Finished.")
        # --- End of Change ---

    def run_auv_thread(self, p):
        self.logger.info("[AUV Sim]: AUV simulation thread started.")
        local_auv_list = []
        surface_station_pos = p['SURFACE_STATION_POS']
        auv_relay_radius = p['AUV_RELAY_RADIUS']
        total_auv_move_time = 0.0

        try:
            while len(self.sim_nodes) == 0 and self.simulation_running.is_set():
                self.logger.debug("[AUV Sim]: Waiting for sim_nodes...")
                time.sleep(0.1)
            if not self.simulation_running.is_set():
                self.logger.info("[AUV Sim]: Exiting early, sim not running.")
                return

            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng = np.random.default_rng(seed + 1)
            temp_auv_list = []
            for i in range(int(p['NUM_AUVS'])):
                # --- MODIFIED: Use new AUV speed constants ---
                speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                # --- End of Change ---
                route = generate_auv_route(p['AREA'], AUV_WAYPOINTS, rng)
                auv = AUV(i, speed, route, surface_station_pos,
                          coverage_radius=AUV_COVERAGE_RADIUS,
                          relay_radius=auv_relay_radius)
                temp_auv_list.append(auv)
                self.logger.info(
                    f"[AUV {i}]: Created. Speed: {speed:.1f} m/s, Relay Radius: {auv_relay_radius}m")

            self.auvs = temp_auv_list
            local_auv_list = temp_auv_list

            last_plot_time = 0
            while self.simulation_running.is_set():
                auv_indices_updated = []
                all_auvs_finished_route = True
                num_moving_auvs = 0

                if not local_auv_list:
                    self.logger.info("[AUV Sim]: No AUVs to run. Exiting loop.")
                    break

                for idx, auv in enumerate(local_auv_list):
                    if auv.target_waypoint_idx < len(auv.route):
                        newly_covered_node_id, relayed_ids, is_moving, collected_new = auv.update(
                            AUV_UPDATE_INTERVAL_S, self.sim_nodes
                        )
                        if collected_new:
                            self.logger.info(
                                f"[AUV {auv.id}]: Collected data from Node {newly_covered_node_id}")
                        if relayed_ids is not None:
                            self.logger.info(
                                f"[AUV {auv.id}]: Relayed data for nodes {relayed_ids} near surface.")

                        auv_indices_updated.append(idx)

                        if is_moving:
                            all_auvs_finished_route = False
                            num_moving_auvs += 1
                
                if all_auvs_finished_route:
                    self.logger.info("[AUV Sim]: All AUVs have completed their routes.")
                    break # Exit the main while loop

                total_auv_move_time += (num_moving_auvs *
                                        AUV_UPDATE_INTERVAL_S)

                current_time = time.time()
                if (current_time - last_plot_time > 0.2):
                    self.q.put({'type': 'plot_auvs', 'auv_indices': list(
                        range(len(local_auv_list)))})
                    last_plot_time = current_time

                time.sleep(AUV_UPDATE_INTERVAL_S)
            
            self.logger.info("[AUV Sim]: Main loop exited.")

        except Exception as e:
            self.logger.critical(f"[AUV Thread] Error: {e}", exc_info=True)
        finally:
            final_auv_report = {}
            if local_auv_list:
                self.logger.info("\n--- AUV FINAL STATS ---")
                total_unique_relayed = set()
                for auv in local_auv_list:
                    report_key = f"--- AUV {auv.id} Report ---"
                    report_lines = [
                        f"  Speed: {auv.speed:.2f} m/s",
                        f"  Nodes Visited: {sorted(list(auv.covered_nodes)) or 'None'}",
                        f"  Data Relayed (Node IDs): {sorted(list(auv.relayed_data_log.keys())) or 'None'}"
                    ]
                    final_auv_report[report_key] = report_lines
                    
                    self.logger.info(report_key)
                    for line in report_lines:
                        self.logger.info(line)
                        
                    total_unique_relayed.update(auv.relayed_data_log.keys())

                self.logger.info(
                    f"\nTotal Unique Nodes Relayed by all AUVs: {len(total_unique_relayed)}")

            self.q.put({'type': 'auv_finished', 'data': {
                'total_auv_move_time': total_auv_move_time,
                'auv_reports': final_auv_report
            }})
            self.logger.info("[AUV Thread] Finished.")

    def run_simulation_thread(self, p):
        seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
        rng = np.random.default_rng(seed)
        self.logger.info(f"OSAR thread started with seed: {seed}")
        surface_station_pos = p['SURFACE_STATION_POS']

        packets_generated = 0
        packets_delivered = 0
        total_data_bits = 0
        total_control_bits = 0
        total_end_to_end_delay = 0.0

        try:
            ch_bw_khz = CH_BANDWIDTH / 1000.0
            start_center_freq = START_FREQ_KHZ + (ch_bw_khz / 2.0)
            channels_khz = start_center_freq + np.arange(int(p["M_CHANNELS"]))

            if p['N_NODES'] == 0:
                self.logger.warning("No nodes to simulate.")
                return

            positions = place_nodes_uniform(int(p['N_NODES']), p['AREA'], rng)
            nodes = [Node(i, positions[i], rng.random(M_SUBCARRIERS)
                          < p['P_BUSY']) for i in range(len(positions))]
            self.sim_nodes = nodes # <-- IMPORTANT: This unblocks other threads
            self.logger.info(f"Created {len(nodes)} sensor nodes.")

            dest_pos = surface_station_pos

            if not nodes:
                self.logger.error("No sensor nodes created.")
                return

            src = max(nodes, key=lambda n: n.depth if isinstance(
                n.depth, (int, float)) and np.isfinite(n.depth) else -float('inf'))

            if src is None:
                self.logger.error("Could not determine source node.")
                return

            self.logger.info(f"Source: Node {src.node_id}, Depth={src.depth:.1f} m")
            self.q.put({'type': 'plot_nodes', 'pos': np.array([n.pos for n in nodes]),
                        'depths': [n.depth for n in nodes], 'src': src.pos, 'dest': dest_pos})
            time.sleep(0.5)

            packets_generated = int(p['NUM_PACKETS'])
            self.logger.info(f"Starting to route {packets_generated} packets...")

            for i in range(packets_generated):
                if not self.simulation_running.is_set():
                    self.logger.info("OSAR routing cancelled.")
                    break
                self.logger.info(f"--- [OSAR] Routing Packet {i+1}/{packets_generated} ---")
                current, visited_pos = src, [src.pos]
                packet_delay, packet_delivered_flag = 0.0, False
                max_hops = int(p['N_NODES'] * 1.5) + 5
                
                for hop_count in range(max_hops):
                    if not self.simulation_running.is_set():
                        break
                    for node in nodes:
                        node.channel_state = rng.random(
                            M_SUBCARRIERS) < p['P_BUSY']
                    total_control_bits += CONTROL_PACKET_LP
                    best_nbr, TD, _, _ = select_best_next_hop(
                        current, nodes, dest_pos, p['TX_RANGE'], channels_khz, LP, CH_BANDWIDTH, p['PT_LINEAR'])
                    
                    if best_nbr is None:
                        self.logger.warning(f"  Node {current.node_id} is stuck. Dropping packet.")
                        break # Break hop_count loop
                        
                    total_control_bits += CONTROL_PACKET_LP
                    total_data_bits += LP
                    packet_delay += TD
                    visited_pos.append(best_nbr.pos)
                    self.logger.debug(f"  {current.node_id} -> {best_nbr.node_id} | Hop Delay: {TD:.4f}s")
                    current = best_nbr
                    
                    if auv_distance(current.pos, dest_pos) < p['TX_RANGE'] or current.depth <= 5.0:
                        buoy_node = Node(
                            node_id='Buoy', pos=dest_pos, channel_state=np.array([False]*M_SUBCARRIERS))
                        idle_final = np.where(
                            ~np.array(current.channel_state, dtype=bool))[0]
                        if not idle_final.any():
                            final_ch_idx = 0
                        else:
                            final_ch_idx = idle_final[0] % len(channels_khz)
                        final_td, _, _ = compute_transmission_delay_paper(
                            current, buoy_node, dest_pos, channels_khz[final_ch_idx], LP, CH_BANDWIDTH, p['PT_LINEAR'])

                        total_data_bits += LP
                        total_control_bits += 2 * CONTROL_PACKET_LP
                        packet_delay += final_td
                        visited_pos.append(dest_pos)
                        self.logger.info(f"  {current.node_id} -> Buoy | Final Hop Delay: {final_td:.4f}s")
                        packets_delivered += 1
                        total_end_to_end_delay += packet_delay
                        packet_delivered_flag = True
                        break # Break hop_count loop
                        
                self.q.put({'type': 'plot_route', 'route_pos': visited_pos,
                            'success': packet_delivered_flag})
                time.sleep(0.1) # Slow down sim for visualization

        except Exception as e:
            self.logger.critical(f"[OSAR Thread] FATAL ERROR: {e}", exc_info=True)
            time.sleep(1.0)

        finally:
            self.q.put({'type': 'osar_finished', 'data': {
                'params': p,
                'packets_generated': packets_generated,
                'packets_delivered': packets_delivered,
                'total_data_bits': total_data_bits,
                'total_control_bits': total_control_bits,
                'total_end_to_end_delay': total_end_to_end_delay
            }})
            self.logger.info("[OSAR Thread] Finished.")


if __name__ == "__main__":
    # Set the main thread's name
    threading.current_thread().name = 'Main-GUI-Thread'
    root = tk.Tk()
    app = SimulationApp(root)
    
    # --- MODIFIED: Add a clean exit handler ---
    def on_closing():
        print("--- Application Closing ---")
        if app.is_simulation_running():
            print("Attempting to stop threads...")
            app.stop_simulation() # Signal threads to stop
            root.destroy() # Close window
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    # --- End of Change ---
    
    root.mainloop()