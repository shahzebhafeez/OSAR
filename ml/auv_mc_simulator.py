# auv_mc_merged_simulator.py
# A complete, self-contained GUI simulation for AUV path mapping,
# real-time ML classification (LC/OC), and data logging.

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
    # --- MODIFIED: Import from 'auv.py' (your provided file) ---
    from auv import (AUV as BaseAUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
                      DEFAULT_AUV_RELAY_RADIUS,AUV_COVERAGE_RADIUS,LOW_BATTERY_THRESHOLD,distance,
                      distance as auv_distance)
    
    # We rename 'distance' to 'auv_distance' to match what the ML functions expect
    
except Exception as e:
    print(
        f"Warning: Could not import 'auv' (auv.py) from: {project_root}. AUV functionality disabled.\n{e}")
    AUV_AVAILABLE = False

    def auv_distance(a, b):
        raise RuntimeError("AUV module not available")

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
# SECTION 1: AUV CORE LOGIC (MERGED)
# ==============================================================================

# --- AUV Parameters ---
# These are loaded from auv.py, but we can add ML params
SVM_UPDATE_PERIOD_S = 5 # How often the MC thread re-classifies AUVs
AUV_UPDATE_INTERVAL_S = 0.1 # From auv.py
SIMULATION_DURATION_S = 60*60 # From auv.py

# --- NEW: Merged AUV Class ---
# Extends the AUV class from your 'auv.py' to add ML flags
class AUV(BaseAUV):
    def __init__(self, *args, **kwargs):
        # Call the parent __init__ from your auv.py
        super().__init__(*args, **kwargs)
        
        # Add the new flags required for ML classification
        self.is_lc = False
        self.is_oc = False
# --- End of Merged Class ---


# ==============================================================================
# SECTION 2: GUI APPLICATION AND SIMULATION CONTROL
# ==============================================================================

# --- Simulation Parameters ---
AREA = 500.0
DEFAULT_N_NODES = 30
DEFAULT_NUM_AUVS = 3
NODE_MAX_DRIFT_M = 10.0 # From auv.py

# --- Re-define Node and place_nodes_randomly (from auv.py) ---
class Node:
    """Represents a sensor node that can drift from an anchor point."""
    def __init__(self, node_id, pos, rng):
        self.node_id = node_id
        self.initial_pos = np.copy(pos) # Anchor point
        self.pos = np.array(pos)
        self.rng = rng

    def update_position(self, max_drift_m):
        """Applies a random horizontal drift relative to the anchor point."""
        drift_x = self.rng.uniform(-max_drift_m, max_drift_m)
        drift_y = self.rng.uniform(-max_drift_m, max_drift_m)
        
        self.pos[0] = self.initial_pos[0] + drift_x
        self.pos[1] = self.initial_pos[1] + drift_y
        # Z position remains anchored

def place_nodes_randomly(N, area, rng):
    nodes = []
    for i in range(N):
        pos = rng.uniform(0, area, 3)
        pos[2] = rng.uniform(area * 0.1, area)
        nodes.append(Node(i, pos, rng)) # Pass rng to Node
    return nodes

# --- NEW: QueueHandler for logging ---
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
        self.root.title("AUV Path & MC Classification Simulation") # <-- New Title
        self.root.geometry("1300x850")
        
        # --- MODIFIED: Add threads from both files ---
        self.orchestrator_thread = None
        self.auv_thread = None
        self.mc_thread = None
        # --- End of Change ---
        
        self.simulation_running = threading.Event()
        self.q = queue.Queue()
        self.sim_nodes = []
        self.auvs = []
        self.sc_nodes = None # Scatter plot for NODES
        self.auv_artists = {}
        self.simulation_log_data = []
        
        # --- NEW: Setup Logger ---
        self.logger = None 
        self.setup_logging()
        # --- End of Change ---

        # --- NEW: ML Model attributes ---
        self.lc_model = None
        self.lc_scaler = None
        self.oc_model = None
        self.oc_scaler = None
        self.models_loaded = False
        if ML_AVAILABLE:
            self.models_loaded = self.load_models(project_root)
        else:
            self.logger.warning("ML not available; models will not be loaded.")
        # --- End of Change ---
        
        # --- THIS IS THE FIX ---
        self.auv_available = AUV_AVAILABLE
        # --- END OF FIX ---

        self.create_widgets()
        self.process_queue()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    # --- NEW: setup_logging (from auv_mc_simulator.py) ---
    def setup_logging(self):
        """Configures the logger to send messages to the GUI queue and a file."""
        self.logger = logging.getLogger('AUV_MC_Sim')
        self.logger.setLevel(logging.DEBUG) 
        self.logger.propagate = False
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 1. File Handler
        log_file_path = ""
        try:
            log_file_path = os.path.join(script_dir, 'auv_mc_merged_log.txt') # <-- New Log File
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s [%(threadName)-18s] [%(levelname)-5s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"CRITICAL: Failed to create log file at {log_file_path}. Error: {e}")

        # 2. Queue Handler
        gui_formatter = logging.Formatter(
            '[%(threadName)-18s] %(message)s'
        )
        queue_handler = QueueHandler(self.q)
        queue_handler.setFormatter(gui_formatter)
        self.logger.addHandler(queue_handler)
        
        self.logger.info(f"Logging configured. Log file: {log_file_path if log_file_path else 'FILE FAILED'}")

    # --- NEW: load_models (from auv_mc_simulator.py) ---
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
        controls_frame = ttk.LabelFrame(left_panel, text="AUV Parameters", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.params = {}
        param_list = {
            "N_NODES": ("Number of Nodes:", DEFAULT_N_NODES),
            "AREA": ("Area Size (m):", AREA),
            "NUM_AUVS": ("Num AUVs:", DEFAULT_NUM_AUVS),
            "AUV_COVERAGE_RADIUS": ("Coverage Radius (m):", AUV_COVERAGE_RADIUS), # From auv.py
            "SEED": ("Random Seed (0=random):", 0)
        }
        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        self.toggle_run_button = ttk.Button(left_panel, text="Run AUV Simulation", command=self.toggle_simulation)
        self.toggle_run_button.pack(fill=tk.X, pady=5)
        
        # --- NEW: Check for ML models before enabling ---
        if not self.models_loaded or not AUV_AVAILABLE:
            self.toggle_run_button.config(
                text="Models/AUV Not Found", state=tk.DISABLED)
        # --- End of Change ---
        
        self.export_button = ttk.Button(left_panel, text="Export Log to CSV", command=self.export_log_to_csv, state=tk.DISABLED)
        self.export_button.pack(fill=tk.X, pady=5)

        log_frame = ttk.LabelFrame(left_panel, text="AUV Data Logger", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=40, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log model/auv status
        if not self.models_loaded:
            self.logger.error(
                f"Error: SVM models not found in {project_root}\nPlease run 'data_collector.py' and 'train_model.py'.")
        if not self.auv_available:
            self.logger.error(
                "Error: AUV components (auv.py) not found in project root. AUV simulation disabled.")

    def log(self, message):
        """Thread-safe logging to the GUI."""
        self.q.put({'type': 'log', 'message': message})

    def is_simulation_running(self):
        # --- MODIFIED: Check orchestrator thread ---
        return self.orchestrator_thread and self.orchestrator_thread.is_alive()

    def toggle_simulation(self):
        if self.is_simulation_running():
            self.stop_simulation()
        else:
            self.start_simulation()

    def stop_simulation(self):
        if self.is_simulation_running():
            self.log("--- SIMULATION STOP REQUESTED BY USER ---")
            self.simulation_running.clear()  # Signal thread to stop
            self.toggle_run_button.config(text="Stopping...", state=tk.DISABLED)

    def start_simulation(self):
        if self.is_simulation_running(): return
        
        self.log_text.delete('1.0', tk.END)
        self.log("--- New AUV + MC Simulation Run Started ---") # <-- New Title
        
        # Clear plots and artists
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.auv_artists.clear()
        self.sc_nodes = None
        
        self.simulation_log_data.clear()
        self.export_button.config(state=tk.DISABLED)
        self.auvs.clear()
        self.sim_nodes.clear()
        
        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            p["N_NODES"] = int(p["N_NODES"])
            p["NUM_AUVS"] = int(p["NUM_AUVS"])
            p["SEED"] = int(p["SEED"])
            # --- MODIFIED: Add AUV_RELAY_RADIUS from auv_mc_simulator.py ---
            p["AUV_RELAY_RADIUS"] = float(self.params["AUV_COVERAGE_RADIUS"].get()) # Re-use this field
            # --- End of Change ---
            p['SURFACE_STATION_POS'] = np.array([p['AREA']/2, p['AREA']/2, 0.0])
            
            # --- NEW: ML Model Check ---
            if not self.models_loaded:
                self.log("Error: Cannot start, ML models not loaded.")
                return
            # --- End of Change ---
            
            self.simulation_running.set()
            # --- MODIFIED: Start orchestrator thread ---
            self.orchestrator_thread = threading.Thread(target=self.run_concurrent_simulation, args=(p,), daemon=True, name="Orchestrator")
            self.orchestrator_thread.start()
            # --- End of Change ---
            
            self.toggle_run_button.config(text="Stop Simulation", state=tk.NORMAL)
            
        except ValueError: self.log("Error: Please enter valid numbers.")
        except Exception as e:
            self.log(f"Error: {e}")
            traceback.print_exc()
            self.toggle_run_button.config(text="Run AUV Simulation", state=tk.NORMAL)

    # --- NEW: Orchestrator Thread (from auv_mc_simulator.py) ---
    def run_concurrent_simulation(self, p):
        """
        Orchestrator thread to run AUV and MC threads concurrently.
        """
        try:
            # --- 1. Create sensor nodes ---
            self.log(f"Creating {int(p['N_NODES'])} sensor nodes...")
            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng_nodes = np.random.default_rng(seed)
            self.sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng_nodes)
            self.log(f"Created {len(self.sim_nodes)} nodes.")
            
            # --- 2. Setup the Plot ---
            nodes_pos = np.array([n.pos for n in self.sim_nodes]) if self.sim_nodes else np.array([])
            self.q.put({'type': 'plot_initial', 'nodes_pos': nodes_pos, 'buoy_pos': p['SURFACE_STATION_POS'], 'area': p['AREA']})
            time.sleep(0.5) # Give GUI time to draw

            # --- 3. Run AUV + MC Simulation (Concurrent) ---
            self.log("\n--- Starting AUV + MC Simulation (Concurrent) ---")
            self.auv_thread = threading.Thread(
                target=self.run_auv_thread, args=(p,), daemon=True, name='AUV-Thread')
            self.mc_thread = threading.Thread(
                target=self.run_mc_logic, args=(p,), daemon=True, name='MC-Thread')

            self.auv_thread.start()
            self.mc_thread.start()

            self.auv_thread.join()
            if not self.simulation_running.is_set():
                 self.log("Simulation stopped during AUV phase.")
                 return

            self.log("--- AUV Simulation Finished ---")

            # --- 4. Stop MC thread ---
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
            self.q.put({'type': 'finished'}) # Send final "finished" msg
    # --- End of New Function ---

    def process_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                if data['type'] == 'log': self.log_text.insert(tk.END, data['message'] + "\n"); self.log_text.see(tk.END)
                elif data['type'] == 'plot_initial': self.plot_initial_setup(data['nodes_pos'], data['buoy_pos'], data['area'])
                elif data['type'] == 'setup_auv_plots': self.setup_auv_plots(data['num_auvs'])
                elif data['type'] == 'update_nodes': self.update_node_plots(data['nodes_pos'])
                elif data['type'] == 'plot_auvs': self.update_auv_plots(data['auv_indices'])
                elif data['type'] == 'finished':
                    self.simulation_running.clear()
                    self.log("\n--- Simulation Finished ---")
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
                if self.simulation_log_data:
                    self.export_button.config(state=tk.NORMAL)
        
        self.root.after(100, self.process_queue)

    def plot_initial_setup(self, nodes_pos, buoy_pos, area):
        if nodes_pos.any():
            self.sc_nodes = self.ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2],
                                         c=nodes_pos[:, 2], cmap="viridis_r", s=50, alpha=0.9,
                                         edgecolors='black', linewidth=0.6, label="Sensor Nodes")
        self.ax.scatter(buoy_pos[0], buoy_pos[1], buoy_pos[2], c="red", s=180, marker="^",
                        label="Surface Buoy", edgecolors='black', linewidth=1.2, depthshade=False)
        
        # --- NEW: Add legend entries for LC/OC ---
        self.ax.scatter([], [], [], c='magenta', s=150, marker='X', label="AUV (Normal)")
        self.ax.scatter([], [], [], c='orange', s=150, marker='X', label="AUV (Charging)")
        self.ax.scatter([], [], [], c='cyan', s=250, marker='X', label="AUV (LC)")
        self.ax.scatter([], [], [], c='red', s=300, marker='X', label="AUV (OC)")
        # --- End of Change ---
        
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
        """Efficiently updates the positions of the node scatter plot."""
        if self.sc_nodes and len(nodes_pos) > 0:
            self.sc_nodes._offsets3d = (nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2])

    def setup_auv_plots(self, num_auvs):
        """Creates the plot artists for each AUV one time."""
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
        """Efficiently updates the data for existing AUV plot artists."""
        for idx in auv_indices:
            if idx in self.auv_artists and idx < len(self.auvs):
                auv = self.auvs[idx]
                artists = self.auv_artists[idx]
                
                # 1. Update Path
                path_data = np.array(auv.traveled_path)
                if path_data.shape[0] > 1:
                    artists['path'].set_data_3d(path_data[:, 0], path_data[:, 1], path_data[:, 2])
                
                # 2. Update Marker
                pos = auv.current_pos
                artists['marker'].set_data_3d([pos[0]], [pos[1]], [pos[2]])
                
                # --- MODIFIED: New color logic ---
                # 3. Update Colors based on state
                color = 'magenta' # Default
                if auv.state == "Returning to Charge":
                    color = 'orange'
                
                # LC/OC status overrides battery status
                if auv.is_lc:
                    color = 'cyan'
                if auv.is_oc:
                    color = 'red'
                
                artists['path'].set_color(color)
                artists['marker'].set_color(color)
                # --- End of Change ---
                
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
                sorted_data = sorted(self.simulation_log_data, key=lambda row: row[1])
                writer.writerows(sorted_data)
            self.log(f"Log successfully exported to {filepath}")
        except IOError as e:
            self.log(f"Error exporting file: {e}")

    # --- NEW: collect_auv_features (from auv_mc_simulator.py) ---
    def collect_auv_features(self):
        features = []
        auv_map = {}

        current_auv_list = list(self.auvs)
        if not self.sim_nodes or not current_auv_list:
            return None, None
        
        for i, auv in enumerate(current_auv_list):
            count = 0
            for node in self.sim_nodes:
                # --- MODIFIED: Use 'distance' (from auv.py) ---
                if distance(auv.current_pos, node.pos) < auv.coverage_radius:
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

    # --- NEW: update_controllers_svm (from auv_mc_simulator.py) ---
    def update_controllers_svm(self, p):
        self.logger.debug("[MC]: Waking up to update controllers...")

        X_features, auv_map = self.collect_auv_features()

        if X_features is None or X_features.shape[0] == 0:
            self.logger.debug("[MC]: AUVs/Nodes not ready. Skipping update.")
            return
        
        # --- MODIFIED: Reset flags on AUV objects from self.auvs ---
        for auv in self.auvs:
            auv.is_lc = False
            auv.is_oc = False
        # --- End of Change ---

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
                auv = auv_map[i] # auv_map holds the actual AUV object
                auv.is_lc = True
                selected_lc_ids.add(auv.id)

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
                auv.is_lc = True # An OC is also an LC
                selected_oc_id = auv.id
                if auv.id not in selected_lc_ids:
                    selected_lc_ids.add(auv.id)

        self.logger.info(f"[MC]: LCs updated: {selected_lc_ids or 'None'}")
        self.logger.info(f"[MC]: OC updated: {selected_oc_id or 'None'}")
        
        # Note: We don't need to send a plot message here,
        # the AUV thread's plot message will pick up the new flags.

    # --- NEW: run_mc_logic (from auv_mc_simulator.py) ---
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
            
    # --- This is the original run_auv_thread from auv.py ---
    # --- It is now our AUV-Thread ---
    def run_auv_thread(self, p):
        self.log("[AUV Sim]: AUV simulation thread started.")
        try:
            start_time = time.time()
            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng = np.random.default_rng(seed + 1) # Use different seed from nodes
            
            # Note: self.sim_nodes was already created by the orchestrator
            nodes_pos = np.array([n.pos for n in self.sim_nodes]) if self.sim_nodes else np.array([])
            
            local_auv_list = []
            for i in range(int(p['NUM_AUVS'])):
                speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                route = generate_auv_route(p['AREA'], p['SURFACE_STATION_POS'], i)
                
                # --- MODIFIED: Use the new merged AUV class ---
                auv = AUV(i, speed, route, p['SURFACE_STATION_POS'], coverage_radius=p['AUV_COVERAGE_RADIUS'])
                # --- End of Change ---
                
                local_auv_list.append(auv)
                self.log(f"[AUV {i}]: Created. Speed: {speed:.1f} m/s, Radius: {auv.coverage_radius} m")
            self.auvs = local_auv_list
            
            self.q.put({'type': 'setup_auv_plots', 'num_auvs': len(local_auv_list)})

            last_log_time = time.time()
            last_speed_change_time = {auv.id: time.time() for auv in local_auv_list}
            last_node_update_time = time.time()

            while self.simulation_running.is_set() and (time.time() - start_time) < SIMULATION_DURATION_S:
                current_time = time.time()
                updated_indices = []
                
                for idx, auv in enumerate(local_auv_list):
                    if current_time - last_speed_change_time[auv.id] > rng.uniform(15, 30):
                        auv.speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                        self.log(f"[AUV {auv.id}]: Speed changed to {auv.speed:.1f} m/s")
                        last_speed_change_time[auv.id] = current_time

                    relayed_ids, _, collected_new = auv.update(AUV_UPDATE_INTERVAL_S, self.sim_nodes)

                    if relayed_ids:
                        self.log(f"[AUV {auv.id}]: Relayed data for nodes {relayed_ids}.")
                    if collected_new:
                        self.log(f"[AUV {auv.id}]: Collected new data from node(s).")
                    
                    updated_indices.append(idx)
                
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
                self.log("[Sim]: Simulation 1-hour duration finished.")

        except Exception as e:
            self.log(f"[AUV Thread] Error: {e}")
            traceback.print_exc()
        finally:
            self.log("\n--- AUV FINAL STATS ---")
            if self.auvs:
                for auv in self.auvs:
                    self.log(f"--- AUV {auv.id} Report ---")
                    self.log(f"  Nodes Detected: {sorted(list(auv.covered_nodes)) or 'None'}")
                    relayed = sorted(list(auv.relayed_data_log.keys()))
                    self.log(f"  Data Relayed (IDs): {relayed or 'None'}")
            
            # Don't send 'finished' here, let the orchestrator do it
            self.log("[AUV Thread] Finished.")
            
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