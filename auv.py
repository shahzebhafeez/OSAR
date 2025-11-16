# auv.py
# A complete, self-contained GUI simulation for AUV path mapping,
# visualization, and data logging with CSV export.
#
# --- FIX v2: Efficient plotting and Run/Stop toggle button ---

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

# --- Use a professional, publication-quality plot style ---
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# SECTION 1: AUV CORE LOGIC
# ==============================================================================

# --- AUV Parameters ---
AUV_COVERAGE_RADIUS = 75.0
AUV_MIN_SPEED = 1.0  # m/s
AUV_MAX_SPEED = 5.0  # m/s
DEFAULT_AUV_RELAY_RADIUS = 50.0 # Distance within which AUV can relay to surface
AUV_UPDATE_INTERVAL_S = 0.1 # How often the AUV position updates (in seconds)
SIMULATION_DURATION_S = 60*60 # Run the simulation for 1 hour (3600 seconds)

# --- Battery Parameters ---
BATTERY_DEPLETION_RATE = 1 # Percent per second at average speed
LOW_BATTERY_THRESHOLD = 20.0  # Percent

# --- NEW: Node Movement Parameters ---
NODE_MAX_DRIFT_M = 10.0 # Max horizontal distance a node can drift from its anchor

# --- Utility Functions ---
def distance(a, b):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def generate_auv_route(area, surface_buoy_pos, auv_id):
    """
    Generates a top-to-bottom-to-top closed loop route for an AUV
    with a horizontal offset based on its ID.
    """
    horizontal_offset = 125.0 * auv_id
    
    start_pos = np.copy(surface_buoy_pos)
    start_pos[0] += horizontal_offset # Offset along the X-axis
    
    bottom_pos = np.copy(start_pos)
    bottom_pos[2] = area # Z = area is the bottom

    route = np.array([start_pos, bottom_pos, start_pos])
    return route

# --- AUV CLASS ---
class AUV:
    """Manages the state, movement, data collection, and relay of a single AUV."""
    def __init__(self, id, speed, route, surface_station_pos,
                 coverage_radius=AUV_COVERAGE_RADIUS,
                 relay_radius=DEFAULT_AUV_RELAY_RADIUS):
        self.id = id
        self.speed = speed
        self.route = route
        self.surface_station_pos = np.array(surface_station_pos) # Central buoy
        self.recharge_pos = np.copy(self.route[0]) # AUV's specific start/recharge point
        self.coverage_radius = coverage_radius
        self.relay_radius = relay_radius
        
        self.current_pos = np.array(self.route[0])
        self.target_waypoint_idx = 1
        
        self.covered_nodes = set()
        self.traveled_path = [np.array(self.route[0])]
        self.data_buffer = {}
        self.relayed_data_log = {}
        
        # --- MODIFIED: Removed artists from here, they will be managed by GUI ---
        
        self.rng = np.random.default_rng()

        self.battery = 100.0
        self.state = "Patrolling" # States: "Patrolling", "Returning to Charge"

    def collect_data(self, node_id):
        """Simulate collecting data from a node."""
        if node_id not in self.data_buffer:
            self.data_buffer[node_id] = time.time()
            return True
        return False

    def relay_data(self):
        """Simulate relaying collected data to the surface station."""
        if not self.data_buffer:
            return None
        relayed_node_ids = list(self.data_buffer.keys())
        for node_id in relayed_node_ids:
            self.relayed_data_log[node_id] = time.time()
        self.data_buffer.clear()
        return relayed_node_ids

    def update(self, dt, nodes):
        """Move the AUV, manage battery, check coverage, and relay data."""
        relayed_node_ids_this_tick = None
        is_moving = True

        # --- Battery & State Logic ---
        self.battery -= BATTERY_DEPLETION_RATE * (self.speed / AUV_MAX_SPEED) * dt
        self.battery = max(0, self.battery)

        if self.battery < LOW_BATTERY_THRESHOLD and self.state == "Patrolling":
            self.state = "Returning to Charge"
        
        if self.state == "Returning to Charge" and distance(self.current_pos, self.recharge_pos) < 5.0:
            self.battery = 100.0
            self.state = "Patrolling"
            self.target_waypoint_idx = 1 # Reset patrol route

        # --- Movement Logic ---
        target_pos = None
        if self.state == "Patrolling":
            if self.target_waypoint_idx >= len(self.route):
                self.target_waypoint_idx = 1 # Loop route
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
                # Clamp move_dist to not overshoot the target
                move_dist = min(move_dist, dist_to_target) 
                
                move_vec = (direction / dist_to_target) * move_dist
                
                # Apply a small, random perpendicular drift
                drift_factor = 0.4
                drift_vec = self.rng.uniform(-1, 1, 3) * move_dist * drift_factor
                unit_direction = direction / dist_to_target
                drift_vec -= np.dot(drift_vec, unit_direction) * unit_direction
                
                self.current_pos += move_vec + drift_vec
        
        # --- Update Path History ---
        self.traveled_path.append(np.copy(self.current_pos))
        if len(self.traveled_path) > 500: # Keep path history to 500 points
            self.traveled_path.pop(0)

        # --- Data Collection & Relay Logic ---
        collected_new_data_flag = False
        for node in nodes:
            if distance(self.current_pos, node.pos) <= self.coverage_radius:
                if node.node_id not in self.covered_nodes:
                    self.covered_nodes.add(node.node_id)
                if self.collect_data(node.node_id):
                    collected_new_data_flag = True

        if distance(self.current_pos, self.surface_station_pos) <= self.relay_radius:
            relayed_node_ids_this_tick = self.relay_data()

        return relayed_node_ids_this_tick, is_moving, collected_new_data_flag

# ==============================================================================
# SECTION 2: GUI APPLICATION AND SIMULATION CONTROL
# ==============================================================================

# --- Simulation Parameters ---
AREA = 500.0
DEFAULT_N_NODES = 30
DEFAULT_NUM_AUVS = 3

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

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AUV Path & Data Logging Simulation")
        self.root.geometry("1300x850")
        self.sim_thread = None
        self.simulation_running = threading.Event()
        self.q = queue.Queue()
        self.sim_nodes = []
        self.auvs = []
        self.sc_nodes = None # Scatter plot for NODES
        self.auv_artists = {} # MODIFIED: Dictionary to hold AUV artists
        self.simulation_log_data = []
        self.create_widgets()
        self.process_queue()
        
        # --- NEW: Add graceful exit ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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
            "AUV_COVERAGE_RADIUS": ("Coverage Radius (m):", AUV_COVERAGE_RADIUS),
            "SEED": ("Random Seed (0=random):", 0)
        }
        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        # --- MODIFIED: Changed to toggle button ---
        self.toggle_run_button = ttk.Button(left_panel, text="Run AUV Simulation", command=self.toggle_simulation)
        self.toggle_run_button.pack(fill=tk.X, pady=5)
        
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

    def log(self, message):
        """Thread-safe logging to the GUI."""
        self.q.put({'type': 'log', 'message': message})

    # --- NEW: Toggle and Stop functions ---
    def is_simulation_running(self):
        return self.sim_thread and self.sim_thread.is_alive()

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
    # --- End of New Functions ---

    def start_simulation(self):
        if self.is_simulation_running(): return
        
        self.log_text.delete('1.0', tk.END)
        self.log("--- New AUV Simulation Run Started ---")
        
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
            p['SURFACE_STATION_POS'] = np.array([p['AREA']/2, p['AREA']/2, 0.0])
            
            self.simulation_running.set()
            self.sim_thread = threading.Thread(target=self.run_auv_thread, args=(p,), daemon=True, name="SimThread")
            self.sim_thread.start()
            
            self.toggle_run_button.config(text="Stop Simulation", state=tk.NORMAL)
            
        except ValueError: self.log("Error: Please enter valid numbers.")
        except Exception as e:
            self.log(f"Error: {e}")
            self.toggle_run_button.config(text="Run AUV Simulation", state=tk.NORMAL)

    def process_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                if data['type'] == 'log': self.log(data['message'])
                elif data['type'] == 'plot_initial': self.plot_initial_setup(data['nodes_pos'], data['buoy_pos'], data['area'])
                
                # --- NEW: Handler for setting up AUV artists ---
                elif data['type'] == 'setup_auv_plots': self.setup_auv_plots(data['num_auvs'])
                
                elif data['type'] == 'update_nodes': self.update_node_plots(data['nodes_pos'])
                elif data['type'] == 'plot_auvs': self.update_auv_plots(data['auv_indices'])
                elif data['type'] == 'finished':
                    self.simulation_running.clear()
                    self.log("\n--- AUV Simulation Finished ---")
                    if self.simulation_log_data:
                        self.export_button.config(state=tk.NORMAL)
                    
        except queue.Empty: pass
        except Exception as e:
            print(f"Error in process_queue: {e}")
            # traceback.print_exc()
            self.log(f"Error processing GUI updates: {e}")
            self.simulation_running.clear()

        # --- MODIFIED: Cleanup logic ---
        if not self.is_simulation_running():
            if self.toggle_run_button['state'] != tk.NORMAL:
                self.toggle_run_button.config(text="Run AUV Simulation", state=tk.NORMAL)
                if self.simulation_log_data:
                    self.export_button.config(state=tk.NORMAL)
        
        self.root.after(100, self.process_queue)
        # --- End of Change ---

    def plot_initial_setup(self, nodes_pos, buoy_pos, area):
        if nodes_pos.any():
            self.sc_nodes = self.ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2],
                                         c=nodes_pos[:, 2], cmap="viridis_r", s=50, alpha=0.9,
                                         edgecolors='black', linewidth=0.6, label="Sensor Nodes")
        self.ax.scatter(buoy_pos[0], buoy_pos[1], buoy_pos[2], c="red", s=180, marker="^",
                        label="Surface Buoy", edgecolors='black', linewidth=1.2, depthshade=False)
        self.ax.scatter([], [], [], c='magenta', s=250, marker='X', label="AUV")
        
        self.ax.set_xlabel("X (m)", fontweight='bold')
        self.ax.set_ylabel("Y (m)", fontweight='bold')
        self.ax.set_zlabel("Depth (m)", fontweight='bold')
        self.ax.set_title("AUV Path Simulation", fontsize=14, fontweight='bold')
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
            # Note: _offsets3d is the "private" but standard way to update 3D scatter
            self.sc_nodes._offsets3d = (nodes_pos[:, 0], nodes_pos[:, 1], nodes_pos[:, 2])
            # self.fig.canvas.draw_idle() # Not needed, update_auv_plots will draw

    # --- NEW: Creates AUV artists once for speed ---
    def setup_auv_plots(self, num_auvs):
        """Creates the plot artists for each AUV one time."""
        self.auv_artists.clear()
        for i in range(num_auvs):
            # Create one artist for the path (a line)
            path_line, = self.ax.plot([], [], [], 
                                      color='magenta', linestyle=':', 
                                      linewidth=1.5, zorder=9, alpha=0.7)
            
            # --- THIS IS THE FIX ---
            # Changed 'c=' to 'color=' and 's=250' to 'markersize=15'
            marker_point, = self.ax.plot([], [], [], 
                                         color='magenta', markersize=15, marker='X',
                                         zorder=10, markeredgecolor='black')
            # --- End of Fix ---
            
            self.auv_artists[i] = {
                'path': path_line,
                'marker': marker_point
            }
        self.fig.canvas.draw_idle()

    # --- MODIFIED: Efficiently updates artist data, no re-plotting ---
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
                
                # 3. Update Colors based on state
                color = 'magenta'
                if auv.state == "Returning to Charge":
                    color = 'orange'
                
                artists['path'].set_color(color)
                artists['marker'].set_color(color) # 'c=' is an alias for 'color'
                
        # Draw all changes at once
        self.fig.canvas.draw_idle()
    # --- End of Change ---

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

    def run_auv_thread(self, p):
        self.q.put({'type': 'log', 'message': "[Sim]: Simulation thread started."})
        try:
            start_time = time.time()
            seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
            rng = np.random.default_rng(seed)
            self.q.put({'type': 'log', 'message': f"[Sim]: Using seed: {seed}"})
            self.sim_nodes = place_nodes_randomly(int(p['N_NODES']), p['AREA'], rng)
            nodes_pos = np.array([n.pos for n in self.sim_nodes]) if self.sim_nodes else np.array([])
            self.q.put({'type': 'plot_initial', 'nodes_pos': nodes_pos, 'buoy_pos': p['SURFACE_STATION_POS'], 'area': p['AREA']})

            local_auv_list = []
            for i in range(int(p['NUM_AUVS'])):
                speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
                route = generate_auv_route(p['AREA'], p['SURFACE_STATION_POS'], i)
                auv = AUV(i, speed, route, p['SURFACE_STATION_POS'], coverage_radius=p['AUV_COVERAGE_RADIUS'])
                local_auv_list.append(auv)
                self.q.put({'type': 'log', 'message': f"[AUV {i}]: Created. Speed: {speed:.1f} m/s, Radius: {auv.coverage_radius} m"})
            self.auvs = local_auv_list
            
            # --- NEW: Send message to create AUV plot artists ---
            self.q.put({'type': 'setup_auv_plots', 'num_auvs': len(local_auv_list)})
            # --- End of Change ---

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
                             # This check is redundant, but good for logging once
                             pass # The log will happen when state *flips*
                        elif auv.state == "Patrolling" and auv.battery == 100.0 and (current_time - last_speed_change_time[auv.id] < 1.5):
                             # Log recharge
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

        except Exception as e:
            print(f"[Sim Thread] Error: {e}")
            # traceback.print_exc()
            self.log(f"[Sim] Error: {e}")
        finally:
            self.log("\n--- AUV FINAL STATS ---")
            if self.auvs:
                for auv in self.auvs:
                    self.log(f"--- AUV {auv.id} Report ---")
                    self.log(f"  Nodes Detected: {sorted(list(auv.covered_nodes)) or 'None'}")
                    relayed = sorted(list(auv.relayed_data_log.keys()))
                    self.log(f"  Data Relayed (IDs): {relayed or 'None'}")
            
            self.q.put({'type': 'finished'})
            print("[Sim Thread] Finished.")
            
    # --- NEW: Graceful exit handler ---
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