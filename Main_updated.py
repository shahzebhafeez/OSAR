# osar_gui.py
# A graphical user interface for the OSAR simulation.
# --- ENHANCED with professional plotting, node labels, and interactive hover tooltips ---

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue

# --- Use a professional, publication-quality plot style ---
plt.style.use('seaborn-v0_8-whitegrid')

# ---------- Global Parameters (Default Values) ----------
LP = 512  # Packet size in bits (64 bytes)
CH_BANDWIDTH = 6e3  # 6 kHz bandwidth per channel
DEFAULT_PT_DB = 150.0 # Default transmit power in dB re uPa
AREA = 500.0
TX_RANGE = 250.0
P_BUSY = 0.7
N_NODES = 30 # Reduced for better label visibility by default
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10

# --- Environmental Parameters (Assumed) ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5
EPS = 1e-12

# ---------- Acoustic & Utility Functions (Unchanged) ----------
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

def distance(a, b):
    return np.linalg.norm(a - b)

def place_nodes_uniform(N, area, rng):
    side = int(np.ceil(N**(1/3)))
    coords = np.linspace(0, area, side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    final_nodes = grid[:N]
    jitter = rng.uniform(-area / (4 * side), area / (4 * side), final_nodes.shape)
    return final_nodes + jitter

class Node:
    def __init__(self, node_id, pos, channel_state):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.depth = pos[2]
        self.channel_state = channel_state
        self.sound_speed = sound_speed(self.depth)

def compute_transmission_delay_paper(src, dest, dest_pos, f_khz, lp, bw, pt):
    vec_id, vec_ij = dest_pos - src.pos, dest.pos - src.pos
    DiD = np.linalg.norm(vec_id) + EPS
    proj_len = np.dot(vec_ij, vec_id) / DiD
    if proj_len <= 0: proj_len = EPS
    N_hop = max(DiD / proj_len, 1.0)
    depth_diff = max(src.depth - dest.depth, 0.0)
    mid_depth_m = max((src.depth + dest.depth) / 2.0, 0.0)
    c = sound_speed(mid_depth_m)
    PD_ij = depth_diff / (c + EPS)
    dist_m = max(np.linalg.norm(dest.pos - src.pos), EPS)
    A_df = path_loss_linear(dist_m, f_khz)
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    snr_linear = pt / (A_df * noise_total + EPS)
    r_ij_ch = bw * np.log2(1.0 + snr_linear)
    if r_ij_ch <= 1e-12: return float('inf'), 0.0, 0.0
    TD = ((lp / r_ij_ch) + PD_ij) * N_hop
    return TD, r_ij_ch, snr_linear

def select_best_next_hop(src, nodes, dest_pos, tx_range, channels, lp, bw, pt):
    candidates = []
    for nbr in nodes:
        if nbr.node_id == src.node_id: continue
        if distance(src.pos, nbr.pos) > tx_range: continue
        if nbr.depth >= src.depth: continue
        if np.dot(dest_pos - src.pos, nbr.pos - src.pos) <= 0: continue
        common_idle = np.where(~src.channel_state & ~nbr.channel_state)[0]
        if len(common_idle) == 0: continue
        best_td, best_ch, best_snr, best_pair = float('inf'), None, None, None
        for idx in common_idle:
            ch_idx = idx % len(channels)
            f_center_khz = channels[ch_idx]
            TD, _, snr_lin = compute_transmission_delay_paper(src, nbr, dest_pos, f_center_khz, lp, bw, pt)
            if TD < best_td:
                best_td, best_ch, best_snr, best_pair = TD, ch_idx, snr_lin, nbr
        if best_pair:
            candidates.append((best_pair, best_td, best_ch, best_snr))
    if not candidates: return None, None, None, None
    return min(candidates, key=lambda x: x[1])

# ---------- Main GUI Application ----------
class SimulationApp:
    """
    A GUI for visualizing the OSAR underwater routing simulation.
    Includes interactive plotting features for node inspection.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("OSAR Underwater Routing Simulation")
        self.root.geometry("1300x850")

        self.sim_thread = None
        self.q = queue.Queue()
        self.radius_plot = None
        self.sim_nodes = [] # To store node objects for hover info
        self.sc = None      # To store the scatter plot artist

        self.create_widgets()
        self.init_hover_functionality()
        self.process_queue()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        controls_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.params = {}
        param_list = {
            "N_NODES": ("Number of Nodes:", N_NODES),
            "AREA": ("Area Size (m):", AREA),
            "TX_RANGE": ("TX Range (m):", TX_RANGE),
            "P_BUSY": ("P(Busy Channel):", P_BUSY),
            "PT_DB": ("Transmit Power (dB):", DEFAULT_PT_DB),
            "M_CHANNELS": ("Num Channels:", DEFAULT_M_CHANNELS),
            "SEED": ("Random Seed (0=random):", 0)
        }

        for i, (key, (text, val)) in enumerate(param_list.items()):
            ttk.Label(controls_frame, text=text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.params[key] = tk.StringVar(value=str(val))
            ttk.Entry(controls_frame, textvariable=self.params[key], width=10).grid(row=i, column=1, sticky=tk.W, pady=2)

        self.run_button = ttk.Button(left_panel, text="Run Simulation", command=self.start_simulation)
        self.run_button.pack(fill=tk.X, pady=5)

        log_frame = ttk.LabelFrame(left_panel, text="Live Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=40, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_hover_functionality(self):
        """Sets up the annotation box and connects the hover event."""
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="yellow", alpha=0.8),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def update_annot(self, ind):
        """Updates the annotation text and position for a selected node."""
        node_index = ind["ind"][0]
        node = self.sim_nodes[node_index]
        pos = self.sc.get_offsets()[node_index]
        
        self.annot.xy = pos
        text = (f"ID: {node.node_id}\n"
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                f"Depth: {node.depth:.1f} m")
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.8)

    def hover(self, event):
        """Handles mouse hover events to show/hide the annotation."""
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def draw_tx_radius(self, center, radius):
        """Helper function to draw the wireframe transmission radius."""
        if self.radius_plot:
            self.radius_plot.remove()

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        self.radius_plot = self.ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.3, linewidth=0.8)
        self.canvas.draw()
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_simulation(self):
        if self.sim_thread and self.sim_thread.is_alive(): return
        
        self.log_text.delete('1.0', tk.END)
        self.ax.clear()

        try:
            p = {key: float(var.get()) for key, var in self.params.items()}
            # Cast integer params
            p["N_NODES"] = int(p["N_NODES"])
            p["M_CHANNELS"] = int(p["M_CHANNELS"])
            p["SEED"] = int(p["SEED"])
            # Convert power from dB to linear for calculations
            p["PT_LINEAR"] = 10**(p["PT_DB"] / 10.0)

            self.sim_thread = threading.Thread(target=self.run_simulation_thread, args=(p,), daemon=True)
            self.sim_thread.start()
            self.run_button.config(state=tk.DISABLED)

        except ValueError:
            self.log("Error: Please enter valid numbers for all parameters.")

    def process_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                if data['type'] == 'log':
                    self.log(data['message'])
                elif data['type'] == 'plot_nodes':
                    self.ax.clear()
                    self.radius_plot = None # Clear any old radius plot
                    
                    self.sim_nodes = data['nodes']
                    
                    # --- ENHANCED PLOTTING AESTHETICS ---
                    # Plot all sensor nodes
                    self.sc = self.ax.scatter(data['pos'][:, 0], data['pos'][:, 1], data['pos'][:, 2], 
                                       c=data['depths'], cmap="viridis_r", s=50, alpha=0.9, 
                                       edgecolors='black', linewidth=0.6, label="Sensor Nodes")
                    
                    # Plot Source and Destination with higher visibility
                    self.ax.scatter(data['src'][0], data['src'][1], data['src'][2], 
                                    c='lime', s=180, marker="o", label="Source Node", 
                                    edgecolors='black', linewidth=1.2, depthshade=False)
                    self.ax.scatter(data['dest'][0], data['dest'][1], data['dest'][2], 
                                    c="red", s=180, marker="^", label="Surface Buoy", 
                                    edgecolors='black', linewidth=1.2, depthshade=False)
                    
                    # Add static node labels if the count is manageable
                    if len(self.sim_nodes) <= 40:
                        for node in self.sim_nodes:
                            self.ax.text(node.pos[0], node.pos[1], node.pos[2], f' {node.node_id}', 
                                         color='black', fontsize=8, zorder=10)
                    
                    # --- AXIS AND TITLE STYLING ---
                    self.ax.set_xlabel("X (m)", fontweight='bold')
                    self.ax.set_ylabel("Y (m)", fontweight='bold')
                    self.ax.set_zlabel("Depth (m)", fontweight='bold')
                    self.ax.set_title("OSAR Underwater Network Topology", fontsize=14, fontweight='bold')
                    self.ax.invert_zaxis()
                    self.ax.legend(loc='upper left')
                    self.ax.view_init(elev=25, azim=-75) # A good default viewing angle
                    self.fig.tight_layout()
                    self.canvas.draw()
                    
                elif data['type'] == 'plot_route':
                    route_pos = np.array(data['route_pos'])
                    self.ax.plot(route_pos[:, 0], route_pos[:, 1], route_pos[:, 2], 
                                 color="mediumblue", linewidth=3.0, marker='o', 
                                 markersize=7, markerfacecolor='orange', zorder=5)
                    self.canvas.draw()
                elif data['type'] == 'draw_radius':
                    self.draw_tx_radius(data['pos'], data['radius'])
                elif data['type'] == 'finished':
                    if self.radius_plot:
                        self.radius_plot.remove()
                        self.radius_plot = None
                    self.run_button.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def run_simulation_thread(self, p):
        seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
        rng = np.random.default_rng(seed)
        self.q.put({'type': 'log', 'message': f"Simulation started with seed: {seed}"})
        
        ch_bw_khz = CH_BANDWIDTH / 1000.0
        start_center_freq = START_FREQ_KHZ + (ch_bw_khz / 2.0)
        channels_khz = start_center_freq + np.arange(p["M_CHANNELS"]) * ch_bw_khz
        self.q.put({'type': 'log', 'message': f"Channels (kHz): {np.round(channels_khz, 1)}"})

        if p['N_NODES'] == 0:
            self.q.put({'type': 'log', 'message': "No nodes to simulate."}); self.q.put({'type': 'finished'}); return
            
        positions = place_nodes_uniform(p['N_NODES'], p['AREA'], rng)
        for i in range(p['N_NODES']):
            if positions[i, 2] < 1.0: positions[i, 2] = rng.uniform(1.0, p['AREA'] / 2)
            
        nodes = [Node(i, positions[i], rng.random(M_SUBCARRIERS) < p['P_BUSY']) for i in range(p['N_NODES'])]
        dest_pos = np.array([p['AREA']/2, p['AREA']/2, 0.0])
        src = max(nodes, key=lambda n: n.depth)
        self.q.put({'type': 'log', 'message': f"Source: Node {src.node_id}, Depth={src.depth:.1f} m"})
        
        self.q.put({
            'type': 'plot_nodes', 
            'nodes': nodes, # Pass the full node objects for hover info
            'pos': np.array([n.pos for n in nodes]),
            'depths': [n.depth for n in nodes],
            'src': src.pos,
            'dest': dest_pos
        })
        time.sleep(1.0)

        route, current = [src.node_id], src
        total_delay, visited_pos = 0.0, [current.pos]
        
        while True:
            for node in nodes: node.channel_state = rng.random(M_SUBCARRIERS) < p['P_BUSY']
            
            # Show radius of the current forwarding node
            self.q.put({'type': 'draw_radius', 'pos': current.pos, 'radius': p['TX_RANGE']})
            time.sleep(0.75) # Pause to make radius visible
            
            best_nbr, TD, ch, snr_lin = select_best_next_hop(current, nodes, dest_pos, p['TX_RANGE'], channels_khz, LP, CH_BANDWIDTH, p['PT_LINEAR'])
            
            attempt = 0
            while best_nbr is None:
                attempt += 1; self.q.put({'type': 'log', 'message': f"🔄 Attempt {attempt}: Re-sensing..."})
                time.sleep(0.2)
                for n in nodes: n.channel_state = rng.random(M_SUBCARRIERS) < p['P_BUSY']
                best_nbr, TD, ch, snr_lin = select_best_next_hop(current, nodes, dest_pos, p['TX_RANGE'], channels_khz, LP, CH_BANDWIDTH, p['PT_LINEAR'])
                if attempt > 20:
                    self.q.put({'type': 'log', 'message': f"⚠️ Node {current.node_id} is stuck. Aborting."}); self.q.put({'type': 'finished'}); return

            total_delay += TD; route.append(best_nbr.node_id); visited_pos.append(best_nbr.pos)
            log_msg = f"→ {best_nbr.node_id} | D={best_nbr.depth:.1f}m | TD={TD:.4f}s | SNR={10*np.log10(snr_lin+EPS):.1f}dB"
            self.q.put({'type': 'log', 'message': log_msg})
            self.q.put({'type': 'plot_route', 'route_pos': visited_pos})
            current = best_nbr

            if distance(current.pos, dest_pos) < p['TX_RANGE'] or current.depth <= 5.0:
                self.q.put({'type': 'draw_radius', 'pos': current.pos, 'radius': p['TX_RANGE']})
                time.sleep(0.75)
                buoy_node = Node(node_id='Buoy', pos=dest_pos, channel_state=np.array([False]*M_SUBCARRIERS))
                best_final_td, _, best_final_snr = float('inf'), -1, 0
                idle_idxs = np.where(~current.channel_state)[0]
                if len(idle_idxs) == 0: idle_idxs = [0] 
                for idx in idle_idxs:
                    ch_idx = idx % len(channels_khz)
                    final_td, _, final_snr = compute_transmission_delay_paper(current, buoy_node, dest_pos, channels_khz[ch_idx], LP, CH_BANDWIDTH, p['PT_LINEAR'])
                    if final_td < best_final_td: best_final_td, _, best_final_snr = final_td, ch_idx, final_snr
                
                total_delay += best_final_td; route.append("Buoy"); visited_pos.append(dest_pos)
                final_log = f"→ Buoy | D=0.0m | TD={best_final_td:.4f}s | SNR={10*np.log10(best_final_snr+EPS):.1f}dB"
                self.q.put({'type': 'log', 'message': final_log})
                self.q.put({'type': 'plot_route', 'route_pos': visited_pos})
                self.q.put({'type': 'log', 'message': "\n✅ Reached surface buoy!"})
                break
        
        self.q.put({'type': 'log', 'message': f"\nFinal Route: {' → '.join(map(str, route))}"})
        self.q.put({'type': 'log', 'message': f"Total Delay: {total_delay:.4f} s"})
        self.q.put({'type': 'finished'})

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()