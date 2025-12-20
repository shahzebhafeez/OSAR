import numpy as np
import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_LOG_PATH = os.path.join(BASE_DIR, "detailed_trace_log.txt")
SUMMARY_LOG_PATH = os.path.join(BASE_DIR, "simulation_progress.csv")
GRAPH_DIR = os.path.join(BASE_DIR, "ml", "Graphs")

# --- META ---
SEED = 42
ENABLE_TRACE_LOGGING = True
NUM_CORES_TO_USE = 4
DEFAULT_NUM_PACKETS = 50   

# --- PHYSICS ---
CH_BANDWIDTH = 6e3
START_FREQ_KHZ = 10
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 10
P_BUSY = 0.3 

# --- ENV ---
TEMP = 10.0
SALINITY = 35.0
K_SPREAD = 1.5
EPS = 1e-12

# --- TRANSMISSION ---
LP = 512
CONTROL_PACKET_LP = 64
DEFAULT_PT_DB = 150.0
TX_RANGE = 250.0
SAFETY_MARGIN = 10.0

# --- ENERGY (JOULES) ---
CONST_EC = 0.5         # Reduced Comm Overhead
CONST_EF = 50.0        # Flight Overhead
CONST_E_IDLE = 0.03
E_BIT_TX = 0.01       
E_MOVE_FIXED = 50.0

# --- WEIGHTING FACTORS (Fixes AttributeError) ---
DEFAULT_W_TIME = 0.3   # <--- ADDED
DEFAULT_W_ENERGY = 0.7  # <--- ADDED

# --- AGENTS ---
AREA = 500.0
NODE_DRIFT_MIN = 1.0
NODE_DRIFT_MAX = 5.0
NODE_MAX_DRIFT_M = 5.0     # <--- ADDED (Alias for compatibility)
FIXED_AUVS = 7
DEFAULT_NUM_AUVS = 7       
DEFAULT_N_NODES = 30       

AUV_COVERAGE_RADIUS = 250.0
AUV_RELAY_RADIUS = 50.0
AUV_MIN_SPEED = 1.0
AUV_MAX_SPEED = 5.0
AUV_UPDATE_INTERVAL_S = 0.1
SVM_UPDATE_PERIOD_S = 5.0

# --- SWEEP ---
SWEEP_STATIC_NODES = [8, 23, 38, 53]
SWEEP_DURATION_S = 1200
SWEEP_PACKET_INTERVAL_S = 15
MONTE_CARLO_RUNS = 20

SCENARIOS = [
    ('DTC with EE-AURS', True, 'DTC'),
    ('SVM with EE-AURS', True, 'SVM'),
    ('RF with EE-AURS', True, 'RF'),
    ('Without EE-AURS', True, 'BASELINE')
]

PLOT_COLORS = {
    'Without EE-AURS': '#E63946',      # Red
    'SVM with EE-AURS': '#1D3557',     # Dark Blue
    'RF with EE-AURS': '#2A9D8F',      # Teal
    'DTC with EE-AURS': '#E9C46A'      # Gold
}