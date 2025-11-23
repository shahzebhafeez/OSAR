# simulation_config.py
# Global parameters for the OSAR + AUV simulation.

# --- OSAR Parameters ---
LP = 512               # Data packet size (bits)
CONTROL_PACKET_LP = 64 # Control packet size (bits)
CH_BANDWIDTH = 6e3     # 6 kHz bandwidth
DEFAULT_PT_DB = 150.0  # Transmit power (dB re uPa)
P_BUSY = 0.8           # Probability channel is busy
M_SUBCARRIERS = 128
DEFAULT_M_CHANNELS = 5
START_FREQ_KHZ = 10
DEFAULT_NUM_PACKETS = 10

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
E_AUV_MOVE_PER_S = 25_000.0 
AUV_HOP_ENERGY_PENALTY = 25_000.0 
DEFAULT_W_TIME = 0.5  
DEFAULT_W_ENERGY = 0.5 

# --- Environmental Parameters ---
TEMP = 10
SALINITY = 35
K_SPREAD = 1.5