# data_generator.py
# Headless simulation for generating AUV training data.

import sys 
import os 

# --- Path correction logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of new logic ---

import numpy as np
import pandas as pd
import time
from auv2 import AUV, generate_auv_route, distance, AUV_MIN_SPEED, AUV_MAX_SPEED, AUV_WAYPOINTS

# --- Simulation Parameters ---
# --- CHANGED: Run for 10 minutes (10 / 60.0 hours) ---
SIMULATION_HOURS = 10 / 60.0 
AREA = 500.0                 # Same as main.py
N_NODES = 30                 # Same as main.py
NUM_AUVS = 10                # Use more AUVs for diverse data
NUM_LCS_TO_SELECT = 3        # The "ground truth" number of LCs to label
NODE_DRIFT_SPEED = 0.5       # m/s, how fast nodes drift
SIM_TICK_RATE = 0.1          # AUV update interval (same as main.py)
RNG_SEED = 42                # For reproducibility

# --- Node Class (simplified for data collection) ---
class SensorNode:
    def __init__(self, id, pos):
        self.node_id = id
        self.pos = np.array(pos)

def update_node_positions(nodes, drift_speed, dt, area, rng):
    """Simulates simple random drift for sensor nodes."""
    for node in nodes:
        # Generate a random 3D drift vector
        drift = rng.normal(scale=drift_speed * dt, size=3)
        # Apply drift, ensuring Z (depth) doesn't go above 0
        node.pos += drift
        node.pos[2] = min(node.pos[2], AREA) # Don't go below 500m
        node.pos[2] = max(node.pos[2], 0.1) # Don't surface
        # Keep X and Y in bounds
        node.pos[0:2] = np.clip(node.pos[0:2], 0, area)

def get_node_density(auv, nodes):
    """Calculates the node density feature for an AUV."""
    count = 0
    for node in nodes:
        if distance(auv.current_pos, node.pos) < auv.coverage_radius:
            count += 1
    return count

def run_data_collection():
    """Runs the headless simulation and saves data to a CSV."""
    print(f"Starting data collection for {SIMULATION_HOURS * 60.0:.0f} simulated minutes...")
    
    rng = np.random.default_rng(RNG_SEED)
    
    # 1. Initialize Sensor Nodes
    # Using a simple grid placement for nodes
    node_positions = np.array([rng.uniform(0, AREA, 3) for _ in range(N_NODES)])
    nodes = [SensorNode(i, pos) for i, pos in enumerate(node_positions)]
    
    # 2. Initialize AUVs
    surface_station_pos = np.array([AREA/2, AREA/2, 0.0]) # Dummy position
    auvs = []
    for i in range(NUM_AUVS):
        speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
        route = generate_auv_route(AREA, AUV_WAYPOINTS, rng)
        auv = AUV(i, speed, route, surface_station_pos)
        auvs.append(auv)
        
    # 3. Simulation Loop
    all_data = []
    sim_start_time = time.time()
    num_ticks = int((SIMULATION_HOURS * 3600) / SIM_TICK_RATE)
    
    print(f"Total ticks to simulate: {num_ticks}") # This will now be 6,000

    for tick in range(num_ticks):
        # Update node positions to simulate drift
        update_node_positions(nodes, NODE_DRIFT_SPEED, SIM_TICK_RATE, AREA, rng)
        
        # Collect features for all AUVs at this timestep
        features_this_tick = []
        for auv in auvs:
            # Update AUV movement
            auv.update(SIM_TICK_RATE, nodes) # We don't care about return values here
            
            # Get features
            density = get_node_density(auv, nodes)
            features_this_tick.append({
                "auv_id": auv.id,
                "auv_pos_x": auv.current_pos[0],
                "auv_pos_y": auv.current_pos[1],
                "auv_pos_z": auv.current_pos[2],
                "auv_speed": auv.speed,
                "nodes_in_range": density
            })
            
        # --- Generate "Ground Truth" Labels (Heuristic) ---
        # Sort AUVs by density to find the best ones
        features_this_tick.sort(key=lambda x: x['nodes_in_range'], reverse=True)
        
        # Label LCs (top N)
        for i, data in enumerate(features_this_tick):
            data['is_lc'] = 1 if i < NUM_LCS_TO_SELECT else 0
        
        # Label OC (the single best LC)
        for i, data in enumerate(features_this_tick):
            data['is_oc'] = 1 if i == 0 and data['is_lc'] == 1 else 0
            
        # Add timestamp and append to master list
        timestamp = tick * SIM_TICK_RATE
        for data in features_this_tick:
            data['timestamp'] = timestamp
            all_data.append(data)
            
        if tick % 1000 == 0:
            print(f"Simulating tick {tick}/{num_ticks}...")
            
    # 4. Save to CSV
    df = pd.DataFrame(all_data)
    # Save in the same directory as the script
    output_path = os.path.join(script_dir, "auv_training_data_10_min.csv")
    df.to_csv(output_path, index=False)
    
    print("\nData collection complete.")
    print(f"Saved {len(df)} samples to '{output_path}'.") # Will be 6,000 ticks * 10 AUVs = 60,000 samples
    print(f"Total real-world time elapsed: {time.time() - sim_start_time:.2f} s")

if __name__ == "__main__":
    run_data_collection()