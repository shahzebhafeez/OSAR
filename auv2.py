# auv.py
# Contains the AUV class and related functions for the OSAR simulation.

import numpy as np
import time # For timestamping collected data

# --- AUV Parameters ---
AUV_COVERAGE_RADIUS = 75.0
AUV_MIN_SPEED = 10.0  # m/s
AUV_MAX_SPEED = 20.0  # m/s
AUV_WAYPOINTS = 5    # Number of waypoints in a random route
DEFAULT_AUV_RELAY_RADIUS = 50.0 # Distance within which AUV can relay to surface

# --- Utility Functions ---
def distance(a, b):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def generate_auv_route(area, num_waypoints, rng):
    """Generates a random route (list of 3D waypoints) for an AUV."""
    route = []
    # Start near the bottom for more realistic data muling
    start_pos = rng.uniform(0, area, 3)
    start_pos[2] = rng.uniform(area*0.7, area) # Start deeper
    route.append(start_pos)

    for _ in range(num_waypoints - 2): # Intermediate waypoints
        waypoint = rng.uniform(0, area, 3)
        route.append(waypoint)

    # End near the surface station
    end_pos = rng.uniform(area*0.4, area*0.6, 3) # Near center horizontally
    end_pos[2] = rng.uniform(0, area*0.1) # Near surface vertically
    route.append(end_pos)

    return np.array(route)

# --- AUV CLASS ---
class AUV:
    """Manages the state, movement, data collection, and relay of a single AUV."""
    def __init__(self, id, speed, route, surface_station_pos,
                 coverage_radius=AUV_COVERAGE_RADIUS,
                 relay_radius=DEFAULT_AUV_RELAY_RADIUS):
        self.id = id
        self.speed = speed
        self.route = route
        self.surface_station_pos = np.array(surface_station_pos)
        self.coverage_radius = coverage_radius
        self.relay_radius = relay_radius

        self.current_pos = np.array(self.route[0])
        self.target_waypoint_idx = 1
        self.covered_nodes = set() # Nodes visited
        self.traveled_path = [np.array(self.route[0])]

        # --- Data Handling Attributes ---
        self.data_buffer = {} # Stores collected data, e.g., {node_id: timestamp}
        self.relayed_data_log = {} # Keeps track of data successfully relayed {node_id: relay_timestamp}

        # --- Controller Status Flags (NEW) ---
        self.is_lc = False
        self.is_oc = False

        # Store artists associated with this AUV (managed by the GUI)
        self.path_artist = None
        self.marker_artist = None

    def collect_data(self, node_id):
        """Simulate collecting data from a node."""
        if node_id not in self.data_buffer:
            collect_time = time.time()
            self.data_buffer[node_id] = collect_time
            # print(f"[AUV {self.id}]: Collected data from Node {node_id} at {collect_time:.1f}")
            return True # Return True if new data was collected
        return False # Data already in buffer

    def relay_data(self):
        """Simulate relaying collected data to the surface station."""
        if not self.data_buffer:
            return None # Nothing to relay

        relay_time = time.time()
        relayed_node_ids = list(self.data_buffer.keys())

        # Log relayed data and clear buffer
        for node_id in relayed_node_ids:
            self.relayed_data_log[node_id] = relay_time # Log successful relay
        self.data_buffer.clear()

        # print(f"[AUV {self.id}]: Relayed data for nodes {relayed_node_ids} at {relay_time:.1f}")
        return relayed_node_ids # Return list of nodes whose data was relayed

    def update(self, dt, nodes):
        """Move the AUV, check coverage, collect data, and check for relay opportunity."""
        newly_covered_node_id = None
        relayed_node_ids_this_tick = None
        is_moving = True # Flag to check if AUV is still active

        # --- Movement Logic ---
        if self.target_waypoint_idx >= len(self.route):
            is_moving = False # Reached end of route
        else:
            target_pos = np.array(self.route[self.target_waypoint_idx])
            direction = target_pos - self.current_pos
            dist_to_target = np.linalg.norm(direction)
            epsilon_dist = 1e-6

            if dist_to_target < epsilon_dist:
                 self.target_waypoint_idx += 1
                 if self.target_waypoint_idx >= len(self.route):
                      is_moving = False
                 else: # Recalculate for new target
                     target_pos = np.array(self.route[self.target_waypoint_idx])
                     direction = target_pos - self.current_pos
                     dist_to_target = np.linalg.norm(direction)
                     if dist_to_target < epsilon_dist:
                          print(f"Warning: AUV {self.id} skipping duplicate waypoint.")
                          is_moving = False # Stop if stuck on duplicate

            if is_moving: # Only move if still active and not stuck
                move_dist = self.speed * dt
                if dist_to_target <= move_dist:
                    self.current_pos = target_pos
                    self.target_waypoint_idx += 1
                    if self.target_waypoint_idx >= len(self.route):
                         is_moving = False # Mark as finished after reaching final WP
                elif dist_to_target > epsilon_dist: # Avoid division by zero
                    move_vec = (direction / dist_to_target) * move_dist
                    self.current_pos = self.current_pos + move_vec

            self.traveled_path.append(np.copy(self.current_pos))

        # --- Coverage Check & Data Collection ---
        collected_new_data_flag = False
        for node in nodes:
            if distance(self.current_pos, node.pos) <= self.coverage_radius:
                if node.node_id not in self.covered_nodes:
                     newly_covered_node_id = node.node_id
                     self.covered_nodes.add(node.node_id) # Log visit

                if self.collect_data(node.node_id):
                    collected_new_data_flag = True


        # --- Relay Check ---
        if distance(self.current_pos, self.surface_station_pos) <= self.relay_radius:
            relayed_node_ids_this_tick = self.relay_data()

        # Return status: newly covered node ID, list of relayed node IDs, movement status, data collected
        return newly_covered_node_id, relayed_node_ids_this_tick, is_moving, collected_new_data_flag