# auv_thread.py
# --- Logic for the AUV (movement and data muling) simulation thread ---

import time
import numpy as np
import traceback

# Import shared components from common.py
from .common import (
    AUV, generate_auv_route, AUV_MIN_SPEED, AUV_MAX_SPEED,
    AUV_WAYPOINTS, AUV_COVERAGE_RADIUS, AUV_UPDATE_INTERVAL_S
)

def run_auv_simulation(q, simulation_running_event, p, global_auv_list, global_node_list):
    """
    Runs the AUV movement and data muling simulation.
    
    Args:
        q: The main message queue to send updates to the GUI.
        simulation_running_event: The threading.Event to signal when to stop.
        p: The dictionary of simulation parameters.
        global_auv_list: The shared list (self.auvs) to populate with
                         the AUVs created in this simulation.
        global_node_list: The shared list (self.sim_nodes) which this
                          thread reads to check for node coverage.
    """
    q.put({'type': 'log', 'message': "[AUV Sim]: AUV simulation thread started."})
    
    surface_station_pos = p['SURFACE_STATION_POS']
    auv_relay_radius = p['AUV_RELAY_RADIUS']
    total_auv_move_time = 0.0
    
    try:
        # Wait until the OSAR thread has created the nodes
        while len(global_node_list) == 0 and simulation_running_event.is_set():
            time.sleep(0.1)
        if not simulation_running_event.is_set(): return

        seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
        rng = np.random.default_rng(seed + 1) # Use a different seed from OSAR
        
        temp_auv_list = []
        for i in range(p['NUM_AUVS']):
            speed = rng.uniform(AUV_MIN_SPEED, AUV_MAX_SPEED)
            route = generate_auv_route(p['AREA'], AUV_WAYPOINTS, rng)
            auv = AUV(i, speed, route, surface_station_pos,
                      coverage_radius=AUV_COVERAGE_RADIUS,
                      relay_radius=auv_relay_radius)
            temp_auv_list.append(auv)
            q.put({'type': 'log', 'message': f"[AUV {i}]: Created. Speed: {speed:.1f} m/s, Relay Radius: {auv_relay_radius}m"})
        
        # --- CRITICAL FIX ---
        # Populate the shared AUV list so the MC thread can see them
        global_auv_list.clear()
        global_auv_list.extend(temp_auv_list)
        # --- END FIX ---
        
        last_plot_time = 0
        while simulation_running_event.is_set():
            auv_indices_updated = []
            all_auvs_finished_route = True
            
            num_moving_auvs = 0
            # Iterate over the shared list.
            # Use list() for a thread-safe-ish snapshot during iteration
            for idx, auv in enumerate(list(global_auv_list)): 
                if auv.target_waypoint_idx < len(auv.route):
                    newly_covered_node_id, relayed_ids, is_moving, collected_new = auv.update(
                        AUV_UPDATE_INTERVAL_S, global_node_list # Pass node list
                    )
                    
                    if collected_new:
                        q.put({'type': 'log', 'message': f"[AUV {auv.id}]: Collected data from Node {newly_covered_node_id}"})
                    if relayed_ids is not None:
                       q.put({'type': 'log', 'message': f"[AUV {auv.id}]: Relayed data for nodes {relayed_ids} near surface."})
                    
                    auv_indices_updated.append(idx)
                    
                    if is_moving:
                        all_auvs_finished_route = False
                        num_moving_auvs += 1
            
            total_auv_move_time += (num_moving_auvs * AUV_UPDATE_INTERVAL_S)

            current_time = time.time()
            # Send plot update to sync with MC changes
            if (current_time - last_plot_time > 0.2):
                q.put({'type': 'plot_auvs', 'auv_indices': list(range(len(global_auv_list)))})
                last_plot_time = current_time

            if all_auvs_finished_route:
                q.put({'type': 'log', 'message': "[AUV Sim]: All AUVs have completed their routes."})
                # Keep running to allow MC logic to function, but stop moving.
                # To stop the sim, we'd break, but we'll let the OSAR thread finish.
                pass 

            time.sleep(AUV_UPDATE_INTERVAL_S)

    except Exception as e:
        print(f"[AUV Thread] Error: {e}")
        traceback.print_exc()
        q.put({'type': 'log', 'message': f"[AUV Sim] Error: {e}"})
    finally:
        final_auv_report = {}
        if global_auv_list:
            q.put({'type': 'log', 'message': "\n--- AUV FINAL STATS ---"})
            total_unique_relayed = set()
            for auv in global_auv_list:
                report_key = f"--- AUV {auv.id} Report ---"
                final_auv_report[report_key] = [
                    f"   Speed: {auv.speed:.2f} m/s",
                    f"   Nodes Visited: {sorted(list(auv.covered_nodes)) or 'None'}",
                    f"   Data Relayed (Node IDs): {sorted(list(auv.relayed_data_log.keys())) or 'None'}"
                ]
                for line in final_auv_report[report_key]:
                    q.put({'type': 'log', 'message': line})
                total_unique_relayed.update(auv.relayed_data_log.keys())
            
            q.put({'type': 'log', 'message': f"\nTotal Unique Nodes Relayed by all AUVs: {len(total_unique_relayed)}"})

        q.put({'type': 'auv_finished', 'data': {
            'total_auv_move_time': total_auv_move_time,
            'auv_reports': final_auv_report
        }})
        print("[AUV Thread] Finished.")