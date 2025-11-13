# osar_thread.py
# --- Logic for the OSAR (static node routing) simulation thread ---

import time
import numpy as np
import traceback

# Import shared components from common.py
from .common import (
    Node, place_nodes_uniform, compute_transmission_delay_paper,
    select_best_next_hop, auv_distance, M_SUBCARRIERS,
    LP, CONTROL_PACKET_LP, CH_BANDWIDTH, START_FREQ_KHZ, EPS
)

def run_osar_simulation(q, simulation_running_event, p, global_node_list):
    """
    Runs the OSAR packet routing simulation.
    
    Args:
        q: The main message queue to send updates to the GUI.
        simulation_running_event: The threading.Event to signal when to stop.
        p: The dictionary of simulation parameters.
        global_node_list: The shared list (self.sim_nodes) to populate with
                          the nodes created in this simulation.
    """
    seed = p['SEED'] if p['SEED'] != 0 else int(time.time())
    rng = np.random.default_rng(seed)
    q.put({'type': 'log', 'message': f"[OSAR Sim]: OSAR thread started with seed: {seed}"})
    
    surface_station_pos = p['SURFACE_STATION_POS']
    
    packets_generated = 0
    packets_delivered = 0
    total_data_bits = 0
    total_control_bits = 0
    total_end_to_end_delay = 0.0
    
    try:
        ch_bw_khz = CH_BANDWIDTH / 1000.0
        start_center_freq = START_FREQ_KHZ + (ch_bw_khz / 2.0)
        channels_khz = start_center_freq + np.arange(p["M_CHANNELS"])
        
        if p['N_NODES'] == 0:
            q.put({'type': 'log', 'message': "No nodes to simulate."})
            return

        # Create nodes
        positions = place_nodes_uniform(p['N_NODES'], p['AREA'], rng)
        nodes = [Node(i, positions[i], rng.random(M_SUBCARRIERS) < p['P_BUSY']) for i in range(p['N_NODES'])]
        dest_pos = surface_station_pos

        if not nodes:
            q.put({'type':'log', 'message':"No sensor nodes created."})
            return
            
        # --- CRITICAL FIX ---
        # Populate the shared node list so other threads (AUV, MC) can see them.
        global_node_list.clear()
        global_node_list.extend(nodes)
        # --- END FIX ---
        
        src = max(global_node_list, key=lambda n: n.depth)
        
        if src is None: 
            q.put({'type':'log', 'message':"Could not determine source node."})
            print("[OSAR Thread] Error: Source node is None.")
            return

        q.put({'type': 'log', 'message': f"[OSAR Sim]: Source: Node {src.node_id}, Depth={src.depth:.1f} m"})
        q.put({'type': 'plot_nodes', 'pos': np.array([n.pos for n in global_node_list]),
               'depths': [n.depth for n in global_node_list], 'src': src.pos, 'dest': dest_pos})
        time.sleep(0.5)

        packets_generated = p['NUM_PACKETS']

        for i in range(packets_generated):
            if not simulation_running_event.is_set(): break
            
            q.put({'type': 'log', 'message': f"\n--- [OSAR] Routing Packet {i+1}/{packets_generated} ---"})
            current, visited_pos = src, [src.pos]
            packet_delay, packet_delivered_flag = 0.0, False
            
            for hop_count in range(p['N_NODES'] + 5):
                if not simulation_running_event.is_set(): break
                
                # Update channel states for this time step
                for node in global_node_list: 
                    node.channel_state = rng.random(M_SUBCARRIERS) < p['P_BUSY']
                    
                total_control_bits += CONTROL_PACKET_LP
                
                best_nbr, TD, _, _ = select_best_next_hop(current, global_node_list, dest_pos, p['TX_RANGE'], channels_khz, LP, CH_BANDWIDTH, p['PT_LINEAR'])
                
                if best_nbr is None:
                    q.put({'type': 'log', 'message': f"   ⚠️ Node {current.node_id} is stuck. Dropping packet."})
                    break
                    
                total_control_bits += CONTROL_PACKET_LP
                total_data_bits += LP
                packet_delay += TD
                visited_pos.append(best_nbr.pos)
                q.put({'type': 'log', 'message': f"  {current.node_id} → {best_nbr.node_id} | Hop Delay: {TD:.4f}s"})
                current = best_nbr
                
                # Check for delivery to surface
                if auv_distance(current.pos, dest_pos) < p['TX_RANGE'] or current.depth <= 5.0:
                    buoy_node = Node(node_id='Buoy', pos=dest_pos, channel_state=np.array([False]*M_SUBCARRIERS))
                    
                    idle_final = np.where(~np.array(current.channel_state, dtype=bool))[0]
                    if not idle_final.any(): final_ch_idx = 0
                    else: final_ch_idx = idle_final[0] % len(channels_khz)
                    
                    final_td, _, _ = compute_transmission_delay_paper(current, buoy_node, dest_pos, channels_khz[final_ch_idx], LP, CH_BANDWIDTH, p['PT_LINEAR'])
                    
                    total_data_bits += LP
                    total_control_bits += 2 * CONTROL_PACKET_LP
                    packet_delay += final_td
                    visited_pos.append(dest_pos)
                    q.put({'type': 'log', 'message': f"  {current.node_id} → Buoy | Final Hop Delay: {final_td:.4f}s"})
                    
                    packets_delivered += 1
                    total_end_to_end_delay += packet_delay
                    packet_delivered_flag = True
                    break
                    
            q.put({'type': 'plot_route', 'route_pos': visited_pos, 'success': packet_delivered_flag})
            time.sleep(0.1)
        
    except Exception as e:
        print(f"--- [OSAR THREAD FATAL ERROR] ---")
        print(f"--- [OSAR Thread] Error: {e} ---")
        traceback.print_exc()
        q.put({'type': 'log', 'message': f"[OSAR Sim] FATAL ERROR: {e}"})
        time.sleep(1.0) 
        
    finally:
        q.put({'type': 'osar_finished', 'data': {
            'params': p,
            'packets_generated': packets_generated,
            'packets_delivered': packets_delivered,
            'total_data_bits': total_data_bits,
            'total_control_bits': total_control_bits,
            'total_end_to_end_delay': total_end_to_end_delay
        }})
        print("[OSAR Thread] Finished.")