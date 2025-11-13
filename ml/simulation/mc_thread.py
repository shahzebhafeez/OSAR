# mc_thread.py
# --- Logic for the MC (Machine Learning Controller) simulation thread ---

import time
import numpy as np
import traceback

# Import shared components from common.py
from .common import auv_distance, SVM_UPDATE_PERIOD_S

# ML imports are specific to this file
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("Error: scikit-learn or joblib not found.")
    print("Please install it: pip install scikit-learn joblib")
    # This module won't run, but the GUI will handle the error.


def _collect_auv_features(auv_list, node_list):
    """Internal helper to gather features from the current AUV state."""
    features = []
    auv_map = {}
    
    # Use list() to create a snapshot for thread-safe iteration
    current_auv_list = list(auv_list) 
    if not node_list or not current_auv_list:
        return None, None
        
    for i, auv in enumerate(current_auv_list):
        count = 0
        for node in node_list:
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

def _update_controllers_svm(q, auv_list, node_list, models):
    """Internal helper to run SVM predictions and update AUV roles."""
    
    q.put({'type': 'log', 'message': "[MC]: Waking up to update controllers..."})
    
    X_features, auv_map = _collect_auv_features(auv_list, node_list)
    
    if X_features is None or X_features.shape[0] == 0:
        q.put({'type': 'log', 'message': "[MC]: AUVs/Nodes not ready. Skipping update."})
        return
        
    # Reset all AUVs (operating on the shared list)
    for auv in auv_list:
        auv.is_lc = False
        auv.is_oc = False
        
    # --- 1. Unpack Models ---
    lc_model = models['lc_model']
    lc_scaler = models['lc_scaler']
    oc_model = models['oc_model']
    oc_scaler = models['oc_scaler']

    # --- 2. Predict LCs ---
    X_scaled_lc = lc_scaler.transform(X_features)
    lc_predictions = lc_model.predict(X_scaled_lc)
    
    selected_lc_ids = set()
    for i, pred in enumerate(lc_predictions):
        if pred == 1:
            auv = auv_map[i]
            auv.is_lc = True
            selected_lc_ids.add(auv.id)

    # --- 3. Predict OCs ---
    X_scaled_oc = oc_scaler.transform(X_features)
    oc_predictions = oc_model.predict(X_scaled_oc)
    
    selected_oc_id = None
    if np.sum(oc_predictions) > 0:
        oc_probabilities = oc_model.predict_proba(X_scaled_oc)[:, 1]
        best_oc_index = -1
        max_prob = -1
        
        for i, pred in enumerate(oc_predictions):
            if pred == 1:
                if oc_probabilities[i] > max_prob:
                    max_prob = oc_probabilities[i]
                    best_oc_index = i
                        
        if best_oc_index != -1:
            auv = auv_map[best_oc_index]
            auv.is_oc = True
            auv.is_lc = True # Post-processing: An OC is always an LC
            selected_oc_id = auv.id
            if auv.id not in selected_lc_ids:
                selected_lc_ids.add(auv.id) 

    q.put({'type': 'log', 'message': f"[MC]: LCs updated: {selected_lc_ids or 'None'}"})
    q.put({'type': 'log', 'message': f"[MC]: OC updated: {selected_oc_id or 'None'}"})
    
    # Trigger a plot update to show new AUV roles (colors/markers)
    q.put({'type': 'plot_auvs', 'auv_indices': list(range(len(auv_list)))})


def run_mc_logic(q, simulation_running_event, p, global_auv_list, global_node_list, models):
    """
    Runs the MC logic thread, periodically updating AUV controller roles.
    
    Args:
        q: The main message queue to send updates to the GUI.
        simulation_running_event: The threading.Event to signal when to stop.
        p: The dictionary of simulation parameters.
        global_auv_list: The shared list of AUVs (self.auvs).
        global_node_list: The shared list of Nodes (self.sim_nodes).
        models: A dictionary containing the loaded SVM models and scalers.
    """
    
    if not all(models.values()):
        q.put({'type': 'log', 'message': "[MC Sim]: Models not loaded. MC thread exiting."})
        return
        
    q.put({'type': 'log', 'message': "[MC Sim]: MC logic thread started."})
    try:
        # Wait for AUV and Node lists to be populated by other threads
        while (len(global_auv_list) == 0 or len(global_node_list) == 0) and simulation_running_event.is_set():
            time.sleep(0.5)

        while simulation_running_event.is_set():
            time.sleep(SVM_UPDATE_PERIOD_S)
            if not simulation_running_event.is_set(): break
            
            # Call the internal helper function to do the work
            _update_controllers_svm(q, global_auv_list, global_node_list, models)
            
    except Exception as e:
        print(f"[MC Thread] Error: {e}")
        traceback.print_exc() # Print full MC error
        q.put({'type': 'log', 'message': f"[MC Sim] Error: {e}"})
    finally:
        print("[MC Thread] Finished.")