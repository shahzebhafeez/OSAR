# train_model_dtc_aggressive.py
# Aggressively regularized Decision Tree Classifier (DTC) to prevent perfect scores.
# This ensures the model generalizes rather than memorizing the synthetic topology.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# --- Model Parameters ---
FEATURES = ['auv_pos_x', 'auv_pos_y', 'auv_pos_z', 'auv_speed', 'nodes_in_range']
DATA_FILE = "Dataset/auv_training_data_10_min.csv"

def train_lc_model_dtc():
    print("--- Training LC Selection Model (DTC) - AGGRESSIVE REGULARIZATION ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        return False
        
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y_lc = df['is_lc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_lc, test_size=0.2, random_state=42)

    # Scaling
    lc_scaler = StandardScaler()
    X_train_scaled = lc_scaler.fit_transform(X_train)
    X_test_scaled = lc_scaler.transform(X_test)

    # --- AGGRESSIVE REGULARIZATION FOR DECISION TREE ---
    print("Training LC Decision Tree...")
    lc_dtc = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42,
        
        # 1. Force Pruning / Shallow Tree
        # A low max_depth prevents the tree from creating complex, specific logic rules.
        max_depth=5,     
        
        # 2. Require Large Leaf Groups
        # High min_samples_leaf forces the model to make broader generalizations
        # rather than isolating specific data points.
        min_samples_leaf=50,  
        
        # 3. Feature Blinding
        # Only looking at ~60% of features per split reduces correlation
        max_features=0.6 
    )
    
    lc_dtc.fit(X_train_scaled, y_train)

    print("\nTraining complete.")
    y_pred = lc_dtc.predict(X_test_scaled)
    print("\nLC Model Evaluation Report (DTC Aggressive):")
    print(classification_report(y_test, y_pred))

    joblib.dump(lc_dtc, "lc_model_dtc.joblib")
    joblib.dump(lc_scaler, "lc_scaler_dtc.joblib")
    print("Saved 'lc_model_dtc.joblib' and 'lc_scaler_dtc.joblib'")
    return True

def train_oc_model_dtc():
    print("\n--- Training OC Selection Model (DTC) - AGGRESSIVE REGULARIZATION ---")
    
    if not os.path.exists(DATA_FILE):
        return False
        
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y_oc = df['is_oc']

    if len(y_oc.unique()) < 2:
        print("Error: Not enough classes for OC training.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y_oc, test_size=0.2, random_state=42, stratify=y_oc)

    oc_scaler = StandardScaler()
    X_train_scaled = oc_scaler.fit_transform(X_train)
    X_test_scaled = oc_scaler.transform(X_test)

    # --- AGGRESSIVE REGULARIZATION FOR DECISION TREE ---
    print("Training OC Decision Tree...")
    oc_dtc = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42,
        
        # Stricter regularizations for OC (Ordinary Cluster)
        max_depth=4,            # Very shallow depth
        min_samples_leaf=60,    # Requires very smooth boundaries
        max_features=0.5        # Only sees 50% of features at split
    )
    
    oc_dtc.fit(X_train_scaled, y_train)

    print("\nTraining complete.")
    y_pred = oc_dtc.predict(X_test_scaled)
    print("\nOC Model Evaluation Report (DTC Aggressive):")
    print(classification_report(y_test, y_pred))

    joblib.dump(oc_dtc, "oc_model_dtc.joblib")
    joblib.dump(oc_scaler, "oc_scaler_dtc.joblib")
    print("Saved 'oc_model_dtc.joblib' and 'oc_scaler_dtc.joblib'")
    return True

if __name__ == "__main__":
    if train_lc_model_dtc():
        train_oc_model_dtc()