# train_model_rf_aggressive.py
# Aggressively regularized to prevent perfect scores on synthetic data.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# --- Model Parameters ---
FEATURES = ['auv_pos_x', 'auv_pos_y', 'auv_pos_z', 'auv_speed', 'nodes_in_range']
DATA_FILE = "Dataset/auv_training_data_10_min.csv"

def train_lc_model_rf():
    print("--- Training LC Selection Model (RF) - AGGRESSIVE REGULARIZATION ---")
    
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

    # --- AGGRESSIVE REGULARIZATION ---
    print("Training LC Random Forest...")
    lc_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        
        # 1. Force shallow trees (Model cannot learn complex rules)
        max_depth=5,     
        
        # 2. Require huge groups (Model cannot learn specific points)
        min_samples_leaf=50,  
        
        # 3. Blind the trees (Each tree only sees 50% of features)
        max_features=0.5,
        
        verbose=1
    )
    
    lc_rf.fit(X_train_scaled, y_train)

    print("\nTraining complete.")
    y_pred = lc_rf.predict(X_test_scaled)
    print("\nLC Model Evaluation Report (Aggressive):")
    print(classification_report(y_test, y_pred))

    joblib.dump(lc_rf, "lc_model_rf.joblib")
    joblib.dump(lc_scaler, "lc_scaler_rf.joblib")
    return True

def train_oc_model_rf():
    print("\n--- Training OC Selection Model (RF) - AGGRESSIVE REGULARIZATION ---")
    
    if not os.path.exists(DATA_FILE):
        return False
        
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y_oc = df['is_oc']

    if len(y_oc.unique()) < 2:
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y_oc, test_size=0.2, random_state=42, stratify=y_oc)

    oc_scaler = StandardScaler()
    X_train_scaled = oc_scaler.fit_transform(X_train)
    X_test_scaled = oc_scaler.transform(X_test)

    # --- AGGRESSIVE REGULARIZATION ---
    print("Training OC Random Forest...")
    oc_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        
        # Even stricter for OC since it was 1.00 previously
        max_depth=4,            # Very shallow
        min_samples_leaf=60,    # Very smooth decision boundaries
        max_features=0.4,       # Only sees 40% of features
        
        verbose=1
    )
    
    oc_rf.fit(X_train_scaled, y_train)

    print("\nTraining complete.")
    y_pred = oc_rf.predict(X_test_scaled)
    print("\nOC Model Evaluation Report (Aggressive):")
    print(classification_report(y_test, y_pred))

    joblib.dump(oc_rf, "oc_model_rf.joblib")
    joblib.dump(oc_scaler, "oc_scaler_rf.joblib")
    return True

if __name__ == "__main__":
    if train_lc_model_rf():
        train_oc_model_rf()