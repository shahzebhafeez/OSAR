# train_model.py
# Loads data, trains SVM models, and SHOWS PROGRESS.

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib # For saving models
import os # To check if file exists

# --- Model Parameters ---
# Features to use for training
FEATURES = ['auv_pos_x', 'auv_pos_y', 'auv_pos_z', 'auv_speed', 'nodes_in_range']
DATA_FILE = "auv_training_data.csv"

def train_lc_model():
    """Trains the SVM to classify AUVs as LCs or non-LCs."""
    print("--- Training LC Selection Model (Task 1: LC vs. non-LC) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print("Please run 'data_collector.py' first.")
        return False
        
    df = pd.read_csv(DATA_FILE)

    # 2. Define Features (X) and Label (y) from *all* data
    X = df[FEATURES]
    y_lc = df['is_lc'] # Label: 1 if LC, 0 if not
    
    if len(X) == 0:
        print("Error: No data found in CSV.")
        return False
        
    print(f"Loaded {len(X)} total samples.")
    print(f"LC class distribution:\n{y_lc.value_counts(normalize=True)}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_lc, test_size=0.2, random_state=42)

    # 4. Scale Features
    lc_scaler = StandardScaler()
    X_train_scaled = lc_scaler.fit_transform(X_train)
    X_test_scaled = lc_scaler.transform(X_test)

    # 5. Train SVM (RBF Kernel, balanced weights for uneven classes)
    print("Training LC SVM (kernel='rbf')...")
    
    # --- MODIFIED: Added verbose=True ---
    lc_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, verbose=True)
    
    lc_svm.fit(X_train_scaled, y_train)

    # 6. Evaluate Model
    print("\nTraining complete.")
    y_pred = lc_svm.predict(X_test_scaled)
    print("\nLC Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model and Scaler
    joblib.dump(lc_svm, "lc_model.joblib")
    joblib.dump(lc_scaler, "lc_scaler.joblib")
    print("Saved 'lc_model.joblib' and 'lc_scaler.joblib'.")
    return True

def train_oc_model():
    """
    Trains the SVM to classify AUVs as the OC or not_OC,
    using the *entire* dataset.
    """
    print("\n--- Training OC Selection Model (Task 2: OC vs. non-OC) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        return False
        
    df = pd.read_csv(DATA_FILE)
    
    # 2. Define Features (X) and Label (y) from *all* data
    X = df[FEATURES]
    y_oc = df['is_oc'] # Label: 1 if OC, 0 if not
    
    if len(X) == 0:
        print("Error: No data found to train OC model.")
        return False
        
    print(f"Loaded {len(X)} total samples for OC training.")
    print(f"OC class distribution:\n{y_oc.value_counts(normalize=True)}")
    
    # Check if we have both classes (OC and non-OC)
    if len(y_oc.unique()) < 2:
        print("Error: The dataset does not contain any OC samples (is_oc == 1).")
        print("Please check 'data_collector.py' logic.")
        return False

    # 3. Split Data (stratify is important for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_oc, test_size=0.2, random_state=42, stratify=y_oc)

    # 4. Scale Features
    oc_scaler = StandardScaler()
    X_train_scaled = oc_scaler.fit_transform(X_train)
    X_test_scaled = oc_scaler.transform(X_test)

    # 5. Train SVM (RBF Kernel)
    print("Training OC SVM (kernel='rbf')...")
    
    # --- MODIFIED: Added verbose=True ---
    # class_weight='balanced' is critical here due to high imbalance
    oc_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, verbose=True)
    
    oc_svm.fit(X_train_scaled, y_train)

    # 6. Evaluate Model
    print("\nTraining complete.")
    y_pred = oc_svm.predict(X_test_scaled)
    print("\nOC Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model and Scaler
    joblib.dump(oc_svm, "oc_model.joblib")
    joblib.dump(oc_scaler, "oc_scaler.joblib")
    print("Saved 'oc_model.joblib' and 'oc_scaler.joblib'.")
    return True

if __name__ == "__main__":
    if train_lc_model():
        train_oc_model()