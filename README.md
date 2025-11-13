
---

## 🚀 Overview

This project simulates an **Ocean Sensor and AUV Relay (OSAR)** network that uses **machine learning** for dynamic selection of **Local Controllers (LCs)** and **Ocean Controllers (OCs)** among multiple AUVs (Autonomous Underwater Vehicles).

The system runs a **multi-threaded** simulation with a GUI for visualization and real-time logging.

### Key Features
- Multi-threaded simulation (OSAR, AUV, and MC threads)
- Real-time GUI logging and 3D plotting
- LC/OC classification via **SVM** and **Random Forest**
- Performance metrics: **PDR**, **ECR**, **RoR**, and **E2E Delay**
- Modular design (ML + simulation + GUI)

---

## ⚙️ Core Threads Overview

### 🧭 1. OSAR Packet Relaying Thread (`run_simulation_thread`)
Simulates packet forwarding among static underwater nodes until packets reach the surface (MC).

**Steps:**
1. Creates stationary nodes.
2. Selects deepest node as packet source.
3. Calls `select_best_next_hop()` to forward packets hop-by-hop.
4. Logs hop delays and delivery status.
5. Ends when all packets delivered or dropped.

**Metrics:**
- Hop Delay  
- E2E Delay  
- PDR (Delivered / Sent packets)

---

### 🤿 2. AUV Trajectory Thread (`run_auv_thread`)
Simulates AUVs moving **elliptically** between surface and depth, spaced horizontally to cover a 500×500×500 m³ region.

**Steps:**
1. Waits for OSAR nodes to initialize.
2. Creates `NUM_AUVS` instances using `auv2.py`.
3. Periodically updates AUV positions (`auv.update()`).
4. Sends updated 3D positions to the GUI.
5. Ends when all AUVs complete routes.

**Energy Model:**
- **ECR (Energy Cost Ratio)** = Flight + Communication energy  
- Powerbanks simulate recharge cycles

---

### 🧠 3. Machine Controller Thread (`run_mc_logic`)
The **Main Controller (MC)** classifies AUVs as LC or OC using trained ML models.

**Steps:**
1. Waits for both nodes and AUVs to initialize.
2. Every `SVM_UPDATE_PERIOD_S` seconds:
   - Collects features: position, speed, nearby nodes
   - Predicts LC/OC using `lc_model` and `oc_model`
   - Updates GUI colors (LC=cyan, OC=red)
3. Repeats until simulation ends.

---

## 🧵 Thread Orchestration Flow

| Phase | Description |
|-------|--------------|
| **1. Sequential** | Runs OSAR-Thread first |
| **2. Parallel** | Launches AUV-Thread + MC-Thread together |
| **3. Cleanup** | Joins all threads, computes metrics, re-enables GUI controls |

---

## 💬 Logging & Queuing System

| Component | Function |
|------------|-----------|
| **Logger** | Logs to `simulation_log.txt` + GUI console |
| **QueueHandler** | Sends messages between threads and GUI |
| **Queue Messages** | `log`, `plot_nodes`, `plot_route`, `plot_auvs`, `osar_finished`, `auv_finished` |
| **process_queue()** | GUI heartbeat — updates plots every 100ms |

---

## 📊 Performance Metrics

| Metric | Description |
|---------|-------------|
| **Precision / Recall / F1 / Accuracy** | ML classification results |
| **PDR (Packet Delivery Ratio)** | Received / Sent packets |
| **RoR (Routing Overhead Ratio)** | Control bits used for handshakes |
| **ECR (Energy Cost Ratio)** | Energy for motion + communication |
| **E2E Delay** | Avg. packet transmission time |

---

## 🧠 ML Models

Located in `/osar/ml/Results`  
Trained using **10-minute AUV data** (`auv_training_data_10_min.py`).

**Models Used:**
- SVM (RBF Kernel)
- Random Forest

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score

---

## 🧩 Key Functions (in `auv2.py`)

| Function | Description |
|-----------|--------------|
| `thorp_absorption_db_per_m()` | Computes acoustic signal absorption |
| `path_loss_linear()` | Combines absorption + geometric spreading loss |
| `noise_psd_linear()` | Models ambient noise (ship/wind) |
| `compute_transmission_delay_paper()` | Calculates delay using SNR & Shannon-Hartley theorem |
| `select_best_next_hop()` | Finds optimal neighbor node (lowest delay, valid range) |

---

## 🧩 Known Issues

- Nodes/AUVs not showing on GUI even though threads are running  
- MC thread not starting after AUV thread completion  
- Possible synchronization or flag handling issue between AUV and MC threads  

---

## 💡 Future Improvements

- Fix thread synchronization between AUV ↔ MC  
- Add live charts for metrics (PDR/ECR trends)  
- Integrate deep learning (e.g., BiLSTM) for better LC/OC prediction  
- Optimize 3D visualization rendering  

---

## 🖥️ How to Run

```bash
# 1. Go to the simulation folder
cd osar/ml

# 2. Run the main simulation file
python Simulation_fixed_v_9.py
