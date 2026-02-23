# OSAR-UW: Underwater Acoustic Networking & Protocol Simulator

### 📌 Project Overview
**OSAR-UW** is a high-fidelity discrete-event simulation platform designed to model the unique challenges of **Underwater Wireless Sensor Networks (UWSNs)**. Unlike terrestrial RF networks, underwater communication relies on acoustic waves characterized by **extreme propagation delays (1500m/s)**, high bit-error rates, and severely limited bandwidth. 

This simulator provides a cross-layer environment to test MAC and Routing protocols against the **Thorp Attenuation Model**, allowing researchers to visualize packet collisions and energy consumption in a 3D oceanic space.



---

### 🚀 Key Features

* **Physical Layer Realism:** Implements **Thorp’s Propagation Model** to calculate frequency-dependent attenuation and ambient noise (shipping, wind, and thermal).
* **High-Latency MAC Scheduling:** Specialized handling for the "Slotted-Prop" effect, where propagation delay often exceeds packet transmission time.
* **3D Mobility Support:** Simulates **Autonomous Underwater Vehicles (AUVs)** and drifting buoys using Gaussian Random Walk and depth-regulated movement.
* **Protocol Sandbox:** Includes pre-configured modules for:
    * **MAC:** Slotted ALOHA, UW-MAC.
    * **Routing:** Depth-Based Routing (DBR) and Vector-Based Forwarding (VBF).
* **Energy Analysis:** Tracks micro-joule consumption for Transmit, Receive, and Idle states to predict node longevity.

---

### 🛠 Tech Stack

* **Engine:** `Python 3.10+` / `SimPy` (Discrete-Event Simulation framework).
* **Math & Physics:** `NumPy` & `SciPy` (Signal-to-Noise Ratio (SNR) and Bit Error Rate (BER) calculations).
* **Visualization:** `Matplotlib 3D` (Node trajectory) & `Plotly` (Interactive packet-loss heatmaps).
* **Configuration:** `YAML` / `JSON` for environmental parameters (Salinity, Temperature, Depth).

---

### 🔌 System Architecture

| Component | Responsibility | Technical Implementation |
| :--- | :--- | :--- |
| **Acoustic Channel** | Signal Attenuation | Thorp Model & Passive Sonar Equation |
| **MAC Layer** | Collision Avoidance | Propagation-Delay Aware Scheduling |
| **Network Layer** | Data Forwarding | Depth-Based Routing (Pressure-Sensor Emulation) |
| **Mobility Manager** | Node Movement | 3D Coordinate Kinematics |

---

### 📊 Performance Metrics

The simulator outputs a comprehensive analytics suite including:
* **PDR (Packet Delivery Ratio):** Effectiveness of routing under varying "Turbid" conditions.
* **End-to-End Latency:** Detailed breakdown of processing vs. propagation delay.
* **Void Hole Analysis:** Identifies communication gaps caused by node sparsity or current-driven drift.

---

### ⚙️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/username/osar-uw-simulator.git](https://github.com/username/osar-uw-simulator.git)
Simulation Results : 


<img width="500" height="290" alt="Combined_PDR" src="https://github.com/user-attachments/assets/dac64f8f-1194-48c3-92a2-17b87e598f51" />

<img width="500" height="290" alt="Combined_ECR" src="https://github.com/user-attachments/assets/910ab1c6-bbf4-477b-a1f6-b35b59169d48" />

<img width="500" height="290" alt="Combined_RoR" src="https://github.com/user-attachments/assets/64ba86bb-f200-43e7-aceb-a27628466a4e" />

<img width="500" height="290" alt="Combined_E2ED" src="https://github.com/user-attachments/assets/98db3fc8-ceb8-4081-80eb-942dd922b163" />

Machine Learning Results on OC/LC classification:

<img width="397" height="181" alt="OC_SVM" src="https://github.com/user-attachments/assets/b9d123b4-e4e5-43ed-8ca3-b3e071a07f44" />

<img width="411" height="165" alt="LC_SVM" src="https://github.com/user-attachments/assets/eb67491c-a452-4af8-8aa1-ee67f70dfe1a" />

<img width="512" height="518" alt="RF" src="https://github.com/user-attachments/assets/3f8aa24d-a651-488d-b112-f1c3cba73298" />

Video Demonstration of Simulation:

https://github.com/user-attachments/assets/a70cc8a5-30fc-4b64-bf31-032ea35a167e


