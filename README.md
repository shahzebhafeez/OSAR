OSAR-UW: Cross-Layer Underwater Acoustic Networking Simulator
📌 Project Overview
A specialized discrete-event simulation platform designed to model Underwater Wireless Sensor Networks (UWSNs). Unlike terrestrial networks, underwater communication relies on acoustic waves, which suffer from low bandwidth (kbps range) and extreme propagation delays (1500m/s). This project simulates the entire networking stack—from Thorp’s attenuation models at the physical layer to depth-based routing (DBR) at the network layer—providing a sandbox to test protocol reliability in harsh, non-linear oceanic environments.

🚀 Key Features
Acoustic PHY Modeling: Implements the Thorp Propagation Model to calculate path loss, ambient noise (shipping, wind, thermal), and frequency-dependent attenuation.

Propagation Delay Simulation: Accurately models the "Slotted-Prop" effect, where high latency leads to frequent packet collisions and requires specialized MAC scheduling.

Dynamic Topology & Mobility: Supports 3D node deployment with Gaussian Random Walk mobility to simulate AUVs (Autonomous Underwater Vehicles) and drifting sensor buoys.

Protocol Sandbox: Pre-configured with Slotted ALOHA for MAC and Depth-Based Routing (DBR), allowing for "Void Hole" detection and energy-consumption analysis.

🛠 Tech Stack
Simulation Engine: Python 3.x / SimPy (Discrete-Event Simulation) or C++ (NS-3/Aqua-Sim).

Mathematics: NumPy & SciPy (Signal attenuation & BER curves).

Visualization: Matplotlib 3D for node trajectory and packet-flow heatmaps.

Data Format: JSON-based configuration files for sea-state parameters (salinity, temperature, depth).

📊 Performance Metrics
Packet Delivery Ratio (PDR): Benchmarked under varying "Turbid" and "Deep Ocean" noise profiles.

End-to-End Latency: Successfully modeled the 5-order-of-magnitude delay increase compared to RF networks.

Energy Efficiency: Tracks "Battery Drain per Bit" to predict the operational lifespan of submerged nodes.

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


