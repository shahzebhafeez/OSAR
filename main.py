import simpy
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_environment, compute_neighbors, visualize_path
from osar_routing import OSARRouter, run_packet_forwarding

def simulate(N: int = 20, num_packets: int = 5, M: int = 5, visualize: bool = True) :
    """Simulate OSAR: Returns (avg_delay, pdr, all_paths)."""
    sensors, buoys = generate_environment(N)
    global G  # For routing access
    G = compute_neighbors(sensors + buoys)
    router = OSARRouter(num_channels=M)
    env = simpy.Environment()
    
    total_delays = []
    delivered = 0
    all_paths = []
    
    # Source: Deepest sensor
    source = max(sensors, key=lambda n: n.depth)
    dest = buoys[0]
    
    # Initial sensing
    pu_active = np.random.exponential(1) > 1
    env.process(router.sense_and_beacon(source, pu_active, env))
    env.run(until=0.01)  # Run sensing
    
    # Assume neighbors also sense similarly (simplified)
    for node in sensors + buoys:
        env.process(router.sense_and_beacon(node, pu_active, env))
    env.run(until=0.02)  # Quick sense all
    
    def packet_gen():
        for pkt in range(num_packets):
            bits = np.random.randint(0, 2, 512)  # Dummy (unused now)
            delay, success, path = yield env.process(
                run_packet_forwarding(env, source, dest, router)
            )
            total_delays.append(delay if success else float('inf'))
            if success:
                delivered += 1
            all_paths.append(path)
            yield env.timeout(1.0)  # Inter-packet
    
    env.process(packet_gen())
    env.run(until=1000.0)
    
    pdr = delivered / num_packets
    avg_delay = np.mean([d for d in total_delays if d < float('inf')]) if any(d < float('inf') for d in total_delays) else float('inf')
    print(f"N={N}, M={M}: Avg Delay={avg_delay:.3f}s, PDR={pdr:.3f}")
    
    if visualize:
        visualize_path(G, all_paths[-1] if all_paths else None, all_paths, 
                       f"OSAR Paths for N={N}, M={M} (Last: Bold Red)")
    
    return avg_delay, pdr, all_paths

if __name__ == "__main__":
    # Multi-N sim (paper-like)
    N_values = [10, 20, 30]
    results = []
    for N in N_values:
        avg_d, pdr, _ = simulate(N=N, num_packets=3, M=5, visualize=True)  # Viz per N
        results.append((N, avg_d, pdr))
    
    # Summary plot: Delay/PDR vs N
    Ns, delays, pdrs = zip(*results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(Ns, delays, 'bo-', linewidth=2)
    ax1.set_xlabel('N'); ax1.set_ylabel('Avg Delay (s)'); ax1.set_title('Delay vs N')
    ax1.grid(True)
    ax2.plot(Ns, pdrs, 'go-', linewidth=2)
    ax2.set_xlabel('N'); ax2.set_ylabel('PDR'); ax2.set_title('PDR vs N')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()