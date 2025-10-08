import networkx as nx
import simpy
import matplotlib.pyplot as plt
import numpy as np

# ---------- Node Placement ----------
def place_nodes_uniform(N, area=500.0, rng=None):
    """Place N nodes randomly in a 3D cube [0, area]^3"""
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(0, area, N)
    ys = rng.uniform(0, area, N)
    zs = rng.uniform(0, area, N)   # depth (0 = surface, area = deep)
    positions = np.column_stack((xs, ys, zs))  # shape (N,3)
    return positions

# ---------- Distance Matrix ----------
def pairwise_distances(positions):
    """Compute pairwise Euclidean distances between nodes"""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists

# ---------- Positive Progress ----------
def positive_progress(src_idx, nbr_idx, positions, dest_pos):
    """Check if neighbor j makes positive progress toward destination"""
    # scr is current node and nbr is the neighbour node
    src = positions[src_idx]
    nbr = positions[nbr_idx]
    vec_sd = dest_pos - src
    vec_sj = nbr - src 
    #np.linalg.norm(vec_sd) for magnitude
    proj = np.dot(vec_sj, vec_sd) / (np.linalg.norm(vec_sd) + 1e-12)#1e-12 to overcome divison by 0
    return proj > 1e-6 #small number meaning proj>0

# ---------- Candidate Neighbors ----------
def candidates_for_src(i, positions, adj, channel_state, dest_pos):
    """Return list of valid neighbors with common idle channels"""
    nbrs = np.where(adj[i])[0]
    cand = []
    for j in nbrs:
        if not positive_progress(i, j, positions, dest_pos):
            continue
        # common idle channels (both idle = 0)
        common_idle = np.where(~channel_state[i] & ~channel_state[j])[0]
        if common_idle.size > 0:
            cand.append((j, common_idle))
    return cand

# ---------- Main Simulation + Plot ----------
def main():
    rng = np.random.default_rng()  

    # Parameters
    N = 30             # number of nodes
    AREA = 500.0       # cube dimension (meters)
    M = 128            # channels
    p_busy = 0.3       # probability channel busy
    tx_range = 150.0   # max comm range (meters)

    # Place nodes randomly
    positions = place_nodes_uniform(N, AREA, rng)
    depths = positions[:, 2]

    # Assign channel states (True=busy, False=idle)
    channel_state = rng.random((N, M)) < p_busy

    # Compute adjacency (neighbors within range)
    dists = pairwise_distances(positions)
    adj = (dists <= tx_range) & (dists > 0)

    # Destination (buoy at center surface)
    dest_pos = np.array([AREA/2, AREA/2, 0.0])

    # Plot setup
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all nodes
    sc = ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
                    c=depths, cmap='viridis', s=50)
    ax.scatter(dest_pos[0], dest_pos[1], dest_pos[2], c='red', s=100, marker='^', label='Buoy')

    # Draw neighbor connections for each node
    for i in range(N):
        cand = candidates_for_src(i, positions, adj, channel_state, dest_pos)
        for j, chans in cand:
            x = [positions[i,0], positions[j,0]]
            y = [positions[i,1], positions[j,1]]
            z = [positions[i,2], positions[j,2]]
            ax.plot(x, y, z, 'gray', alpha=0.4)

    # Labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Node Topology with Candidate Neighbor Links")
    fig.colorbar(sc, ax=ax, label="Depth (m)")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
