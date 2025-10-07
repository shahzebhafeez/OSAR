import numpy as np
import scipy
import networkx as nx
from typing import List,Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class UnderwaterNode:
    def __init__(self,id:int,pos:np.ndarray,depth:float):
        self.id = id # sensor node 1 
        self.pos = pos   # position = [x,y,z]
        self.depth = depth 
        self.idle_channels : List[int] = []  # indices of idle channels (0 to M-1)
        self.neighbor_nodes : dict = {} # {neighbor_id : distance}

def generate_environment(N:int,area_size:float = 500.0,max_depth:float=500.0):
    sensors = []
    for i in range(N):
        pos = np.random.uniform(0,area_size,3) # [x,y,z] => pos => [0,1,2] => [x,y,z] = pos[3] => depth
        pos[2] = np.random.uniform(0,max_depth) # depth 
        sensors.append(UnderwaterNode(i,pos,pos[2]))
    buoy_position = np.array([area_size/2,area_size/2,0]) # surface_buoy => [250,250,0]
    buoy = UnderwaterNode(N,buoy_position,0.0)  
    return sensors,[buoy] 
# sensors => {"id":0,"pos":[10,20,30],"depth":200,..}
# buoy => {"id":30,"pos":[250,250,0],"depth":0.0,..}

def compute_neighbors(nodes: List[UnderwaterNode],tx_range : float = 250.0):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id,pos=node.pos,depth=node.depth)
    for i,node1 in enumerate(nodes):
        for j,node2 in enumerate(nodes[i+1:],i+1):
            dist = np.linalg.norm(node1.pos - node2.pos)
            if dist <= tx_range:
                G.add_edge(node1.id,node2.id,weight=dist)
                node1.neighbor_nodes[node2.id] = dist
                node2.neighbor_nodes[node1.id] = dist

    return G

def visualize_path(G: nx.Graph, path: List[int], all_paths: List[List[int]] = None, title: str = "OSAR Routing Path"):
    """Visualize network with highlighted path (2D top-down)."""
    if not path:  # Handle empty path
        print("No path provided; skipping visualization.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 3d path visualization => G.nodes[node]['pos'][2] => z
    # pos_3d = {"node":(10.0,23.1,44.5)}
    all_xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    all_ys = [G.nodes[n]['pos'][1] for n in G.nodes]
    all_zs = [G.nodes[n]['pos'][2] for n in G.nodes]

    scatter = ax.scatter(all_xs, all_ys, all_zs, c=all_zs, cmap='Blues_r', 
                         s=100, alpha=0.7)
    for u, v in G.edges():
        if G.has_edge(u, v):  # Ensure edge exists
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]], 
                    'gray', alpha=0.3, linewidth=1)
            
    # Labels for all nodes (small)
    for i, n in enumerate(G.nodes):
        ax.text(all_xs[i], all_ys[i], all_zs[i], str(n), fontsize=6, 
                ha='center', va='center', color='black')

    # Draw full graph (gray edges, nodes colored by depth)
    # depths = [230.0,33.4,10,....]
    # Highlight path (if provided)
    if path and len(path) > 1:
        # Filter valid nodes
        valid_path = [n for n in path if n in G]
        if len(valid_path) > 1:
            path_xs = [G.nodes[n]['pos'][0] for n in valid_path]
            path_ys = [G.nodes[n]['pos'][1] for n in valid_path]
            path_zs = [G.nodes[n]['pos'][2] for n in valid_path]
            
            # Red thick line for path
            ax.plot(path_xs, path_ys, path_zs, 'red', linewidth=4, alpha=1.0, label='Path')
            
            # Larger red markers for path nodes
            ax.scatter(path_xs, path_ys, path_zs, c='red', s=200, alpha=1.0, edgecolor='black')
            
            # Bold labels for path nodes
            for i, n in enumerate(valid_path):
                ax.text(path_xs[i], path_ys[i], path_zs[i], str(n), fontsize=10, 
                        ha='center', va='center', color='red', fontweight='bold')
    
    # Highlight path (red thick edges/nodes)
    # example path = [0,5,30]
    # Overlay previous paths (faded red)
    if all_paths:
        for prev_path in all_paths:
            valid_prev = [n for n in prev_path if n in G]
            if len(valid_prev) > 1:
                prev_xs = [G.nodes[n]['pos'][0] for n in valid_prev]
                prev_ys = [G.nodes[n]['pos'][1] for n in valid_prev]
                prev_zs = [G.nodes[n]['pos'][2] for n in valid_prev]
                ax.plot(prev_xs, prev_ys, prev_zs, 'red', linewidth=2, alpha=0.4)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')  # Z downward (surface=0, deep=500)
    ax.set_title(title)
    ax.invert_zaxis()

    # Colorbar for depth
    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Depth (m)')
    
    # Legend (if path)
    if path:
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def get_distance(G: nx.Graph, node_id1: int, node_id2: int, print_info: bool = True) -> float:
    if node_id1 not in G or node_id2 not in G:
        raise ValueError(f"Node {node_id1} or {node_id2} not in graph.")
    
    pos1 = G.nodes[node_id1]['pos']
    pos2 = G.nodes[node_id2]['pos']
    euclidean_dist = np.linalg.norm(pos1 - pos2)
    
    if G.has_edge(node_id1, node_id2):
        edge_dist = G[node_id1][node_id2]['weight']
        dist = edge_dist
        status = " (direct neighbor)"
    else:
        dist = euclidean_dist
        status = " (not direct neighbor; using Euclidean)"
    
    if print_info:
        print(f"Node {node_id1} position: ({pos1[0]:.1f}, {pos1[1]:.1f}, {pos1[2]:.1f}) m (depth: {pos1[2]:.1f}m)")
        print(f"Node {node_id2} position: ({pos2[0]:.1f}, {pos2[1]:.1f}, {pos2[2]:.1f}) m (depth: {pos2[2]:.1f}m)")
        print(f"Distance between node {node_id1} and node {node_id2}: "
              f"{dist:.2f}m{status}\n")
    
    return dist

    

# Quick setup for tests
def setup_small_net():
    sensors, buoys = generate_environment(5, 200.0, 200.0)  # Small: N=5, area=200m, depth=200m
    all_nodes = sensors + buoys
    G = compute_neighbors(all_nodes, tx_range=100.0)  # Short range for few neighbors
    node_dict = {node.id: node for node in all_nodes}  # Lookup for candidates
    return all_nodes, G, node_dict, sensors[0], buoys[0]  # Return source (sensor 0), dest (buoy 5)

all_nodes, G, node_dict, source, dest = setup_small_net()
print(f"Test net: {len(all_nodes)} nodes, {len(G.edges())} edges")



