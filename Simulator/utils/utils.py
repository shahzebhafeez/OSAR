"""
utils/utils.py

Utility functions and small data structures used across the simulation.

Contains:
- UnderwaterNode class (lightweight container)
- generate_environment: create N random sensor nodes + buoys
- compute_neighbors: populate neighbor relationships using euclidean distance and tx_range
- visualize_path: 3D matplotlib visualizer for the graph and route
- get_distance: helper to return and print distances
"""
import numpy as np
import networkx as nx
from typing import List, Tuple
import matplotlib.pyplot as plt


class UnderwaterNode:
    def __init__(self, node_id: int, pos: np.ndarray, depth: float = None):
        self.id = int(node_id)
        self.pos = np.array(pos, dtype=float)
        self.depth = float(depth) if depth is not None else float(self.pos[2])
        self.idle_channels: List[int] = []
        self.neighbor_nodes: dict = {}  # neighbor_id -> distance

    def __repr__(self):
        return f"Node({self.id}, depth={self.depth:.1f}m)"


def generate_environment(N: int, area_size: float = 500.0, max_depth: float = 500.0, seed: int = None) -> Tuple[List[UnderwaterNode], List[UnderwaterNode]]:
    """
    Generate N random sensor nodes in a cubic underwater volume and a surface buoy at center.
    Returns sensors list and buoys list (single buoy by default).
    """
    rng = np.random.default_rng(seed)
    sensors = []
    for i in range(N):
        pos = rng.uniform(0, area_size, 3)
        pos[2] = rng.uniform(0, max_depth)  # depth
        sensors.append(UnderwaterNode(i, pos, pos[2]))
    buoy_pos = np.array([area_size / 2.0, area_size / 2.0, 0.0])
    buoy = UnderwaterNode(N, buoy_pos, 0.0)
    return sensors, [buoy]


def compute_neighbors(nodes: List[UnderwaterNode], tx_range: float = 250.0) -> nx.Graph:
    """
    Build an undirected NetworkX graph of nodes where edges exist if euclidean distance <= tx_range.
    Also populate each node.neighbor_nodes dict with neighbor_id: distance.
    """
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id, pos=node.pos, depth=node.depth)
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes[i + 1:], i + 1):
            dist = float(np.linalg.norm(n1.pos - n2.pos))
            if dist <= tx_range:
                G.add_edge(n1.id, n2.id, weight=dist)
                n1.neighbor_nodes[n2.id] = dist
                n2.neighbor_nodes[n1.id] = dist
    return G


def visualize_path(G: nx.Graph, path: List[int], all_paths: List[List[int]] = None, title: str = "Routing Path"):
    """
    Visualize the network in 3D with nodes and edges. Highlight a given path (bold red).
    """
    if not G:
        print("No graph to visualize.")
        return
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass

    nodes = list(G.nodes)
    xs = [G.nodes[n]['pos'][0] for n in nodes]
    ys = [G.nodes[n]['pos'][1] for n in nodes]
    zs = [G.nodes[n]['pos'][2] for n in nodes]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=zs, cmap='Blues_r', s=60, alpha=0.8)

    # draw edges
    for u, v in G.edges():
        p1 = G.nodes[u]['pos']
        p2 = G.nodes[v]['pos']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.3)

    # highlight path
    if path and len(path) > 1:
        valid = [n for n in path if n in G]
        px = [G.nodes[n]['pos'][0] for n in valid]
        py = [G.nodes[n]['pos'][1] for n in valid]
        pz = [G.nodes[n]['pos'][2] for n in valid]
        ax.plot(px, py, pz, color='red', linewidth=3, alpha=1.0)
        ax.scatter(px, py, pz, color='red', s=120, edgecolor='k')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title(title)
    ax.invert_zaxis()
    plt.colorbar(sc, label='Depth (m)', shrink=0.6)
    plt.show()


def get_distance(G: nx.Graph, node_id1: int, node_id2: int, print_info: bool = True) -> float:
    """
    Return euclidean distance between two nodes in graph G. If there's an edge, returns edge weight.
    """
    if node_id1 not in G or node_id2 not in G:
        raise ValueError(f"Node {node_id1} or {node_id2} not in graph.")
    pos1 = G.nodes[node_id1]['pos']
    pos2 = G.nodes[node_id2]['pos']
    if G.has_edge(node_id1, node_id2):
        dist = float(G[node_id1][node_id2]['weight'])
        status = " (direct neighbor)"
    else:
        dist = float(np.linalg.norm(pos1 - pos2))
        status = " (not direct neighbor)"
    if print_info:
        print(f"Node {node_id1} pos: {pos1}, Node {node_id2} pos: {pos2}")
        print(f"Distance: {dist:.2f} m{status}")
    return dist
