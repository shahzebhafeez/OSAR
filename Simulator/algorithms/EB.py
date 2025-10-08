"""
algorithms/eb.py

Energy-beacon / Extended-beacon helpers. Compute/aggregate/summarize
sensing results shared by neighbors. In this simplified implementation,
we provide utilities for generating sensing outcomes and merging them.
"""
import numpy as np
from typing import List
from ..utils.utils import UnderwaterNode


def random_sensing(node: UnderwaterNode, M: int, pu_active: bool, busy_prob: float = 0.5) -> None:
    """
    Simulate channel sensing for a node. Fills node.idle_channels list.

    Parameters:
      - M: number of channels
      - pu_active: whether PUs are currently active in the region (approx)
      - busy_prob: conditional probability a channel is busy if PU active
    """
    node.idle_channels.clear()
    for ch in range(M):
        if pu_active:
            if np.random.rand() < busy_prob:
                # busy channel
                continue
        node.idle_channels.append(ch)


def merge_neighbor_sensing(node: UnderwaterNode, neighbors: List[UnderwaterNode]) -> dict:
    """
    Create a summary table of neighbors' idle channels.
    Returns dict: neighbor_id -> set(idle_channels)
    """
    summary = {}
    for nbr in neighbors:
        summary[nbr.id] = set(nbr.idle_channels)
    return summary
