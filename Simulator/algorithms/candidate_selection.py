"""
algorithms/candidate_selection.py

Utilities to select candidate relay nodes from a node's neighbor set
based on OSAR constraints (upward progress, common idle channel, in-range).
"""
import numpy as np
from typing import List, Tuple, Optional
from ..utils.utils import UnderwaterNode, get_distance


def find_candidates(current: UnderwaterNode,
                    graph,
                    nodes_by_id: dict,
                    tx_range: float) -> List[UnderwaterNode]:
    """
    Return list of UnderwaterNode objects that are valid candidate relays for `current`.

    Conditions:
      - neighbor must exist in graph and be within tx_range (graph already encodes edges)
      - neighbor must be shallower (depth < current.depth)
      - neighbor must make positive projection towards destination (this check is left to routing)
      - neighbor object must be present in nodes_by_id
    """
    candidates = []
    for nbr_id, dist in current.neighbor_nodes.items():
        if nbr_id not in nodes_by_id:
            continue
        nbr = nodes_by_id[nbr_id]
        # already precomputed neighbor distances; ensure within tx_range
        if dist > tx_range:
            continue
        if nbr.depth >= current.depth:
            continue
        candidates.append(nbr)
    return candidates


def have_common_idle(current: UnderwaterNode, candidate: UnderwaterNode) -> bool:
    """
    Return True if current and candidate share at least one idle channel.
    Both nodes maintain idle_channels as a list of channel indices.
    """
    return len(set(current.idle_channels) & set(candidate.idle_channels)) > 0
