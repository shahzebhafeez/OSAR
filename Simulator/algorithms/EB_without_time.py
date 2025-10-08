"""
algorithms/eb_without_time.py

A minimal variant of EB utilities that ignore sensing time and temporal aspects.
Useful for quick simulations where sensing is instantaneous.
"""
from .EB import random_sensing, merge_neighbor_sensing
# This file intentionally re-exports the simple EB helpers for clarity.
__all__ = ["random_sensing", "merge_neighbor_sensing"]
