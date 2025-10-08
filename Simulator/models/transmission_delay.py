"""
models/transmission_delay.py

Transmission-delay related helpers built on the channel model.

Implements:
- compute_data_rate: Shannon approximation using bandwidth and SNR
- compute_transmission_delay: Eq. (9) style TD = LP / rate + PD * est_hops
"""
import numpy as np
from typing import Tuple
from .channel_model import snr, propagation_delay


def compute_data_rate(dist_m: float, f_khz: float, bandwidth_hz: float = 6000.0, Pt: float = 1.0) -> float:
    """
    Approximate achievable data rate (bits/sec) using Shannon formula:
      R = B * log2(1 + SNR)
    Uses channel_model.snr for SNR computation.
    """
    snr_lin = snr(dist_m, f_khz, Pt=Pt, delta_f=bandwidth_hz)
    # protect against numerical issues
    return bandwidth_hz * np.log2(1.0 + max(snr_lin, 0.0))


def compute_transmission_delay(src_pos: np.ndarray, src_depth: float,
                               dst_pos: np.ndarray, dst_depth: float,
                               f_khz: float, LP_bits: int = 512,
                               bandwidth_hz: float = 6000.0, Pt: float = 1.0,
                               dest_pos: np.ndarray = None) -> Tuple[float, float]:
    """
    Compute TD_ij for a single candidate link (i->j).

    Returns (TD_seconds, propagation_delay_seconds).

    Parameters:
      - src_pos, dst_pos: numpy arrays (3,) with positions in meters
      - src_depth, dst_depth: depths (meters)
      - f_khz: center frequency (kHz) used to compute SNR
      - LP_bits: packet length in bits
      - bandwidth_hz: channel bandwidth
      - Pt: transmit power

    Implementation:
      TD = LP / rate + PD * est_hops
      PD = |depth_i - depth_j| / c  (using propagation_delay over horizontal distance)
      est_hops estimated as max(src_depth / (depth_i - depth_j), 1)
    """
    dist_m = np.linalg.norm(src_pos - dst_pos)
    rate_bps = compute_data_rate(dist_m, f_khz, bandwidth_hz, Pt=Pt)
    # avoid divide by zero
    if rate_bps <= 1e-12:
        tx_time = float('inf')
    else:
        tx_time = LP_bits / rate_bps

    # Use propagation delay based on horizontal distance and average sound speed (approx)
    pd_seconds = propagation_delay(dist_m, z_km=(max(src_depth, dst_depth) / 1000.0))
    # advance toward surface (positive if src deeper than dst)
    depth_diff = src_depth - dst_depth
    if depth_diff <= 0:
        est_hops = float('inf')  # invalid upward progress
    else:
        est_hops = max(src_depth / depth_diff, 1.0)

    TD = tx_time + pd_seconds * est_hops
    return TD, pd_seconds
