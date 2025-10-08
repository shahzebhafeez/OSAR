"""
models/channel_model.py

Underwater acoustic channel model utilities:
- sound_speed: sound speed vs depth/temp/salinity
- thorp_absorption: absorption (dB/km)
- noise_psd: total noise PSD (linear)
- path_loss: path loss A(d,f) in linear scale
- snr: compute linear SNR for given Pt, bandwidth, distance
- propagation_delay: distance / sound speed

All frequencies in kHz where noted. Distances are in meters unless name includes _km.
"""
import numpy as np
from typing import Tuple


def sound_speed(z_km: float, s_ppt: float = 35.0, T_c: float = 10.0) -> float:
    """
    Sound speed c(z,s,T) in m/s (Eq. 4).
    z_km: depth in kilometers
    s_ppt: salinity in parts per thousand (ppt)
    T_c: temperature in Celsius
    """
    T = T_c / 10.0
    return (1449.05 + 45.7 * T - 5.21 * T**2 + 0.23 * T**3 +
            (1.333 - 0.126 * T + 0.009 * T**2) * (s_ppt - 35) +
            16.3 * z_km + 0.18 * z_km**2)


def thorp_absorption(f_khz: float) -> float:
    """
    Thorp's absorption formula (approx). Returns absorption in dB/km (Eq. 8).
    """
    f2 = f_khz**2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303


def noise_psd(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    """
    Total noise PSD N(f) in linear units (power per Hz).
    Returns linear PSD (W/Hz relative units) — the original empirical equations
    give dB-reµPa^2/Hz. Here we convert the sum in dB to linear scale.
    Parameters shipping and wind are relative indices (0..1).
    """
    Nt = 17 - 30 * np.log10(f_khz)
    Ns = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f_khz) - 60 * np.log10(f_khz + 0.03)
    Nw = 50 + 7.5 * wind + 20 * np.log10(f_khz) - 40 * np.log10(f_khz + 0.4)
    Nth = -15 + 20 * np.log10(f_khz)
    N_total_db = Nt + Ns + Nw + Nth
    # convert dB (re µPa^2/Hz) to linear. For simulations we treat this as relative linear PSD.
    return 10 ** (N_total_db / 10.0)


def path_loss(d_m: float, f_khz: float, k: float = 1.5) -> float:
    """
    Path loss A(d,f) in linear scale.
    Uses A(d,f) = d^k * a(f)^d where a(f) is linear absorption per km.
    Implementation:
      - convert distance to km
      - alpha_db_per_km = thorp_absorption(f_khz)
      - alpha_linear_per_km = 10^(alpha_db_per_km / 10)
      - total_absorption = alpha_linear_per_km ** d_km
      - spreading = d_km ** k
    Returns linear attenuation factor A(d,f) (unitless, >=1).
    """
    d_km = max(d_m / 1000.0, 1e-9)
    alpha_db_per_km = thorp_absorption(f_khz)
    alpha_linear_per_km = 10 ** (alpha_db_per_km / 10.0)
    spreading = d_km ** k
    absorption = alpha_linear_per_km ** d_km
    return spreading * absorption


def snr(d_m: float, f_khz: float, Pt: float = 1.0, delta_f: float = 6000.0,
        shipping: float = 0.5, wind: float = 0.0) -> float:
    """
    Compute linear SNR for transmit power Pt (same units as noise) and channel
    of width delta_f (Hz). f_khz in kHz is used for noise/absorption lookup.
    Returns SNR (linear).
    """
    A = path_loss(d_m, f_khz)
    Nf = noise_psd(f_khz, shipping=shipping, wind=wind)
    # delta_f in Hz; noise PSD is per Hz, so total noise = Nf * delta_f
    noise_total = Nf * (delta_f / 1000.0) if delta_f > 1 else Nf * delta_f
    # Note: Nf computed from empirical dB->linear scale; units are relative.
    return Pt / (A * noise_total + 1e-30)


def propagation_delay(d_m: float, z_km: float = 0.5, s_ppt: float = 35.0, T_c: float = 10.0) -> float:
    """
    Compute propagation delay (seconds) for distance d_m using local sound speed.
    z_km used to compute sound speed (approximate).
    """
    c = sound_speed(z_km, s_ppt, T_c)
    return d_m / c
