# physics_equations.py
import numpy as np
import config as c

def distance_3d(a, b):
    """Calculates Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def thorp_absorption_db_per_m(f_khz: float) -> float:
    """
    Calculates absorption coefficient using Thorp's formula.
    Returns: dB per meter
    """
    f2 = f_khz**2
    db_per_km = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.00303
    return db_per_km / 1000.0

def path_loss_linear(d_m: float, f_khz: float, k: float = c.K_SPREAD) -> float:
    """
    Calculates Path Loss (Attenuation) in linear scale.
    PL = d^k * alpha^d
    """
    if d_m <= 0: d_m = c.EPS
    alpha_db_per_m = thorp_absorption_db_per_m(f_khz)
    a_linear_per_m = 10 ** (alpha_db_per_m / 10.0)
    return (d_m ** k) * (a_linear_per_m ** d_m)

def noise_psd_linear(f_khz: float, shipping: float = 0.5, wind: float = 0.0) -> float:
    """
    Calculates Noise Power Spectral Density (Linear).
    Combines Turbulence, Shipping, Wind, and Thermal noise.
    """
    f_khz_safe = f_khz + c.EPS
    Nt_db = 17 - 30 * np.log10(f_khz_safe)
    Ns_db = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f_khz_safe) - 60 * np.log10(f_khz_safe + 0.03)
    Nw_db = 50 + 7.5 * wind + 20 * np.log10(f_khz_safe) - 40 * np.log10(f_khz_safe + 0.4)
    Nth_db = -15 + 20 * np.log10(f_khz_safe)
    
    return (10**(Nt_db/10.0) + 10**(Ns_db/10.0) + 
            10**(Nw_db/10.0) + 10**(Nth_db/10.0))

def sound_speed(z_m: float, s_ppt: float = c.SALINITY, T_c: float = c.TEMP) -> float:
    """
    Mackenzie's nine-term equation for Speed of Sound in water.
    """
    z_km = z_m / 1000.0
    T = T_c / 10.0
    return (1449.05 + 45.7*T - 5.21*(T**2) + 0.23*(T**3) +
            (1.333 - 0.126*T + 0.009*(T**2))*(s_ppt - 35) +
            16.3*z_km + 0.18*(z_km**2))

def compute_transmission_delay_paper(src_pos, src_depth, dest_pos, dest_depth, f_khz, lp, bw, pt_linear):
    """
    Calculates End-to-End Link Delay: Propagation Delay + Transmission Delay.
    Also returns: Path Loss Factor (A_df) for energy calcs.
    """
    dist_m = max(distance_3d(src_pos, dest_pos), c.EPS)
    
    if dist_m > c.TX_RANGE:
        return float('inf'), 0.0, 0.0, 0.0

    A_df = path_loss_linear(dist_m, f_khz) 
    Nf = noise_psd_linear(f_khz)
    noise_total = Nf * bw
    if noise_total < c.EPS: noise_total = c.EPS
    
    pt_lin = max(pt_linear, c.EPS)
    snr_linear = pt_lin / (A_df * noise_total + c.EPS)
    
    # Shannon Capacity
    if snr_linear <= 0: 
        r_ij_ch = c.EPS
    else: 
        r_ij_ch = bw * np.log2(1.0 + snr_linear)
    
    if r_ij_ch <= c.EPS: 
        return float('inf'), 0.0, 0.0, A_df
    
    # Speed of sound at average depth
    avg_depth = max((src_depth + dest_depth)/2.0, 0.0)
    sound_c = sound_speed(avg_depth)
    
    prop_delay = dist_m / sound_c
    trans_delay = lp / r_ij_ch
    
    return (prop_delay + trans_delay), r_ij_ch, snr_linear, A_df

def calculate_auv_cost_metric(phys_delay, A_df, pt_db=c.DEFAULT_PT_DB):
    """
    Calculates the specific heuristic cost for selecting an AUV.
    Cost = Delay + Energy_Factor + Power_Factor
    """
    path_loss_db = 10 * np.log10(A_df)
    pr_db = pt_db - path_loss_db
    if pr_db <= 0: pr_db = 0.001 
    
    power_factor = pt_db / pr_db 
    energy_factor = c.CONST_EC / c.CONST_EF
    
    return phys_delay + energy_factor + power_factor

def calculate_ecr(d_bits, c_bits, num_nodes, duration):
    """
    Calculates Energy Consumption Ratio based on user formula:
    ECR = (E_data + E_ctrl) / (E_data + E_ctrl + E_move + E_idle)
    """
    e_data = d_bits * c.E_BIT_TX
    e_ctrl = c_bits * c.E_BIT_TX
    e_move = c.E_MOVE_FIXED
    e_idle = c.CONST_E_IDLE * num_nodes * duration
    
    numerator = e_data + e_ctrl
    denominator = e_data + e_ctrl + e_move + e_idle
    
    if denominator > 0:
        return numerator / denominator
    return 0.0