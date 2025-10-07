import numpy as np


def sound_speed(z_km, s_ppt, T_c):
    """Sound speed c(z,s,T) in m/s (eq. 4)."""
    T = T_c / 10
    return (1449.05 + 45.7*T - 5.21*T**2 + 0.23*T**3 +
            (1.333 - 0.126*T + 0.009*T**2)*(s_ppt - 35) +
            16.3*z_km + 0.18*z_km**2)


def noise_psd(f_khz, s=0.5, w=0):
    """Total noise N(f) [dB re µPa/Hz]."""
    Nt = 17 - 30*np.log10(f_khz)
    Ns = 40 + 20*(s-0.5) + 26*np.log10(f_khz) - 60*np.log10(f_khz+0.03)
    Nw = 50 + 7.5*w + 20*np.log10(f_khz) - 40*np.log10(f_khz+0.4)
    Nth = -15 + 20*np.log10(f_khz)
    N_total_db = Nt + Ns + Nw + Nth
    return 10**(N_total_db/10)  # linear



def thorp_absorption(f_khz):
    """Absorption coefficient a(f) [dB/km] using Thorp’s formula (eq. 8)."""
    f2 = f_khz**2
    return 0.11*f2/(1+f2) + 44*f2/(4100+f2) + 2.75e-4*f2 + 0.00303


def path_loss(d_km, f_khz, k=1.5):
    """Path loss A(d,f) in linear scale."""
    alpha_db_per_km = thorp_absorption(f_khz)
    alpha = 10**(alpha_db_per_km/10)  # convert dB to linear per km
    return (d_km**k) * (alpha**d_km)


def snr(d_km, f_khz, Pt=1.0, delta_f=1.0):
    """Compute SNR (eq. 5)."""
    A = path_loss(d_km, f_khz)
    Nf = noise_psd(f_khz)
    return Pt / (A * Nf * delta_f)

def energy_detection(y, noise_var=1.0, threshold_factor=2.0):
    """Energy detection: returns 1 if idle, 0 if busy."""
    energies = np.sum(np.abs(y)**2, axis=0) / y.shape[0]
    threshold = threshold_factor * noise_var
    return (energies < threshold).astype(int)


def propagation_delay(d_m: float, z_km: float = 0.5, s_ppt: float = 35.0, T_c: float = 10.0) -> float:
    c = sound_speed(z_km, s_ppt, T_c)  # m/s
    return d_m / c

if __name__ == "__main__":
    # Channel characteristics
    z, s, T = 1.0, 35, 15
    c = sound_speed(z, s, T)
    print(f"Sound speed at depth={z}km, T={T}°C, salinity={s}ppt: {c:.2f} m/s")

    f_khz = 20
    d_km = 1
    print("Absorption coefficient (dB/km):", thorp_absorption(f_khz))
    print("Path loss (linear):", path_loss(d_km, f_khz))
    print("Noise PSD:", noise_psd(f_khz))
    print("SNR:", snr(d_km, f_khz))

    P = 64
    pu_signal = np.random.randn(10, P) + 1j*np.random.randn(10, P)
    idle_mask = energy_detection(pu_signal)
    print("Idle mask:", idle_mask)