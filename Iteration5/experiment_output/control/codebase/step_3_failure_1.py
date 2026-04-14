# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = False

def cic_density(pos, N_grid, L_box):
    grid = np.zeros((N_grid, N_grid, N_grid), dtype=np.float32)
    H = L_box / N_grid
    pos_g = pos / H
    idx = np.floor(pos_g).astype(int)
    d = pos_g - idx
    t = 1.0 - d
    idx1 = (idx + 1) % N_grid
    idx0 = idx % N_grid
    w000 = t[:, 0] * t[:, 1] * t[:, 2]
    w001 = t[:, 0] * t[:, 1] * d[:, 2]
    w010 = t[:, 0] * d[:, 1] * t[:, 2]
    w011 = t[:, 0] * d[:, 1] * d[:, 2]
    w100 = d[:, 0] * t[:, 1] * t[:, 2]
    w101 = d[:, 0] * t[:, 1] * d[:, 2]
    w110 = d[:, 0] * d[:, 1] * t[:, 2]
    w111 = d[:, 0] * d[:, 1] * d[:, 2]
    np.add.at(grid, (idx0[:, 0], idx0[:, 1], idx0[:, 2]), w000)
    np.add.at(grid, (idx0[:, 0], idx0[:, 1], idx1[:, 2]), w001)
    np.add.at(grid, (idx0[:, 0], idx1[:, 1], idx0[:, 2]), w010)
    np.add.at(grid, (idx0[:, 0], idx1[:, 1], idx1[:, 2]), w011)
    np.add.at(grid, (idx1[:, 0], idx0[:, 1], idx0[:, 2]), w100)
    np.add.at(grid, (idx1[:, 0], idx0[:, 1], idx1[:, 2]), w101)
    np.add.at(grid, (idx1[:, 0], idx1[:, 1], idx0[:, 2]), w110)
    np.add.at(grid, (idx1[:, 0], idx1[:, 1], idx1[:, 2]), w111)
    return grid

def compute_pk(pos, L_box, N_grid):
    N_part = len(pos)
    grid = cic_density(pos, N_grid, L_box)
    mean_density = N_part / (N_grid**3)
    delta = grid / mean_density - 1.0
    delta_k = np.fft.rfftn(delta)
    kx = np.fft.fftfreq(N_grid, d=L_box/N_grid) * 2 * np.pi
    ky = np.fft.fftfreq(N_grid, d=L_box/N_grid) * 2 * np.pi
    kz = np.fft.rfftfreq(N_grid, d=L_box/N_grid) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    H = L_box / N_grid
    def W(k):
        arg = k * H / 2.0
        s = np.ones_like(arg)
        mask = arg != 0
        s[mask] = np.sin(arg[mask]) / arg[mask]
        return s**2
    W_k = W(KX) * W(KY) * W(KZ)
    P_k_3D = np.abs(delta_k)**2 * (L_box**3) / (N_grid**6)
    P_shot = L_box**3 / N_part
    P_k_3D_corrected = P_k_3D / W_k**2 - P_shot
    k_min = 2 * np.pi / L_box
    k_max = np.pi / H
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 30)
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    P_k_1D = np.zeros(len(k_centers))
    P_k_1D_raw = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            P_k_1D[i] = np.mean(P_k_3D_corrected[mask])
            P_k_1D_raw[i] = np.mean((P_k_3D / W_k**2)[mask])
        else:
            P_k_1D[i] = np.nan
            P_k_1D_raw[i] = np.nan
    return k_centers, P_k_1D, P_k_1D_raw, P_shot

def P_lin(k_h, Omega_m=0.311, Omega_b=0.049, h=0.677, n_s=0.966, sigma8=0.810):
    theta_cmb = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / (Omega_m * h**2)) / np.sqrt(1.0 + 10.0 * (Omega_b * h**2)**0.75)
    alpha_gamma = 1.0 - 0.328 * np.log(431.0 * Omega_m * h**2) * (Omega_b / Omega_m) + 0.38 * np.log(22.3 * Omega_m * h**2) * (Omega_b / Omega_m)**2
    k_Mpc = k_h * h
    Gamma_eff = Omega_m * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_Mpc * s)**4))
    q = k_h * theta_cmb**2 / Gamma_eff
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T_k = L0 / (L0 + C0 * q**2)
    P_unnorm = k_h**n_s * T_k**2
    return P_unnorm

def get_normalized_Plin(k_h, Omega_m=0.311, Omega_b=0.049, h=0.677, n_s=0.966, sigma8=0.810):
    k_int = np.logspace(-4, 3, 2000)
    P_int = P_lin(k_int, Omega_m, Omega_b, h, n_s, sigma8)
    R = 8.0
    x = k_int * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    integrand = k_int**2 * P_int * W**2 / (2.0 * np.pi**2)
    sigma8_unnorm_sq = np.trapz(integrand, k_int)
    A = sigma8**2 / sigma8_unnorm_sq
    return A * P_lin(k_h, Omega_m, Omega_b, h, n_s, sigma8)

if __name__ == '__main__':
    data_dir = "data/"
    output_dir = "data/"
    N_grid = 256
    L_box = 500.0
    all_k = None
    P_gal_all = []
    P_halo_all = []
    P_gal_raw_all = []
    P_shot_gal_all = []
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(halo_file):
            continue
        gal_data = np.load(gal_file)
        halo_data = np.load(halo_file)
        gal_pos = gal_data[:, 0:3]
        halo_pos = halo_data[:, 0:3]
        k_centers, P_gal, P_gal_raw, P_shot_gal = compute_pk(gal_pos, L_box, N_grid)
        _, P_halo, _, _ = compute_pk(halo_pos, L_box, N_grid)
        if all_k is None:
            all_k = k_centers
        P_gal_all.append(P_gal)
        P_halo_all.append(P_halo)
        P_gal_raw_all.append(P_gal_raw)
        P_shot_gal_all.append(P_shot_gal)
    P_gal_mean = np.nanmean(P_gal_all, axis=0)
    P_halo_mean = np.nanmean(P_halo_all, axis=0)
    P_gal_raw_mean = np.nanmean(P_gal_raw_all, axis=0)
    P_shot_gal_mean = np.mean(P_shot_gal_all)
    P_lin_k = get_normalized_Plin(all_k)
    mask_large_scale = all_k < 0.05
    b_eff = np.sqrt(np.nanmean(P_halo_mean[mask_large_scale] / P_lin_k[mask_large_scale]))
    P_lin_scaled = b_eff**2 * P_lin_k
    valid_k_mask = P_gal_raw_mean > 2 * P_shot_gal_mean
    if np.any(valid_k_mask):
        k_valid = all_k[valid_k_mask]
        k_min_valid = np.min(k_valid)
        k_max_valid = np.max(k_valid)
        k_range_str = str(round(k_min_valid, 4)) + " to " + str(round(k_max_valid, 4)) + " h/Mpc"
    else:
        k_range_str = "None"
    plt.figure(figsize=(8, 6))
    plt.loglog(all_k, P_gal_mean, label='Galaxy P(k)', color='blue', lw=2)
    plt.loglog(all_k, P_halo_mean, label='Halo P_hh(k)', color='red', lw=2)
    plt.loglog(all_k, P_lin_scaled, label='Theoretical b^2 P_lin(k)', color='black', linestyle='--', lw=2)
    plt.axhline(P_shot_gal_mean, color='blue', linestyle=':', label='Galaxy Shot Noise')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)^3]')
    plt.title('3D Power Spectrum Validation')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, "power_spectrum_validation_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    out_npz = os.path.join(output_dir, "power_spectra.npz")
    np.savez(out_npz, k=all_k, P_gal_mean=P_gal_mean, P_halo_mean=P_halo_mean, P_lin_scaled=P_lin_scaled, P_shot_gal=P_shot_gal_mean, b_eff=b_eff)