# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.optimize import curve_fit

def fit_models():
    data_dir = "data"
    orig_data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    satellites_path = os.path.join(data_dir, "processed_satellites.npy")
    satellites = np.load(satellites_path)
    x = satellites['r_sat'] / satellites['r_vir']
    alpha_mle = np.mean(x)
    alpha_std = np.std(x, ddof=1) / np.sqrt(len(x))
    print("--- Radial Scale Parameter (alpha) ---")
    print("Ground truth: 1.0")
    print("Recovered alpha: " + str(np.round(alpha_mle, 4)) + " +/- " + str(np.round(alpha_std, 4)))
    profiles_path = os.path.join(data_dir, "empirical_radial_profiles.npz")
    prof_data = np.load(profiles_path)
    mass_bins = prof_data['mass_bins']
    mean_masses = np.zeros(3)
    for b in range(3):
        m_low = mass_bins[b]
        m_high = mass_bins[b+1]
        masses_in_bin = []
        for i in range(10):
            halo_path = os.path.join(orig_data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
            halo_cat = np.load(halo_path)
            halo_M_vir = halo_cat[:, 6]
            mask = (halo_M_vir >= m_low) & (halo_M_vir < m_high)
            masses_in_bin.append(halo_M_vir[mask])
        masses_in_bin = np.concatenate(masses_in_bin)
        mean_masses[b] = np.mean(masses_in_bin)
    halo_counts = prof_data['halo_counts']
    sat_counts = prof_data['sat_counts']
    mean_halo_counts = np.mean(halo_counts, axis=0)
    mean_sat_counts = np.mean(sat_counts, axis=0)
    N_sat_obs = mean_sat_counts / mean_halo_counts
    safe_halo_counts = np.where(halo_counts > 0, halo_counts, 1)
    N_sat_realizations = sat_counts / safe_halo_counts
    N_sat_err = np.std(N_sat_realizations, axis=0, ddof=1) / np.sqrt(10)
    N_sat_err = np.where(N_sat_err == 0, 1e-5, N_sat_err)
    def model_N_sat_scaled(M_scaled, M_sat_scaled, alpha_sat):
        return (M_scaled / M_sat_scaled)**alpha_sat
    mean_masses_scaled = mean_masses / 1e13
    popt, pcov = curve_fit(model_N_sat_scaled, mean_masses_scaled, N_sat_obs, sigma=N_sat_err, absolute_sigma=True, p0=[1.0, 1.0])
    M_sat_fit = popt[0] * 1e13
    alpha_sat_fit = popt[1]
    M_sat_err = np.sqrt(pcov[0, 0]) * 1e13
    alpha_sat_err = np.sqrt(pcov[1, 1])
    print("\n--- HOD Parameters ---")
    print("Ground truth: M_sat = 1.00e+13 M_sun/h, alpha_sat = 1.0")
    print("Recovered M_sat (referred to as M_min in instructions): " + np.format_float_scientific(M_sat_fit, precision=2) + " +/- " + np.format_float_scientific(M_sat_err, precision=2) + " M_sun/h")
    print("Recovered alpha_sat: " + str(np.round(alpha_sat_fit, 4)) + " +/- " + str(np.round(alpha_sat_err, 4)))
    output_path = os.path.join(data_dir, "recovered_parameters.npz")
    np.savez(output_path, alpha_mle=alpha_mle, alpha_std=alpha_std, M_sat_fit=M_sat_fit, M_sat_err=M_sat_err, alpha_sat_fit=alpha_sat_fit, alpha_sat_err=alpha_sat_err, mean_masses=mean_masses, N_sat_obs=N_sat_obs, N_sat_err=N_sat_err)
    print("\nRecovered parameters saved to " + output_path)

if __name__ == '__main__':
    fit_models()