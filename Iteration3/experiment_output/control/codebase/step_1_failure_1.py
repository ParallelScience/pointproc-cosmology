# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.stats import ks_1samp

def cdf_exp_profile(x):
    return 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2)

if __name__ == '__main__':
    data_dir = "data/"
    rho_crit = 2.77536627e11
    L_box = 500.0
    mass_bins = [(13.0, 13.5), (13.5, 14.0), (14.0, np.inf)]
    aggregated_x = {i: [] for i in range(len(mass_bins))}
    total_qualifying_halos = 0
    total_satellites = 0
    print("--- Empirical Satellite Density Profiling ---")
    for i in range(10):
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        M_vir = halos[:, 6]
        qualifying_mask = M_vir > 1e13
        n_qualifying_halos = np.sum(qualifying_mask)
        sats = gals[gals[:, 8] == 0]
        n_sats = len(sats)
        print("Realization " + str(i).zfill(2) + ": " + str(n_qualifying_halos) + " qualifying halos, " + str(n_sats) + " satellites")
        total_qualifying_halos += n_qualifying_halos
        total_satellites += n_sats
        halo_ln_M = np.log(M_vir)
        sat_ln_M = sats[:, 7]
        sort_idx = np.argsort(halo_ln_M)
        sorted_halo_ln_M = halo_ln_M[sort_idx]
        idx = np.searchsorted(sorted_halo_ln_M, sat_ln_M)
        idx = np.clip(idx, 1, len(sorted_halo_ln_M) - 1)
        left_diff = np.abs(sat_ln_M - sorted_halo_ln_M[idx - 1])
        right_diff = np.abs(sat_ln_M - sorted_halo_ln_M[idx])
        closest_sorted_idx = np.where(left_diff < right_diff, idx - 1, idx)
        matched_halo_idx = sort_idx[closest_sorted_idx]
        matched_M_vir = M_vir[matched_halo_idx]
        R_vir = (3.0 * matched_M_vir / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        dx = np.abs(sats[:, 0] - halos[matched_halo_idx, 0])
        dx = np.minimum(dx, L_box - dx)
        dy = np.abs(sats[:, 1] - halos[matched_halo_idx, 1])
        dy = np.minimum(dy, L_box - dy)
        dz = np.abs(sats[:, 2] - halos[matched_halo_idx, 2])
        dz = np.minimum(dz, L_box - dz)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        x = r / R_vir
        log10_M = np.log10(matched_M_vir)
        for b_idx, (m_low, m_high) in enumerate(mass_bins):
            mask = (log10_M >= m_low) & (log10_M < m_high)
            aggregated_x[b_idx].extend(x[mask])
    print("Total across 10 realizations: " + str(total_qualifying_halos) + " qualifying halos, " + str(total_satellites) + " satellites")
    bins = np.linspace(0, 10, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_volumes = (4.0 * np.pi / 3.0) * (bins[1:]**3 - bins[:-1]**3)
    n_theoretical = np.exp(-bin_centers) / (8.0 * np.pi)
    results = {'mass_bins': np.array(mass_bins), 'bin_centers': bin_centers, 'n_theoretical': n_theoretical}
    for b_idx, (m_low, m_high) in enumerate(mass_bins):
        x_vals = np.array(aggregated_x[b_idx])
        if len(x_vals) == 0:
            continue
        stat, p_value = ks_1samp(x_vals, cdf_exp_profile)
        counts, _ = np.histogram(x_vals, bins=bins)
        n_empirical = counts / bin_volumes / len(x_vals)
        results['n_empirical_bin_' + str(b_idx)] = n_empirical
        results['ks_stat_bin_' + str(b_idx)] = stat
        results['ks_pvalue_bin_' + str(b_idx)] = p_value
        results['x_vals_bin_' + str(b_idx)] = x_vals
    all_x_vals = np.concatenate([aggregated_x[b_idx] for b_idx in range(len(mass_bins))])
    overall_stat, overall_p_value = ks_1samp(all_x_vals, cdf_exp_profile)
    results['overall_ks_stat'] = overall_stat
    results['overall_ks_pvalue'] = overall_p_value
    results['all_x_vals'] = all_x_vals
    output_file = os.path.join(data_dir, "empirical_satellite_profile.npz")
    np.savez(output_file, **results)
    print("Aggregated results saved to " + output_file)