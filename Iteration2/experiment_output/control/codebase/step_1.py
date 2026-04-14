# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.stats import kstest
from scipy.spatial import cKDTree

def theoretical_cdf(x):
    return 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2)

def main():
    data_dir = "data/"
    n_realizations = 10
    ks_stats = []
    p_values = []
    mass_bins = [1e13, 10**13.5, 1e14, 1e15]
    r_norm_bins = np.linspace(0, 5, 51)
    r_norm_centers = 0.5 * (r_norm_bins[:-1] + r_norm_bins[1:])
    V_shell = (4.0/3.0) * np.pi * (r_norm_bins[1:]**3 - r_norm_bins[:-1]**3)
    all_empirical_profiles = []
    for i in range(n_realizations):
        gal_path = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + str(i).zfill(2) + '.npy'
        halo_path = '/home/node/work/projects/pointproc_cosmology/data/halo_catalog_' + str(i).zfill(2) + '.npy'
        galaxies = np.load(gal_path)
        halos = np.load(halo_path)
        is_central = galaxies[:, 8].astype(bool)
        satellites = galaxies[~is_central]
        ln_M_halo = np.log(halos[:, 6])
        tree = cKDTree(ln_M_halo[:, None])
        _, halo_indices = tree.query(satellites[:, 7:8])
        matched_halos = halos[halo_indices]
        if halos.shape[1] >= 8:
            R_vir = matched_halos[:, 7]
        else:
            Omega_m = 0.311
            rho_c = 2.77536627e11
            rho_m = Omega_m * rho_c
            Delta_m = 200.0
            R_vir = (matched_halos[:, 6] / (4.0/3.0 * np.pi * Delta_m * rho_m))**(1.0/3.0)
        dx = satellites[:, 0] - matched_halos[:, 0]
        dy = satellites[:, 1] - matched_halos[:, 1]
        dz = satellites[:, 2] - matched_halos[:, 2]
        L = 500.0
        dx = dx - L * np.round(dx / L)
        dy = dy - L * np.round(dy / L)
        dz = dz - L * np.round(dz / L)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r_norm = r / R_vir
        stat, p_val = kstest(r_norm, theoretical_cdf)
        ks_stats.append(stat)
        p_values.append(p_val)
        print("Realization " + str(i) + ":")
        print("  Total galaxies: " + str(len(galaxies)) + ", Satellites: " + str(len(satellites)))
        print("  Mean r_norm: " + str(np.round(np.mean(r_norm), 4)))
        print("  Max r_norm: " + str(np.round(np.max(r_norm), 4)))
        print("  KS D-statistic: " + str(np.round(stat, 4)) + ", p-value: " + str(p_val))
        profiles = []
        for j in range(len(mass_bins)-1):
            m_low = mass_bins[j]
            m_high = mass_bins[j+1]
            halo_mask = (halos[:, 6] >= m_low) & (halos[:, 6] < m_high)
            n_halos_in_bin = np.sum(halo_mask)
            sat_mask = (matched_halos[:, 6] >= m_low) & (matched_halos[:, 6] < m_high)
            r_norm_sat = r_norm[sat_mask]
            n_sats_in_bin = len(r_norm_sat)
            mean_sats = n_sats_in_bin / n_halos_in_bin if n_halos_in_bin > 0 else 0
            print("  Mass bin [" + str(m_low) + ", " + str(m_high) + "): " + str(n_halos_in_bin) + " halos, " + str(n_sats_in_bin) + " satellites, " + str(np.round(mean_sats, 2)) + " sats/halo")
            counts, _ = np.histogram(r_norm_sat, bins=r_norm_bins)
            if n_halos_in_bin > 0:
                density = counts / (n_halos_in_bin * V_shell)
            else:
                density = np.zeros_like(counts, dtype=float)
            profiles.append(density)
        all_empirical_profiles.append(profiles)
        print("-" * 40)
    ks_stats = np.array(ks_stats)
    p_values = np.array(p_values)
    all_empirical_profiles = np.array(all_empirical_profiles)
    print("\n=== Aggregated KS Test Statistics (10 Realizations) ===")
    print("Mean D-statistic: " + str(np.round(np.mean(ks_stats), 4)) + " +/- " + str(np.round(np.std(ks_stats), 4)))
    print("Mean p-value: " + str(np.mean(p_values)) + " +/- " + str(np.std(p_values)))
    output_file = os.path.join(data_dir, "empirical_satellite_profiles.npz")
    np.savez(output_file, ks_stats=ks_stats, p_values=p_values, mass_bins=mass_bins, r_norm_bins=r_norm_bins, r_norm_centers=r_norm_centers, empirical_profiles=all_empirical_profiles)
    print("\nResults successfully saved to " + output_file)

if __name__ == '__main__':
    main()