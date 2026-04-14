# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

def extract_radial_profiles():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    satellites_path = os.path.join(output_dir, "processed_satellites.npy")
    satellites = np.load(satellites_path)
    max_mass = 0.0
    halo_masses_all = []
    for i in range(10):
        halo_path = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        halo_cat = np.load(halo_path)
        halo_M_vir = halo_cat[:, 6]
        max_mass = max(max_mass, np.max(halo_M_vir))
        halo_masses_all.append(halo_M_vir)
    mass_bins = np.logspace(13, np.log10(max_mass * 1.01), 4)
    r_bins = np.linspace(0.0, 5.0, 21)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    shell_volumes = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    profiles = np.zeros((10, 3, len(r_centers)))
    halo_counts = np.zeros((10, 3))
    sat_counts = np.zeros((10, 3))
    for i in range(10):
        halo_M_vir = halo_masses_all[i]
        sat_mask = satellites['realization_id'] == i
        sats_i = satellites[sat_mask]
        for b in range(3):
            m_low = mass_bins[b]
            m_high = mass_bins[b+1]
            h_mask = (halo_M_vir >= m_low) & (halo_M_vir < m_high)
            n_halos = np.sum(h_mask)
            halo_counts[i, b] = n_halos
            s_mask = (sats_i['halo_mass'] >= m_low) & (sats_i['halo_mass'] < m_high)
            sats_in_bin = sats_i[s_mask]
            n_sats = len(sats_in_bin)
            sat_counts[i, b] = n_sats
            if n_halos > 0:
                counts, _ = np.histogram(sats_in_bin['r_sat'], bins=r_bins)
                profile = counts / shell_volumes / n_halos
                profiles[i, b, :] = profile
    mean_profiles = np.mean(profiles, axis=0)
    var_profiles = np.var(profiles, axis=0, ddof=1)
    print("Mass Bins (M_sun/h):")
    for b in range(3):
        m_low_str = np.format_float_scientific(mass_bins[b], precision=2)
        m_high_str = np.format_float_scientific(mass_bins[b+1], precision=2)
        print("  Bin " + str(b) + ": [" + m_low_str + ", " + m_high_str + "]")
    print("\nBin Populations and Mean Satellite Counts (averaged across 10 realizations):")
    mean_halo_counts = np.mean(halo_counts, axis=0)
    mean_sat_counts = np.mean(sat_counts, axis=0)
    for b in range(3):
        mean_sats_per_halo = mean_sat_counts[b] / mean_halo_counts[b] if mean_halo_counts[b] > 0 else 0
        print("  Bin " + str(b) + ":")
        print("    Mean number of halos: " + str(np.round(mean_halo_counts[b], 1)))
        print("    Mean number of satellites: " + str(np.round(mean_sat_counts[b], 1)))
        print("    Mean satellites per halo: " + str(np.round(mean_sats_per_halo, 4)))
    output_path = os.path.join(output_dir, "empirical_radial_profiles.npz")
    np.savez(output_path, mass_bins=mass_bins, r_bins=r_bins, r_centers=r_centers, profiles=profiles, mean_profiles=mean_profiles, var_profiles=var_profiles, halo_counts=halo_counts, sat_counts=sat_counts)
    print("\nEmpirical profiles and statistics saved to " + output_path)

if __name__ == '__main__':
    extract_radial_profiles()