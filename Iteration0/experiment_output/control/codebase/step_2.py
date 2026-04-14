# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

def main():
    data_dir = "data/"
    mass_bins = [13.0, 13.5, 14.0, 14.5, 15.5]
    n_mass_bins = len(mass_bins) - 1
    x_bins = np.linspace(0, 10, 101)
    x_bin_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    V_shell_x = (4.0 / 3.0) * np.pi * (x_bins[1:]**3 - x_bins[:-1]**3)
    r_bins = np.linspace(0, 5, 51)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    V_shell_r = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    all_counts_x = np.zeros((10, n_mass_bins, len(x_bin_centers)))
    all_density_norm_x = np.zeros((10, n_mass_bins, len(x_bin_centers)))
    all_density_phys_x = np.zeros((10, n_mass_bins, len(x_bin_centers)))
    all_counts_r = np.zeros((10, n_mass_bins, len(r_bin_centers)))
    all_density_r = np.zeros((10, n_mass_bins, len(r_bin_centers)))
    all_n_halos = np.zeros((10, n_mass_bins))
    all_n_sats = np.zeros((10, n_mass_bins))
    print("Computing stacked radial density profiles for satellite galaxies...\n")
    for i in range(10):
        in_path = os.path.join(data_dir, "matched_catalog_" + str(i).zfill(2) + ".npz")
        data = np.load(in_path)
        galaxies = data['galaxies']
        halos = data['halos']
        halo_R_vir = data['halo_R_vir']
        matched_halo_idx = data['matched_halo_idx']
        is_sat = galaxies[:, 8] == 0
        sat_galaxies = galaxies[is_sat]
        sat_halo_idx = matched_halo_idx[is_sat]
        sat_pos = sat_galaxies[:, :3]
        halo_pos = halos[sat_halo_idx, :3]
        dx = sat_pos - halo_pos
        dx = dx - 500.0 * np.round(dx / 500.0)
        r = np.linalg.norm(dx, axis=1)
        R_vir_sat = halo_R_vir[sat_halo_idx]
        x = r / R_vir_sat
        halo_M_vir = halos[:, 6]
        halo_logM = np.log10(halo_M_vir)
        print("Realization " + str(i).zfill(2) + ":")
        print("  Total satellites: " + str(len(sat_galaxies)))
        print("  Max physical distance r: " + str(np.round(np.max(r), 4)) + " Mpc/h")
        print("  Max normalized distance x = r/R_vir: " + str(np.round(np.max(x), 4)))
        for b in range(n_mass_bins):
            mask_halo = (halo_logM >= mass_bins[b]) & (halo_logM < mass_bins[b+1])
            n_halos = np.sum(mask_halo)
            all_n_halos[i, b] = n_halos
            if n_halos > 0:
                sum_R_vir_3 = np.sum(halo_R_vir[mask_halo]**3)
            else:
                sum_R_vir_3 = 0.0
            sat_halo_logM = halo_logM[sat_halo_idx]
            mask_sat = (sat_halo_logM >= mass_bins[b]) & (sat_halo_logM < mass_bins[b+1])
            x_sat = x[mask_sat]
            r_sat = r[mask_sat]
            all_n_sats[i, b] = len(x_sat)
            counts_x, _ = np.histogram(x_sat, bins=x_bins)
            all_counts_x[i, b, :] = counts_x
            if n_halos > 0:
                all_density_norm_x[i, b, :] = counts_x / (n_halos * V_shell_x)
                all_density_phys_x[i, b, :] = counts_x / (sum_R_vir_3 * V_shell_x)
            counts_r, _ = np.histogram(r_sat, bins=r_bins)
            all_counts_r[i, b, :] = counts_r
            if n_halos > 0:
                all_density_r[i, b, :] = counts_r / (n_halos * V_shell_r)
            print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "): " + str(n_halos) + " halos, " + str(len(x_sat)) + " satellites")
        print("-" * 40)
    out_path = os.path.join(data_dir, "radial_profiles.npz")
    np.savez(out_path, mass_bins=mass_bins, x_bins=x_bins, x_bin_centers=x_bin_centers, r_bins=r_bins, r_bin_centers=r_bin_centers, counts_x=all_counts_x, density_norm_x=all_density_norm_x, density_phys_x=all_density_phys_x, counts_r=all_counts_r, density_r=all_density_r, n_halos=all_n_halos, n_sats=all_n_sats)
    print("Saved radial profiles to " + out_path)
    print("\nSummary of Realization 00 Density Profiles (first 5 bins in x):")
    for b in range(n_mass_bins):
        print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "):")
        print("    x_centers:      " + str(np.round(x_bin_centers[:5], 4)))
        print("    density_norm_x: " + str(np.round(all_density_norm_x[0, b, :5], 4)))

if __name__ == '__main__':
    main()