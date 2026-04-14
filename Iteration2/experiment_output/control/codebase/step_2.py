# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.spatial import cKDTree

def main():
    data_dir = "data/"
    n_realizations = 10
    n_sub = 20
    n_centers = 5000
    L = 500.0
    M_min = 3e11
    M_sat = 1e13
    alpha_sat = 1.0
    r_bins = np.logspace(np.log10(1), np.log10(30), 20)
    r_idx_1 = np.argmin(np.abs(r_bins - 1.0))
    r_idx_5 = np.argmin(np.abs(r_bins - 5.0))
    r_idx_10 = np.argmin(np.abs(r_bins - 10.0))
    r_idx_30 = np.argmin(np.abs(r_bins - 30.0))
    all_vpf_mean = []
    all_vpf_std = []
    all_vpf_sub = []
    print("Starting Monte Carlo Theoretical VPF Construction...\n")
    for i in range(n_realizations):
        halo_path = '/home/node/work/projects/pointproc_cosmology/data/halo_catalog_' + str(i).zfill(2) + '.npy'
        halos = np.load(halo_path)
        M = halos[:, 6]
        pos = halos[:, 0:3]
        if halos.shape[1] >= 8:
            R_vir = halos[:, 7]
        else:
            Omega_m = 0.311
            rho_c = 2.77536627e11
            rho_m = Omega_m * rho_c
            Delta_m = 200.0
            R_vir = (M / (4.0/3.0 * np.pi * Delta_m * rho_m))**(1.0/3.0)
        central_mask = M >= M_min
        central_pos = pos[central_mask]
        sat_mask = M >= M_sat
        M_sat_halos = M[sat_mask]
        R_vir_sat = R_vir[sat_mask]
        pos_sat_halos = pos[sat_mask]
        N_sat_expected = (M_sat_halos / M_sat)**alpha_sat
        vpf_sub = np.zeros((n_sub, len(r_bins)))
        n_gals_sub = np.zeros(n_sub)
        for j in range(n_sub):
            N_sat_actual = np.random.poisson(N_sat_expected)
            total_sats = np.sum(N_sat_actual)
            if total_sats > 0:
                sat_centers = np.repeat(pos_sat_halos, N_sat_actual, axis=0)
                sat_R_vir = np.repeat(R_vir_sat, N_sat_actual)
                r = np.random.gamma(shape=3.0, scale=sat_R_vir)
                phi = np.random.uniform(0, 2*np.pi, total_sats)
                costheta = np.random.uniform(-1, 1, total_sats)
                sintheta = np.sqrt(1 - costheta**2)
                dx = r * sintheta * np.cos(phi)
                dy = r * sintheta * np.sin(phi)
                dz = r * costheta
                sat_pos = sat_centers + np.column_stack((dx, dy, dz))
                all_pos = np.vstack((central_pos, sat_pos))
            else:
                all_pos = central_pos
            all_pos = all_pos % L
            n_gals_sub[j] = len(all_pos)
            random_centers = np.random.uniform(0, L, (n_centers, 3))
            tree = cKDTree(all_pos, boxsize=L)
            d_min, _ = tree.query(random_centers, k=1)
            for k, r_val in enumerate(r_bins):
                vpf_sub[j, k] = np.mean(d_min > r_val)
        vpf_mean = np.mean(vpf_sub, axis=0)
        vpf_std = np.std(vpf_sub, axis=0)
        all_vpf_mean.append(vpf_mean)
        all_vpf_std.append(vpf_std)
        all_vpf_sub.append(vpf_sub)
        print("Realization " + str(i) + ":")
        print("  Average number of simulated galaxies: " + str(np.round(np.mean(n_gals_sub), 1)))
        print("  VPF(r=" + str(np.round(r_bins[r_idx_1], 2)) + ") = " + str(np.round(vpf_mean[r_idx_1], 4)) + " +/- " + str(np.round(vpf_std[r_idx_1], 4)))
        print("  VPF(r=" + str(np.round(r_bins[r_idx_5], 2)) + ") = " + str(np.round(vpf_mean[r_idx_5], 4)) + " +/- " + str(np.round(vpf_std[r_idx_5], 4)))
        print("  VPF(r=" + str(np.round(r_bins[r_idx_10], 2)) + ") = " + str(np.round(vpf_mean[r_idx_10], 4)) + " +/- " + str(np.round(vpf_std[r_idx_10], 4)))
        print("  VPF(r=" + str(np.round(r_bins[r_idx_30], 2)) + ") = " + str(np.round(vpf_mean[r_idx_30], 4)) + " +/- " + str(np.round(vpf_std[r_idx_30], 4)))
        print("-" * 40)
    all_vpf_mean = np.array(all_vpf_mean)
    all_vpf_std = np.array(all_vpf_std)
    all_vpf_sub = np.array(all_vpf_sub)
    output_file = os.path.join(data_dir, "theoretical_vpf.npz")
    np.savez(output_file, r_bins=r_bins, vpf_mean=all_vpf_mean, vpf_std=all_vpf_std, vpf_all=all_vpf_sub)
    print("\nTheoretical VPFs successfully saved to " + output_file)

if __name__ == '__main__':
    main()