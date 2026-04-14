# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree

if __name__ == '__main__':
    input_data_dir = "/home/node/work/projects/pointproc_cosmology/data/"
    output_data_dir = "data/"
    L_box = 500.0
    rho_crit = 2.77536627e11
    r_edges = np.logspace(np.log10(0.5), np.log10(20.0), 16)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    V_shell = (4.0 * np.pi / 3.0) * (r_edges[1:]**3 - r_edges[:-1]**3)
    V_box = L_box**3
    RR_n_analytical = V_shell / V_box
    xi_emp_all = []
    xi_dec_all = []
    print("--- Critical Scale Determination (r_crit) ---")
    print("Processing 10 realizations...")
    for i in range(10):
        np.random.seed(42 + i)
        halo_file = os.path.join(input_data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_file = os.path.join(input_data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        is_central = gals[:, 8] == 1
        centrals = gals[is_central]
        sats = gals[~is_central]
        M_vir = halos[:, 6]
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
        r_new = np.random.gamma(shape=3.0, scale=R_vir)
        cos_theta = np.random.uniform(-1, 1, size=len(sats))
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2*np.pi, size=len(sats))
        dx = r_new * sin_theta * np.cos(phi)
        dy = r_new * sin_theta * np.sin(phi)
        dz = r_new * cos_theta
        host_x = halos[matched_halo_idx, 0]
        host_y = halos[matched_halo_idx, 1]
        host_z = halos[matched_halo_idx, 2]
        new_sat_x = (host_x + dx) % L_box
        new_sat_y = (host_y + dy) % L_box
        new_sat_z = (host_z + dz) % L_box
        new_sats_pos = np.column_stack((new_sat_x, new_sat_y, new_sat_z))
        centrals_pos = centrals[:, :3]
        dec_pos = np.vstack((centrals_pos, new_sats_pos))
        emp_pos = gals[:, :3]
        N_gal = len(emp_pos)
        N_rand = 5 * N_gal
        rand_pos = np.random.uniform(0, L_box, (N_rand, 3))
        tree_emp = cKDTree(emp_pos, boxsize=L_box)
        tree_dec = cKDTree(dec_pos, boxsize=L_box)
        tree_rand = cKDTree(rand_pos, boxsize=L_box)
        def get_counts(t1, t2, is_auto):
            counts = t1.count_neighbors(t2, r_edges, cumulative=True)
            diff_counts = np.diff(counts)
            if is_auto:
                return diff_counts / 2.0
            return diff_counts
        DD_emp = get_counts(tree_emp, tree_emp, True)
        DR_emp = get_counts(tree_emp, tree_rand, False)
        DD_dec = get_counts(tree_dec, tree_dec, True)
        DR_dec = get_counts(tree_dec, tree_rand, False)
        RR = get_counts(tree_rand, tree_rand, True)
        f_D = N_gal * (N_gal - 1) / 2.0
        f_DR = N_gal * N_rand
        f_R = N_rand * (N_rand - 1) / 2.0
        DD_emp_n = DD_emp / f_D
        DR_emp_n = DR_emp / f_DR
        DD_dec_n = DD_dec / f_D
        DR_dec_n = DR_dec / f_DR
        RR_n = RR / f_R
        RR_n = np.where(RR_n == 0, RR_n_analytical, RR_n)
        xi_emp = (DD_emp_n - 2.0 * DR_emp_n + RR_n) / RR_n
        xi_dec = (DD_dec_n - 2.0 * DR_dec_n + RR_n) / RR_n
        xi_emp_all.append(xi_emp)
        xi_dec_all.append(xi_dec)
        print("  Realization " + str(i).zfill(2) + " done.")
    xi_emp_all = np.array(xi_emp_all)
    xi_dec_all = np.array(xi_dec_all)
    delta_xi = xi_emp_all - xi_dec_all
    mean_xi_emp = np.mean(xi_emp_all, axis=0)
    mean_xi_dec = np.mean(xi_dec_all, axis=0)
    mean_delta_xi = np.mean(delta_xi, axis=0)
    std_delta_xi = np.std(delta_xi, axis=0, ddof=1)
    err_delta_xi = std_delta_xi / np.sqrt(10.0)
    rel_diff = mean_delta_xi / np.where(mean_xi_emp == 0, np.inf, mean_xi_emp)
    is_within_1sigma = np.abs(mean_delta_xi) < err_delta_xi
    r_crit = None
    for j in range(len(r_centers)):
        if is_within_1sigma[j]:
            if j > 0:
                r_prev = r_centers[j-1]
                r_curr = r_centers[j]
                val_prev = np.abs(mean_delta_xi[j-1]) - err_delta_xi[j-1]
                val_curr = np.abs(mean_delta_xi[j]) - err_delta_xi[j]
                fraction = val_prev / (val_prev - val_curr)
                r_crit = r_prev + fraction * (r_curr - r_prev)
            else:
                r_crit = r_centers[j]
            break
    print("\n--- Summary across 10 realizations ---")
    if r_crit is not None:
        print("Critical scale r_crit determined at: " + str(np.round(r_crit, 3)) + " Mpc/h")
    else:
        print("Critical scale r_crit not found within the probed range.")
    output_file = os.path.join(output_data_dir, "rcrit_results.npz")
    np.savez(output_file, r_edges=r_edges, r_centers=r_centers, xi_emp_all=xi_emp_all, xi_dec_all=xi_dec_all, r_crit=r_crit if r_crit is not None else -1.0)
    print("Results saved to " + output_file)