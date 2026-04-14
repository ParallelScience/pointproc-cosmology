# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import time

def generate_decoupled_model(halos, gals, L_box, rho_crit):
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
    return dec_pos

def compute_2pcf(pos, L_box, r_fine):
    tree = cKDTree(pos, boxsize=L_box)
    counts = tree.count_neighbors(tree, r_fine, cumulative=True)
    DD = np.diff(counts) / 2.0
    N = len(pos)
    V_shell = (4.0 * np.pi / 3.0) * (r_fine[1:]**3 - r_fine[:-1]**3)
    RR = (N * (N - 1) / 2.0) * (V_shell / L_box**3)
    RR = np.where(RR == 0, np.inf, RR)
    xi = DD / RR - 1.0
    return xi

def count_triplets(pos, L_box, r_bins, cos_bins):
    tree = cKDTree(pos, boxsize=L_box)
    r_max = max([r[1] for r in r_bins])
    neighbors = tree.query_ball_point(pos, r=r_max)
    n_cos_bins = len(cos_bins) - 1
    n_r_bins = len(r_bins)
    DDD = np.zeros((n_r_bins, n_cos_bins))
    for i, nbrs in enumerate(neighbors):
        nbrs = np.array(nbrs)
        nbrs = nbrs[nbrs != i]
        n_nbrs = len(nbrs)
        if n_nbrs < 2:
            continue
        pos_i = pos[i]
        pos_nbrs = pos[nbrs]
        dx = pos_nbrs - pos_i
        dx = dx - L_box * np.round(dx / L_box)
        d_i = np.linalg.norm(dx, axis=1)
        idx_j, idx_k = np.triu_indices(n_nbrs, k=1)
        r1 = d_i[idx_j]
        r2 = d_i[idx_k]
        for b_idx, (r_min, r_max_bin) in enumerate(r_bins):
            mask = (r1 >= r_min) & (r1 < r_max_bin) & (r2 >= r_min) & (r2 < r_max_bin)
            if not np.any(mask):
                continue
            r1_m = r1[mask]
            r2_m = r2[mask]
            j_m = idx_j[mask]
            k_m = idx_k[mask]
            dx_jk = pos_nbrs[j_m] - pos_nbrs[k_m]
            dx_jk = dx_jk - L_box * np.round(dx_jk / L_box)
            r3_m = np.linalg.norm(dx_jk, axis=1)
            denom = 2.0 * r1_m * r2_m
            valid = denom > 0
            if not np.any(valid):
                continue
            cos_theta = (r1_m[valid]**2 + r2_m[valid]**2 - r3_m[valid]**2) / denom[valid]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            counts, _ = np.histogram(cos_theta, bins=cos_bins)
            DDD[b_idx] += counts
    return DDD

def compute_zeta(DDD, pos, L_box, r_bins, cos_bins, r_fine, xi):
    n_r_bins = len(r_bins)
    n_cos_bins = len(cos_bins) - 1
    zeta = np.zeros((n_r_bins, n_cos_bins))
    N = len(pos)
    dcos = cos_bins[1] - cos_bins[0]
    r_fine_c = (r_fine[:-1] + r_fine[1:]) / 2.0
    for b_idx, (r_min, r_max_bin) in enumerate(r_bins):
        V = (4.0 * np.pi / 3.0) * (r_max_bin**3 - r_min**3)
        RRR_total = N * (N - 1.0) * (N - 2.0) / 2.0 * (V / L_box**3)**2
        RRR_bin = RRR_total * (dcos / 2.0)
        r_c = 0.75 * (r_max_bin**4 - r_min**4) / (r_max_bin**3 - r_min**3)
        xi_1 = np.interp(r_c, r_fine_c, xi)
        for c_idx in range(n_cos_bins):
            cos_c = (cos_bins[c_idx] + cos_bins[c_idx+1]) / 2.0
            r3 = r_c * np.sqrt(2.0 - 2.0 * cos_c)
            xi_3 = np.interp(r3, r_fine_c, xi)
            if RRR_bin > 0:
                zeta[b_idx, c_idx] = DDD[b_idx, c_idx] / RRR_bin - 1.0 - 2.0 * xi_1 - xi_3
            else:
                zeta[b_idx, c_idx] = 0.0
    return zeta

if __name__ == '__main__':
    input_data_dir = "/home/node/work/projects/pointproc_cosmology/data/"
    output_data_dir = "data/"
    L_box = 500.0
    rho_crit = 2.77536627e11
    r_bins = [(1.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    cos_bins = np.linspace(-1, 1, 11)
    cos_centers = (cos_bins[:-1] + cos_bins[1:]) / 2.0
    r_fine = np.linspace(0, 20, 201)
    zeta_emp_all = []
    zeta_dec_all = []
    print("--- Three-Point Correlation Function (3PCF) Analysis ---")
    print("Processing 10 realizations...")
    start_time = time.time()
    for i in range(10):
        np.random.seed(42 + i)
        halo_file = os.path.join(input_data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_file = os.path.join(input_data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        emp_pos = gals[:, :3]
        dec_pos = generate_decoupled_model(halos, gals, L_box, rho_crit)
        xi_emp = compute_2pcf(emp_pos, L_box, r_fine)
        DDD_emp = count_triplets(emp_pos, L_box, r_bins, cos_bins)
        zeta_emp = compute_zeta(DDD_emp, emp_pos, L_box, r_bins, cos_bins, r_fine, xi_emp)
        zeta_emp_all.append(zeta_emp)
        xi_dec = compute_2pcf(dec_pos, L_box, r_fine)
        DDD_dec = count_triplets(dec_pos, L_box, r_bins, cos_bins)
        zeta_dec = compute_zeta(DDD_dec, dec_pos, L_box, r_bins, cos_bins, r_fine, xi_dec)
        zeta_dec_all.append(zeta_dec)
        print("  Realization " + str(i).zfill(2) + " done. Time elapsed: " + str(np.round(time.time() - start_time, 1)) + "s")
    zeta_emp_all = np.array(zeta_emp_all)
    zeta_dec_all = np.array(zeta_dec_all)
    mean_zeta_emp = np.mean(zeta_emp_all, axis=0)
    mean_zeta_dec = np.mean(zeta_dec_all, axis=0)
    std_zeta_emp = np.std(zeta_emp_all, axis=0, ddof=1)
    std_zeta_dec = np.std(zeta_dec_all, axis=0, ddof=1)
    output_file = os.path.join(output_data_dir, "3pcf_results.npz")
    np.savez(output_file, r_bins=np.array(r_bins), cos_bins=cos_bins, cos_centers=cos_centers, zeta_emp_all=zeta_emp_all, zeta_dec_all=zeta_dec_all)
    print("\nResults saved to " + output_file)
    print("\n--- 3PCF Summary (Mean +/- Std across 10 realizations) ---")
    eq_idx = 7
    sq_idx = 9
    print("\nEquilateral Configurations (cos(theta) ~ 0.5):")
    for b_idx, (r_min, r_max_bin) in enumerate(r_bins):
        r_c = 0.75 * (r_max_bin**4 - r_min**4) / (r_max_bin**3 - r_min**3)
        emp_val = mean_zeta_emp[b_idx, eq_idx]
        emp_err = std_zeta_emp[b_idx, eq_idx]
        dec_val = mean_zeta_dec[b_idx, eq_idx]
        dec_err = std_zeta_dec[b_idx, eq_idx]
        print("  r1, r2 in [" + str(r_min) + ", " + str(r_max_bin) + "] (rc~" + str(np.round(r_c, 1)) + "):")
        print("    Empirical: " + str(np.round(emp_val, 1)) + " +/- " + str(np.round(emp_err, 1)))
        print("    Decoupled: " + str(np.round(dec_val, 1)) + " +/- " + str(np.round(dec_err, 1)))
    print("\nSqueezed Configurations (cos(theta) ~ 0.9):")
    for b_idx, (r_min, r_max_bin) in enumerate(r_bins):
        r_c = 0.75 * (r_max_bin**4 - r_min**4) / (r_max_bin**3 - r_min**3)
        emp_val = mean_zeta_emp[b_idx, sq_idx]
        emp_err = std_zeta_emp[b_idx, sq_idx]
        dec_val = mean_zeta_dec[b_idx, sq_idx]
        dec_err = std_zeta_dec[b_idx, sq_idx]
        print("  r1, r2 in [" + str(r_min) + ", " + str(r_max_bin) + "] (rc~" + str(np.round(r_c, 1)) + "):")
        print("    Empirical: " + str(np.round(emp_val, 1)) + " +/- " + str(np.round(emp_err, 1)))
        print("    Decoupled: " + str(np.round(dec_val, 1)) + " +/- " + str(np.round(dec_err, 1)))