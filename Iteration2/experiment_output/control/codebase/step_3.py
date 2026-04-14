# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree

def generate_mc_catalog(halos, L, M_min, M_sat, alpha_sat):
    if len(halos) == 0:
        return np.zeros((0, 3))
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
    if len(M_sat_halos) > 0:
        N_sat_expected = (M_sat_halos / M_sat)**alpha_sat
        N_sat_actual = np.random.poisson(N_sat_expected)
        total_sats = np.sum(N_sat_actual)
    else:
        total_sats = 0
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
        if len(central_pos) > 0:
            all_pos = np.vstack((central_pos, sat_pos))
        else:
            all_pos = sat_pos
    else:
        all_pos = central_pos
    if len(all_pos) > 0:
        return all_pos % L
    else:
        return np.zeros((0, 3))

def compute_vpf(pos, random_centers, r_bins, L):
    if len(pos) == 0:
        return np.ones_like(r_bins)
    tree = cKDTree(pos, boxsize=L)
    d_min, _ = tree.query(random_centers, k=1)
    return np.array([np.mean(d_min > r) for r in r_bins])

def main():
    data_dir = "data/"
    n_realizations = 10
    L = 500.0
    M_min = 3e11
    M_sat = 1e13
    alpha_sat = 1.0
    theo_vpf_data = np.load(os.path.join(data_dir, "theoretical_vpf.npz"))
    r_bins = theo_vpf_data['r_bins']
    vpf_mean_theo = theo_vpf_data['vpf_mean']
    mass_bins = [1e13, 10**13.5, 1e14, 1e15]
    gamma_grid = np.round(np.arange(0.1, 1.1, 0.1), 1)
    global_residuals = []
    mass_bin_residuals = []
    gamma_estimates = []
    pseudo_likelihoods = []
    for i in range(n_realizations):
        gal_path = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + str(i).zfill(2) + '.npy'
        halo_path = '/home/node/work/projects/pointproc_cosmology/data/halo_catalog_' + str(i).zfill(2) + '.npy'
        galaxies = np.load(gal_path)
        halos = np.load(halo_path)
        tree_gal = cKDTree(galaxies[:, 0:3], boxsize=L)
        random_centers = np.random.uniform(0, L, (5000, 3))
        d_min, _ = tree_gal.query(random_centers, k=1)
        vpf_emp = np.array([np.mean(d_min > r) for r in r_bins])
        res_global = vpf_emp - vpf_mean_theo[i]
        global_residuals.append(res_global)
        ln_M_halo = np.log(halos[:, 6])
        tree_halo = cKDTree(ln_M_halo[:, None])
        _, halo_indices = tree_halo.query(galaxies[:, 7:8])
        matched_halos = halos[halo_indices]
        res_bins = []
        for j in range(len(mass_bins)-1):
            m_low = mass_bins[j]
            m_high = mass_bins[j+1]
            mask_gal = (matched_halos[:, 6] >= m_low) & (matched_halos[:, 6] < m_high)
            galaxies_bin = galaxies[mask_gal]
            vpf_emp_bin = compute_vpf(galaxies_bin[:, 0:3], random_centers, r_bins, L)
            mask_halo = (halos[:, 6] >= m_low) & (halos[:, 6] < m_high)
            halos_bin = halos[mask_halo]
            vpf_theo_bin_sum = np.zeros_like(r_bins)
            n_mc = 10
            for _ in range(n_mc):
                mc_pos = generate_mc_catalog(halos_bin, L, M_min, M_sat, alpha_sat)
                vpf_theo_bin_sum += compute_vpf(mc_pos, random_centers, r_bins, L)
            vpf_theo_bin = vpf_theo_bin_sum / n_mc
            res_bins.append(vpf_emp_bin - vpf_theo_bin)
        mass_bin_residuals.append(res_bins)
        pairs = tree_gal.query_pairs(r=1.0)
        S = 2 * len(pairs)
        central_mask = halos[:, 6] >= M_min
        central_pos = halos[central_mask, 0:3]
        t_cen = tree_gal.query_ball_point(central_pos, r=1.0, return_length=True)
        sat_mask = halos[:, 6] >= M_sat
        sat_halos = halos[sat_mask]
        M_sat_halos = sat_halos[:, 6]
        if sat_halos.shape[1] >= 8:
            R_vir_sat = sat_halos[:, 7]
        else:
            R_vir_sat = (M_sat_halos / (4.0/3.0 * np.pi * 200.0 * 0.311 * 2.77536627e11))**(1.0/3.0)
        N_expected = (M_sat_halos / M_sat)**alpha_sat
        N_dummy = 50
        dummy_centers = np.repeat(sat_halos[:, 0:3], N_dummy, axis=0)
        dummy_R_vir = np.repeat(R_vir_sat, N_dummy)
        r_dummy = np.random.gamma(shape=3.0, scale=dummy_R_vir)
        phi = np.random.uniform(0, 2*np.pi, len(r_dummy))
        costheta = np.random.uniform(-1, 1, len(r_dummy))
        sintheta = np.sqrt(1 - costheta**2)
        dx = r_dummy * sintheta * np.cos(phi)
        dy = r_dummy * sintheta * np.sin(phi)
        dz = r_dummy * costheta
        dummy_pos = (dummy_centers + np.column_stack((dx, dy, dz))) % L
        t_sat = tree_gal.query_ball_point(dummy_pos, r=1.0, return_length=True)
        t_sat_reshaped = t_sat.reshape(-1, N_dummy)
        L_gamma = np.zeros_like(gamma_grid)
        for idx, g in enumerate(gamma_grid):
            I_cen = np.sum(g**t_cen)
            gamma_t_sat = g**t_sat_reshaped
            mean_gamma_t_sat = np.mean(gamma_t_sat, axis=1)
            I_sat = np.sum(N_expected * mean_gamma_t_sat)
            L_gamma[idx] = S * np.log(g) - (I_cen + I_sat)
        best_gamma = gamma_grid[np.argmax(L_gamma)]
        gamma_estimates.append(best_gamma)
        pseudo_likelihoods.append(L_gamma)
    global_residuals = np.array(global_residuals)
    mass_bin_residuals = np.array(mass_bin_residuals)
    gamma_estimates = np.array(gamma_estimates)
    pseudo_likelihoods = np.array(pseudo_likelihoods)
    output_file = os.path.join(data_dir, "residual_vpf_strauss.npz")
    np.savez(output_file, r_bins=r_bins, mass_bins=mass_bins, gamma_grid=gamma_grid, global_residuals=global_residuals, mass_bin_residuals=mass_bin_residuals, gamma_estimates=gamma_estimates, pseudo_likelihoods=pseudo_likelihoods)

if __name__ == '__main__':
    main()