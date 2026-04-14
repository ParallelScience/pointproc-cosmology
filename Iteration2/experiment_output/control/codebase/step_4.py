# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import os

def main():
    data_dir = 'data/'
    n_realizations = 10
    L_box = 500.0
    r_bins_M = np.linspace(0, 5, 11)
    r_centers_M = 0.5 * (r_bins_M[:-1] + r_bins_M[1:])
    r_bins_2pcf = np.linspace(5, 10, 11)
    r_centers_2pcf = 0.5 * (r_bins_2pcf[:-1] + r_bins_2pcf[1:])
    all_M_r = []
    all_pearson_r = []
    all_xi_emp = []
    all_xi_dec = []
    print('Starting Mass-Conditioned Marked Correlation and Transition Scale Analysis...\n')
    for i in range(n_realizations):
        gal_path = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + str(i).zfill(2) + '.npy'
        halo_path = '/home/node/work/projects/pointproc_cosmology/data/halo_catalog_' + str(i).zfill(2) + '.npy'
        galaxies = np.load(gal_path)
        halos = np.load(halo_path)
        pos = galaxies[:, 0:3]
        L_gal = np.exp(galaxies[:, 6])
        tree_gal = cKDTree(pos, boxsize=L_box)
        pairs = tree_gal.query_pairs(r=5.0)
        pairs = np.array(list(pairs))
        mean_L_sq = np.mean(L_gal)**2
        M_r = np.zeros(len(r_bins_M) - 1)
        if len(pairs) > 0:
            idx1 = pairs[:, 0]
            idx2 = pairs[:, 1]
            dx = pos[idx1, 0] - pos[idx2, 0]
            dy = pos[idx1, 1] - pos[idx2, 1]
            dz = pos[idx1, 2] - pos[idx2, 2]
            dx = dx - L_box * np.round(dx / L_box)
            dy = dy - L_box * np.round(dy / L_box)
            dz = dz - L_box * np.round(dz / L_box)
            d = np.sqrt(dx**2 + dy**2 + dz**2)
            weights = L_gal[idx1] * L_gal[idx2]
            for k in range(len(r_bins_M) - 1):
                mask = (d > r_bins_M[k]) & (d <= r_bins_M[k+1])
                if np.sum(mask) > 0:
                    M_r[k] = np.mean(weights[mask]) / mean_L_sq
                else:
                    M_r[k] = np.nan
        else:
            M_r = np.full(len(r_bins_M) - 1, np.nan)
        all_M_r.append(M_r)
        is_central = galaxies[:, 8].astype(bool)
        satellites = galaxies[~is_central]
        centrals = galaxies[is_central]
        ln_M_halo = np.log(halos[:, 6])
        tree_halo = cKDTree(ln_M_halo[:, None])
        _, idx_cen_halo = tree_halo.query(centrals[:, 7:8], k=1)
        halo_to_cen = {h_idx: c_idx for c_idx, h_idx in enumerate(idx_cen_halo)}
        _, idx_sat_halo = tree_halo.query(satellites[:, 7:8], k=1)
        matched_halos = halos[idx_sat_halo]
        sat_cen_idx = np.array([halo_to_cen[h] for h in idx_sat_halo])
        matched_centrals = centrals[sat_cen_idx]
        ln_L_sat = satellites[:, 6]
        ln_L_cen = matched_centrals[:, 6]
        lum_offset = ln_L_sat - ln_L_cen
        if halos.shape[1] >= 8:
            R_vir = matched_halos[:, 7]
        else:
            Omega_m = 0.311
            rho_c = 2.77536627e11
            rho_m = Omega_m * rho_c
            Delta_m = 200.0
            R_vir = (matched_halos[:, 6] / (4.0/3.0 * np.pi * Delta_m * rho_m))**(1.0/3.0)
        dx_sat = satellites[:, 0] - matched_halos[:, 0]
        dy_sat = satellites[:, 1] - matched_halos[:, 1]
        dz_sat = satellites[:, 2] - matched_halos[:, 2]
        dx_sat = dx_sat - L_box * np.round(dx_sat / L_box)
        dy_sat = dy_sat - L_box * np.round(dy_sat / L_box)
        dz_sat = dz_sat - L_box * np.round(dz_sat / L_box)
        r_sat = np.sqrt(dx_sat**2 + dy_sat**2 + dz_sat**2)
        r_norm = r_sat / R_vir
        pearson_r = np.corrcoef(lum_offset, r_norm)[0, 1]
        all_pearson_r.append(pearson_r)
        log10_M = np.log10(matched_halos[:, 6])
        mass_bins = np.arange(12.8, 15.6, 0.2)
        new_dx = np.zeros_like(dx_sat)
        new_dy = np.zeros_like(dy_sat)
        new_dz = np.zeros_like(dz_sat)
        for j in range(len(mass_bins) - 1):
            m_low = mass_bins[j]
            m_high = mass_bins[j+1]
            mask = (log10_M >= m_low) & (log10_M < m_high)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                shuffled_idx = np.random.permutation(idx)
                new_dx[idx] = dx_sat[shuffled_idx]
                new_dy[idx] = dy_sat[shuffled_idx]
                new_dz[idx] = dz_sat[shuffled_idx]
        dec_sat_x = (matched_halos[:, 0] + new_dx) % L_box
        dec_sat_y = (matched_halos[:, 1] + new_dy) % L_box
        dec_sat_z = (matched_halos[:, 2] + new_dz) % L_box
        dec_pos = np.vstack([centrals[:, 0:3], np.column_stack([dec_sat_x, dec_sat_y, dec_sat_z])])
        tree_dec = cKDTree(dec_pos, boxsize=L_box)
        counts_DD_emp = tree_gal.count_neighbors(tree_gal, r_bins_2pcf)
        counts_DD_dec = tree_dec.count_neighbors(tree_dec, r_bins_2pcf)
        diff_DD_emp = np.diff(counts_DD_emp)
        diff_DD_dec = np.diff(counts_DD_dec)
        N_D = len(pos)
        DD_emp_norm = diff_DD_emp / (N_D * (N_D - 1))
        DD_dec_norm = diff_DD_dec / (N_D * (N_D - 1))
        V_shell = (4.0/3.0) * np.pi * (r_bins_2pcf[1:]**3 - r_bins_2pcf[:-1]**3)
        V_box = L_box**3
        RR_norm = V_shell / V_box
        xi_emp = DD_emp_norm / RR_norm - 1.0
        xi_dec = DD_dec_norm / RR_norm - 1.0
        all_xi_emp.append(xi_emp)
        all_xi_dec.append(xi_dec)
        print('Realization ' + str(i) + ':')
        print('  Pearson r (lum_offset vs r_norm): ' + str(np.round(pearson_r, 4)))
        print('  M(r) in [0.0, 0.5] Mpc/h: ' + str(np.round(M_r[0], 4)))
        print('  Empirical 2PCF in [5.0, 5.5] Mpc/h: ' + str(np.round(xi_emp[0], 4)))
        print('  Decoupled 2PCF in [5.0, 5.5] Mpc/h: ' + str(np.round(xi_dec[0], 4)))
        print('-' * 40)
    all_M_r = np.array(all_M_r)
    all_pearson_r = np.array(all_pearson_r)
    all_xi_emp = np.array(all_xi_emp)
    all_xi_dec = np.array(all_xi_dec)
    cov_xi_emp = np.cov(all_xi_emp, rowvar=False)
    cov_xi_dec = np.cov(all_xi_dec, rowvar=False)
    cov_xi_diff = np.cov(all_xi_emp - all_xi_dec, rowvar=False)
    print('\n=== Aggregated Results (10 Realizations) ===')
    print('Mean Pearson r: ' + str(np.round(np.mean(all_pearson_r), 4)) + ' +/- ' + str(np.round(np.std(all_pearson_r), 4)))
    print('\nMean Marked Correlation M(r):')
    for k in range(len(r_centers_M)):
        print('  r = ' + str(np.round(r_centers_M[k], 2)) + ' Mpc/h: ' + str(np.round(np.mean(all_M_r[:, k]), 4)) + ' +/- ' + str(np.round(np.std(all_M_r[:, k]), 4)))
    print('\nMean Empirical 2PCF:')
    for k in range(len(r_centers_2pcf)):
        print('  r = ' + str(np.round(r_centers_2pcf[k], 2)) + ' Mpc/h: ' + str(np.round(np.mean(all_xi_emp[:, k]), 4)) + ' +/- ' + str(np.round(np.std(all_xi_emp[:, k]), 4)))
    print('\nMean Decoupled 2PCF:')
    for k in range(len(r_centers_2pcf)):
        print('  r = ' + str(np.round(r_centers_2pcf[k], 2)) + ' Mpc/h: ' + str(np.round(np.mean(all_xi_dec[:, k]), 4)) + ' +/- ' + str(np.round(np.std(all_xi_dec[:, k]), 4)))
    print('\nCovariance Matrix Diagonal (Empirical - Decoupled 2PCF):')
    print(np.diag(cov_xi_diff))
    output_file = os.path.join(data_dir, 'marked_corr_and_transition.npz')
    np.savez(output_file, r_bins_M=r_bins_M, r_centers_M=r_centers_M, all_M_r=all_M_r, all_pearson_r=all_pearson_r, r_bins_2pcf=r_bins_2pcf, r_centers_2pcf=r_centers_2pcf, all_xi_emp=all_xi_emp, all_xi_dec=all_xi_dec, cov_xi_emp=cov_xi_emp, cov_xi_dec=cov_xi_dec, cov_xi_diff=cov_xi_diff)
    print('\nResults successfully saved to ' + output_file)

if __name__ == '__main__':
    main()