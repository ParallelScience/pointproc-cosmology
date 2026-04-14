# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    input_data_dir = '/home/node/work/projects/pointproc_cosmology/data/'
    output_data_dir = 'data/'
    L_box = 500.0
    rho_crit = 2.77536627e11
    r_bins = np.linspace(0, 5, 21)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    M_r_all = []
    pearson_all = []
    spearman_all = []
    all_m = []
    all_x = []
    print('--- Mass-Conditioned Marked Correlation ---')
    print('Processing 10 realizations...')
    for i in range(10):
        halo_file = os.path.join(input_data_dir, 'halo_catalog_' + str(i).zfill(2) + '.npy')
        gal_file = os.path.join(input_data_dir, 'galaxy_catalog_' + str(i).zfill(2) + '.npy')
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        centrals = gals[gals[:, 8] == 1]
        sats = gals[gals[:, 8] == 0]
        sort_idx_c = np.argsort(centrals[:, 7])
        sorted_centrals_ln_M = centrals[sort_idx_c, 7]
        idx_c = np.searchsorted(sorted_centrals_ln_M, sats[:, 7])
        idx_c = np.clip(idx_c, 1, len(sorted_centrals_ln_M) - 1)
        left_diff_c = np.abs(sats[:, 7] - sorted_centrals_ln_M[idx_c - 1])
        right_diff_c = np.abs(sats[:, 7] - sorted_centrals_ln_M[idx_c])
        closest_idx_c = np.where(left_diff_c < right_diff_c, idx_c - 1, idx_c)
        matched_central_idx = sort_idx_c[closest_idx_c]
        central_ln_L = centrals[matched_central_idx, 6]
        m = sats[:, 6] - central_ln_L
        halo_ln_M = np.log(halos[:, 6])
        sort_halo_idx = np.argsort(halo_ln_M)
        sorted_halo_ln_M = halo_ln_M[sort_halo_idx]
        idx_h = np.searchsorted(sorted_halo_ln_M, sats[:, 7])
        idx_h = np.clip(idx_h, 1, len(sorted_halo_ln_M) - 1)
        left_diff_h = np.abs(sats[:, 7] - sorted_halo_ln_M[idx_h - 1])
        right_diff_h = np.abs(sats[:, 7] - sorted_halo_ln_M[idx_h])
        closest_idx_h = np.where(left_diff_h < right_diff_h, idx_h - 1, idx_h)
        matched_halo_idx = sort_halo_idx[closest_idx_h]
        matched_halos = halos[matched_halo_idx]
        host_x = matched_halos[:, 0]
        host_y = matched_halos[:, 1]
        host_z = matched_halos[:, 2]
        M_vir = matched_halos[:, 6]
        R_vir = (3.0 * M_vir / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        dx = np.abs(sats[:, 0] - host_x)
        dx = np.minimum(dx, L_box - dx)
        dy = np.abs(sats[:, 1] - host_y)
        dy = np.minimum(dy, L_box - dy)
        dz = np.abs(sats[:, 2] - host_z)
        dz = np.minimum(dz, L_box - dz)
        r_sat = np.sqrt(dx**2 + dy**2 + dz**2)
        x_sat = r_sat / R_vir
        all_m.extend(m)
        all_x.extend(x_sat)
        mean_m_i = np.mean(m)
        tree = cKDTree(sats[:, :3], boxsize=L_box)
        pairs = tree.query_pairs(r=5.0, output_type='ndarray')
        sum_m_prod_i = np.zeros(len(r_bins) - 1)
        count_pairs_i = np.zeros(len(r_bins) - 1)
        if len(pairs) > 0:
            idx1 = pairs[:, 0]
            idx2 = pairs[:, 1]
            pos1 = sats[idx1, :3]
            pos2 = sats[idx2, :3]
            dx_p = np.abs(pos1[:, 0] - pos2[:, 0])
            dx_p = np.minimum(dx_p, L_box - dx_p)
            dy_p = np.abs(pos1[:, 1] - pos2[:, 1])
            dy_p = np.minimum(dy_p, L_box - dy_p)
            dz_p = np.abs(pos1[:, 2] - pos2[:, 2])
            dz_p = np.minimum(dz_p, L_box - dz_p)
            d_p = np.sqrt(dx_p**2 + dy_p**2 + dz_p**2)
            m_prod = m[idx1] * m[idx2]
            counts, _ = np.histogram(d_p, bins=r_bins)
            sums, _ = np.histogram(d_p, bins=r_bins, weights=m_prod)
            sum_m_prod_i += sums
            count_pairs_i += counts
        valid_bins = count_pairs_i > 0
        M_r_i = np.full_like(r_centers, np.nan)
        M_r_i[valid_bins] = (sum_m_prod_i[valid_bins] / count_pairs_i[valid_bins]) / (mean_m_i**2)
        M_r_all.append(M_r_i)
        p_corr, p_p = pearsonr(m, x_sat)
        s_corr, s_p = spearmanr(m, x_sat)
        pearson_all.append(p_corr)
        spearman_all.append(s_corr)
        print('  Realization ' + str(i).zfill(2) + ' done. Pearson: ' + str(np.round(p_corr, 4)) + ', Spearman: ' + str(np.round(s_corr, 4)))
    M_r_all = np.array(M_r_all)
    pearson_all = np.array(pearson_all)
    spearman_all = np.array(spearman_all)
    all_m = np.array(all_m)
    all_x = np.array(all_x)
    global_pearson, global_pearson_p = pearsonr(all_m, all_x)
    global_spearman, global_spearman_p = spearmanr(all_m, all_x)
    mean_pearson = np.mean(pearson_all)
    std_pearson = np.std(pearson_all, ddof=1)
    mean_spearman = np.mean(spearman_all)
    std_spearman = np.std(spearman_all, ddof=1)
    mean_M_r = np.nanmean(M_r_all, axis=0)
    std_M_r = np.nanstd(M_r_all, axis=0, ddof=1)
    print('\n--- Summary across 10 realizations ---')
    print('Global Pearson correlation:  ' + str(np.round(global_pearson, 4)) + ' (p-value: ' + str(global_pearson_p) + ')')
    print('Global Spearman correlation: ' + str(np.round(global_spearman, 4)) + ' (p-value: ' + str(global_spearman_p) + ')')
    print('Mean per-realization Pearson:  ' + str(np.round(mean_pearson, 4)) + ' +/- ' + str(np.round(std_pearson, 4)))
    print('Mean per-realization Spearman: ' + str(np.round(mean_spearman, 4)) + ' +/- ' + str(np.round(std_spearman, 4)))
    print('\nMarked Correlation Function M(r) for r < 5 Mpc/h:')
    print('r_center (Mpc/h) | Mean M(r) | Std M(r)')
    print('-' * 45)
    for i in range(len(r_centers)):
        print(str(np.round(r_centers[i], 3)).rjust(16) + ' | ' + str(np.round(mean_M_r[i], 4)).rjust(9) + ' | ' + str(np.round(std_M_r[i], 4)).rjust(8))
    output_file = os.path.join(output_data_dir, 'marked_correlation_results.npz')
    np.savez(output_file, r_bins=r_bins, r_centers=r_centers, M_r_all=M_r_all, pearson_all=pearson_all, spearman_all=spearman_all, global_pearson=global_pearson, global_spearman=global_spearman)
    print('\nResults saved to ' + output_file)