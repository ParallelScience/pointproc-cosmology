# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree

def get_pair_counts(tree1, tree2, r_bins, is_auto=False):
    counts_cum = tree1.count_neighbors(tree2, r_bins)
    counts_bin = counts_cum[1:] - counts_cum[:-1]
    if is_auto:
        counts_bin = counts_bin / 2.0
    return counts_bin

def compute_2pcf_optimized(pos_c, pos_s, pos_s_null, pos_r, r_bins, boxsize=500.0):
    tree_c = cKDTree(pos_c, boxsize=boxsize)
    tree_s = cKDTree(pos_s, boxsize=boxsize)
    tree_s_null = cKDTree(pos_s_null, boxsize=boxsize)
    tree_r = cKDTree(pos_r, boxsize=boxsize)
    N_c = len(pos_c)
    N_s = len(pos_s)
    N_r = len(pos_r)
    norm_cs = float(N_c) * float(N_s)
    norm_cr = float(N_c) * float(N_r)
    norm_sr = float(N_s) * float(N_r)
    norm_ss = float(N_s) * float(N_s - 1) / 2.0
    norm_rr = float(N_r) * float(N_r - 1) / 2.0
    R_R = get_pair_counts(tree_r, tree_r, r_bins, is_auto=True)
    RR_norm = R_R / norm_rr
    RR_norm_safe = np.maximum(RR_norm, 1e-15)
    D_c_R = get_pair_counts(tree_c, tree_r, r_bins, is_auto=False)
    DR_c_norm = D_c_R / norm_cr
    D_c_D_s = get_pair_counts(tree_c, tree_s, r_bins, is_auto=False)
    D_s_R = get_pair_counts(tree_s, tree_r, r_bins, is_auto=False)
    D_s_D_s = get_pair_counts(tree_s, tree_s, r_bins, is_auto=True)
    DD_cs_norm = D_c_D_s / norm_cs
    DR_s_norm = D_s_R / norm_sr
    DD_ss_norm = D_s_D_s / norm_ss
    xi_cs = (DD_cs_norm - DR_c_norm - DR_s_norm + RR_norm) / RR_norm_safe
    xi_ss = (DD_ss_norm - 2 * DR_s_norm + RR_norm) / RR_norm_safe
    D_c_D_s_null = get_pair_counts(tree_c, tree_s_null, r_bins, is_auto=False)
    D_s_R_null = get_pair_counts(tree_s_null, tree_r, r_bins, is_auto=False)
    D_s_D_s_null = get_pair_counts(tree_s_null, tree_s_null, r_bins, is_auto=True)
    DD_cs_norm_null = D_c_D_s_null / norm_cs
    DR_s_norm_null = D_s_R_null / norm_sr
    DD_ss_norm_null = D_s_D_s_null / norm_ss
    xi_cs_null = (DD_cs_norm_null - DR_c_norm - DR_s_norm_null + RR_norm) / RR_norm_safe
    xi_ss_null = (DD_ss_norm_null - 2 * DR_s_norm_null + RR_norm) / RR_norm_safe
    invalid = RR_norm == 0
    xi_cs[invalid] = np.nan
    xi_ss[invalid] = np.nan
    xi_cs_null[invalid] = np.nan
    xi_ss_null[invalid] = np.nan
    return xi_cs, xi_ss, xi_cs_null, xi_ss_null

def main():
    data_dir = 'data/'
    r_bins = np.linspace(0, 5, 26)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    n_bins = len(r_bin_centers)
    all_xi_cs = np.zeros((10, n_bins))
    all_xi_ss = np.zeros((10, n_bins))
    all_xi_cs_null = np.zeros((10, n_bins))
    all_xi_ss_null = np.zeros((10, n_bins))
    N_r = 300000
    print('Computing 1-halo regime 2PCF (r < 5 Mpc/h) for data and null models...\n')
    for i in range(10):
        gal_path = os.path.join(data_dir, 'matched_catalog_' + str(i).zfill(2) + '.npz')
        data = np.load(gal_path)
        galaxies = data['galaxies']
        is_central = galaxies[:, 8] == 1
        pos_c = galaxies[is_central, :3]
        pos_s = galaxies[~is_central, :3]
        null_path = os.path.join(data_dir, 'null_catalog_' + str(i).zfill(2) + '.npy')
        null_galaxies = np.load(null_path)
        pos_s_null = null_galaxies[~is_central, :3]
        np.random.seed(100 + i)
        pos_r = np.random.uniform(0, 500.0, (N_r, 3))
        xi_cs, xi_ss, xi_cs_null, xi_ss_null = compute_2pcf_optimized(pos_c, pos_s, pos_s_null, pos_r, r_bins, boxsize=500.0)
        all_xi_cs[i] = xi_cs
        all_xi_ss[i] = xi_ss
        all_xi_cs_null[i] = xi_cs_null
        all_xi_ss_null[i] = xi_ss_null
        print('Realization ' + str(i).zfill(2) + ' completed.')
    if np.isnan(all_xi_cs).any() or np.isnan(all_xi_ss).any():
        print('\nWarning: NaN values encountered in 2PCF (likely due to RR=0). Filling with 0 for covariance computation.')
        all_xi_cs = np.nan_to_num(all_xi_cs)
        all_xi_ss = np.nan_to_num(all_xi_ss)
        all_xi_cs_null = np.nan_to_num(all_xi_cs_null)
        all_xi_ss_null = np.nan_to_num(all_xi_ss_null)
    cov_xi_cs = np.cov(all_xi_cs, rowvar=False)
    cov_xi_ss = np.cov(all_xi_ss, rowvar=False)
    cov_xi_cs_null = np.cov(all_xi_cs_null, rowvar=False)
    cov_xi_ss_null = np.cov(all_xi_ss_null, rowvar=False)
    out_path = os.path.join(data_dir, '2pcf_1halo.npz')
    np.savez(out_path, r_bins=r_bins, r_bin_centers=r_bin_centers, xi_cs=all_xi_cs, xi_ss=all_xi_ss, xi_cs_null=all_xi_cs_null, xi_ss_null=all_xi_ss_null, cov_xi_cs=cov_xi_cs, cov_xi_ss=cov_xi_ss, cov_xi_cs_null=cov_xi_cs_null, cov_xi_ss_null=cov_xi_ss_null)
    print('\nSaved 2PCF components and covariance matrices to ' + out_path)
    mean_xi_cs = np.mean(all_xi_cs, axis=0)
    mean_xi_ss = np.mean(all_xi_ss, axis=0)
    mean_xi_cs_null = np.mean(all_xi_cs_null, axis=0)
    mean_xi_ss_null = np.mean(all_xi_ss_null, axis=0)
    print('\nSummary of Mean 2PCF (first 10 bins):')
    print('r_centers (Mpc/h):               ' + str(np.round(r_bin_centers[:10], 3)))
    print('Data Central-Satellite xi_cs:    ' + str(np.round(mean_xi_cs[:10], 2)))
    print('Data Satellite-Satellite xi_ss:  ' + str(np.round(mean_xi_ss[:10], 2)))
    print('Null Central-Satellite xi_cs:    ' + str(np.round(mean_xi_cs_null[:10], 2)))
    print('Null Satellite-Satellite xi_ss:  ' + str(np.round(mean_xi_ss_null[:10], 2)))
    print('\nVariance (diagonal of covariance matrix) for first 10 bins:')
    print('Var Data xi_cs:                  ' + str(np.round(np.diag(cov_xi_cs)[:10], 2)))
    print('Var Data xi_ss:                  ' + str(np.round(np.diag(cov_xi_ss)[:10], 2)))

if __name__ == '__main__':
    main()