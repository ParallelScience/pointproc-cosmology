# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import time
import warnings
import os

def test_spatial_mark_independence():
    n_realizations = 10
    boxsize = 500.0
    spearman_results = []
    bins = np.linspace(0, 5, 26)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    sat_L_profiles = np.zeros((n_realizations, len(bin_centers)))
    cen_L_means = np.zeros(n_realizations)
    for i in range(n_realizations):
        gal_filepath = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + str(i).zfill(2) + '.npy'
        halo_filepath = '/home/node/work/projects/pointproc_cosmology/data/halo_catalog_' + str(i).zfill(2) + '.npy'
        galaxy_data = np.load(gal_filepath)
        halo_data = np.load(halo_filepath)
        halo_M_vir = halo_data[:, 6]
        if halo_data.shape[1] > 7:
            halo_R_vir = halo_data[:, 7]
        else:
            rho_m = 0.311 * 2.775e11
            halo_R_vir = (halo_M_vir / ( (4.0/3.0) * np.pi * 200.0 * rho_m ))**(1.0/3.0)
        halo_coords = halo_data[:, :3]
        tree = cKDTree(halo_coords, boxsize=boxsize)
        unique_masses, inverse_indices = np.unique(galaxy_data[:, 7], return_inverse=True)
        sort_idx = np.argsort(inverse_indices)
        sorted_gals = galaxy_data[sort_idx]
        sorted_inv = inverse_indices[sort_idx]
        split_idx = np.where(np.diff(sorted_inv) != 0)[0] + 1
        groups = np.split(sorted_gals, split_idx)
        r_dist_sorted = np.zeros(len(sorted_gals))
        R_vir_gal_sorted = np.zeros(len(sorted_gals))
        current_idx = 0
        for group in groups:
            n_group = len(group)
            central_mask = group[:, 8] == 1
            if np.sum(central_mask) > 0:
                center = group[central_mask, :3][0]
            else:
                center = np.mean(group[:, :3], axis=0)
            _, halo_idx = tree.query(center)
            dx = np.abs(group[:, 0] - center[0])
            dy = np.abs(group[:, 1] - center[1])
            dz = np.abs(group[:, 2] - center[2])
            dx = np.minimum(dx, boxsize - dx)
            dy = np.minimum(dy, boxsize - dy)
            dz = np.minimum(dz, boxsize - dz)
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            r_dist_sorted[current_idx : current_idx + n_group] = dists
            R_vir_gal_sorted[current_idx : current_idx + n_group] = halo_R_vir[halo_idx]
            current_idx += n_group
        x_norm = r_dist_sorted / R_vir_gal_sorted
        ln_L = sorted_gals[:, 6]
        is_central = sorted_gals[:, 8] == 1
        is_satellite = sorted_gals[:, 8] == 0
        cen_x = x_norm[is_central]
        cen_L = ln_L[is_central]
        sat_x = x_norm[is_satellite]
        sat_L = ln_L[is_satellite]
        for k in range(len(bin_centers)):
            mask = (sat_x >= bins[k]) & (sat_x < bins[k+1])
            if np.sum(mask) > 0:
                sat_L_profiles[i, k] = np.mean(sat_L[mask])
            else:
                sat_L_profiles[i, k] = np.nan
        cen_L_means[i] = np.mean(cen_L)
        if len(cen_x) > 1 and np.std(cen_x) > 0:
            corr_c, p_c = spearmanr(cen_x, cen_L)
        else:
            corr_c, p_c = np.nan, np.nan
        if len(sat_x) > 1 and np.std(sat_x) > 0:
            corr_s, p_s = spearmanr(sat_x, sat_L)
        else:
            corr_s, p_s = np.nan, np.nan
        spearman_results.append({'realization': i, 'corr_c': corr_c, 'p_c': p_c, 'corr_s': corr_s, 'p_s': p_s})
    print('Spearman Rank Correlation between Luminosity Mark (ln_L) and Halo-centric Distance (r/R_vir):')
    print('Realization | Centrals r_s | Centrals p-val | Satellites r_s | Satellites p-val')
    valid_corr_c = []
    valid_corr_s = []
    for res in spearman_results:
        c_r = str(np.round(res['corr_c'], 4)) if not np.isnan(res['corr_c']) else 'NaN'
        c_p = str('%e' % res['p_c']) if not np.isnan(res['p_c']) else 'NaN'
        s_r = str(np.round(res['corr_s'], 4)) if not np.isnan(res['corr_s']) else 'NaN'
        s_p = str('%e' % res['p_s']) if not np.isnan(res['p_s']) else 'NaN'
        print(str(res['realization']).ljust(11) + ' | ' + c_r.rjust(12) + ' | ' + c_p.rjust(14) + ' | ' + s_r.rjust(14) + ' | ' + s_p.rjust(16))
        if not np.isnan(res['corr_c']):
            valid_corr_c.append(res['corr_c'])
        if not np.isnan(res['corr_s']):
            valid_corr_s.append(res['corr_s'])
    mean_corr_c = np.mean(valid_corr_c) if len(valid_corr_c) > 0 else np.nan
    mean_corr_s = np.mean(valid_corr_s) if len(valid_corr_s) > 0 else np.nan
    print('-' * 75)
    c_r_mean = str(np.round(mean_corr_c, 4)) if not np.isnan(mean_corr_c) else 'NaN'
    s_r_mean = str(np.round(mean_corr_s, 4)) if not np.isnan(mean_corr_s) else 'NaN'
    print('Mean        | ' + c_r_mean.rjust(12) + ' | ' + '-'.rjust(14) + ' | ' + s_r_mean.rjust(14) + ' | ' + '-'.rjust(16))
    save_path = os.path.join('data', 'mark_independence_results.npz')
    np.savez(save_path, bins=bins, bin_centers=bin_centers, sat_L_profiles=sat_L_profiles, cen_L_means=cen_L_means)
    print('Results saved to ' + save_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_sat_L = np.nanmean(sat_L_profiles, axis=0)
    n_valid = np.sum(~np.isnan(sat_L_profiles), axis=0)
    sem_sat_L = np.zeros_like(mean_sat_L)
    for k in range(len(bin_centers)):
        if n_valid[k] > 0:
            sem_sat_L[k] = np.nanstd(sat_L_profiles[:, k]) / np.sqrt(n_valid[k])
        else:
            sem_sat_L[k] = np.nan
    mean_cen_L = np.nanmean(cen_L_means)
    sem_cen_L = np.nanstd(cen_L_means) / np.sqrt(n_realizations)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    valid_bins = ~np.isnan(mean_sat_L)
    plt.errorbar(bin_centers[valid_bins], mean_sat_L[valid_bins], yerr=sem_sat_L[valid_bins], fmt='o-', color='b', label='Satellites', capsize=3)
    plt.errorbar([0.0], [mean_cen_L], yerr=[sem_cen_L], fmt='s', color='r', markersize=8, label='Centrals', capsize=3)
    plt.xlabel('Normalized Halo-centric Distance (r / R_vir)')
    plt.ylabel('Mean Luminosity Mark (ln L)')
    plt.title('Spatial-Mark Independence: Luminosity Profile')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/mark_independence_profile_1_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    print('Plot saved to ' + plot_path)
    plt.close()

if __name__ == '__main__':
    test_spatial_mark_independence()