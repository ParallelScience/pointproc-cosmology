# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

def compute_G_F(coords, boxsize, r_bins, n_random=1000000):
    tree = cKDTree(coords, boxsize=boxsize)
    dists_G, _ = tree.query(coords, k=2, workers=-1)
    if dists_G.ndim > 1:
        dists_G = dists_G[:, 1]
    G_counts = np.sum(dists_G[:, None] <= r_bins[None, :], axis=0)
    G_r = G_counts / float(len(dists_G))
    random_points = np.random.uniform(0, boxsize, (n_random, 3))
    dists_F, _ = tree.query(random_points, k=1, workers=-1)
    if dists_F.ndim > 1:
        dists_F = dists_F[:, 0]
    F_counts = np.sum(dists_F[:, None] <= r_bins[None, :], axis=0)
    F_r = F_counts / float(len(dists_F))
    return G_r, F_r

def compute_j_function():
    n_realizations = 10
    boxsize = 500.0
    r_bins = np.linspace(0.2, 15.0, 75)
    n_random = 1000000
    J_orig_all = np.zeros((n_realizations, len(r_bins)))
    J_shuf_all = np.zeros((n_realizations, len(r_bins)))
    np.random.seed(42)
    for i in range(n_realizations):
        filepath = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + str(i).zfill(2) + '.npy'
        galaxy_data = np.load(filepath)
        coords = galaxy_data[:, :3]
        G_orig, F_orig = compute_G_F(coords, boxsize, r_bins, n_random)
        denom_orig = 1.0 - F_orig
        denom_orig[denom_orig <= 0] = np.nan
        J_orig_all[i] = (1.0 - G_orig) / denom_orig
        shuffled_coords = coords.copy()
        unique_masses, inverse_indices = np.unique(galaxy_data[:, 7], return_inverse=True)
        for j in range(len(unique_masses)):
            idx = np.where(inverse_indices == j)[0]
            group = galaxy_data[idx]
            cen_mask = group[:, 8] == 1
            sat_mask = group[:, 8] == 0
            if np.sum(sat_mask) > 0:
                if np.sum(cen_mask) > 0:
                    center = group[cen_mask, :3][0]
                else:
                    center = np.mean(group[:, :3], axis=0)
                sat_coords = group[sat_mask, :3]
                rel_coords = sat_coords - center
                rel_coords = rel_coords - np.round(rel_coords / boxsize) * boxsize
                dists = np.linalg.norm(rel_coords, axis=1)
                u = np.random.randn(len(dists), 3)
                norms = np.linalg.norm(u, axis=1)
                norms[norms == 0] = 1.0
                u /= norms[:, None]
                new_rel_coords = u * dists[:, None]
                new_sat_coords = center + new_rel_coords
                new_sat_coords = np.mod(new_sat_coords, boxsize)
                sat_idx = idx[sat_mask]
                shuffled_coords[sat_idx] = new_sat_coords
        G_shuf, F_shuf = compute_G_F(shuffled_coords, boxsize, r_bins, n_random)
        denom_shuf = 1.0 - F_shuf
        denom_shuf[denom_shuf <= 0] = np.nan
        J_shuf_all[i] = (1.0 - G_shuf) / denom_shuf
    mean_J_orig = np.nanmean(J_orig_all, axis=0)
    std_J_orig = np.nanstd(J_orig_all, axis=0)
    mean_J_shuf = np.nanmean(J_shuf_all, axis=0)
    std_J_shuf = np.nanstd(J_shuf_all, axis=0)
    def find_first_deviation(mean_J, std_J, n, r_bins):
        sem_J = std_J / np.sqrt(n)
        dev = np.abs(1.0 - mean_J) > 3.0 * sem_J
        dev = dev & (np.abs(1.0 - mean_J) > 0.01)
        idx = np.where(dev)[0]
        if len(idx) > 0:
            return r_bins[idx[0]]
        return np.nan
    dev_r_orig = find_first_deviation(mean_J_orig, std_J_orig, n_realizations, r_bins)
    dev_r_shuf = find_first_deviation(mean_J_shuf, std_J_shuf, n_realizations, r_bins)
    dev_orig_str = str(np.round(dev_r_orig, 2)) if not np.isnan(dev_r_orig) else 'None'
    dev_shuf_str = str(np.round(dev_r_shuf, 2)) if not np.isnan(dev_r_shuf) else 'None'
    print('J-function Analysis Results:')
    print('Radius of first significant deviation from CSR (J=1):')
    print('  Original Catalog: ' + dev_orig_str + ' Mpc/h')
    print('  Shuffled Catalog: ' + dev_shuf_str + ' Mpc/h')
    print('\nJ-function values at selected radii (Original vs Shuffled):')
    print('r [Mpc/h] | J_orig (Mean +- Std) | J_shuf (Mean +- Std)')
    radii_to_print = [1.0, 5.0, 10.0, 15.0]
    for r_val in radii_to_print:
        idx = np.argmin(np.abs(r_bins - r_val))
        r_actual = r_bins[idx]
        j_o = mean_J_orig[idx]
        j_o_std = std_J_orig[idx]
        j_s = mean_J_shuf[idx]
        j_s_std = std_J_shuf[idx]
        print(str(np.round(r_actual, 1)).ljust(9) + ' | ' + str(np.round(j_o, 4)).ljust(6) + ' +- ' + str(np.round(j_o_std, 4)).ljust(6) + ' | ' + str(np.round(j_s, 4)).ljust(6) + ' +- ' + str(np.round(j_s_std, 4)))
    save_path = os.path.join('data', 'j_function_results.npz')
    np.savez(save_path, r_bins=r_bins, J_orig_all=J_orig_all, mean_J_orig=mean_J_orig, std_J_orig=std_J_orig, J_shuf_all=J_shuf_all, mean_J_shuf=mean_J_shuf, std_J_shuf=std_J_shuf)
    print('\nResults saved to ' + save_path)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(r_bins, mean_J_orig, 'b-', label='Original Catalog (Mean)')
    plt.fill_between(r_bins, mean_J_orig - std_J_orig, mean_J_orig + std_J_orig, color='b', alpha=0.3, label='Original 1 Std Dev')
    plt.plot(r_bins, mean_J_shuf, 'r--', label='Shuffled Catalog (Mean)')
    plt.fill_between(r_bins, mean_J_shuf - std_J_shuf, mean_J_shuf + std_J_shuf, color='r', alpha=0.3, label='Shuffled 1 Std Dev')
    plt.axhline(1.0, color='k', linestyle=':', label='CSR (Poisson)')
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('J(r)')
    plt.title('J-function: Original vs Shuffled Satellite Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/j_function_plot_1_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    print('Plot saved to ' + plot_path)
    plt.close()

if __name__ == '__main__':
    compute_j_function()