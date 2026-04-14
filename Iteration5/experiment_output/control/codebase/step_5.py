# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import cKDTree

matplotlib.rcParams['text.usetex'] = False
warnings.filterwarnings('ignore', category=RuntimeWarning)

def compute_marked_correlation(pos, m, r_bins, L_box):
    tree = cKDTree(pos, boxsize=L_box)
    r_max = r_bins[-1]
    pairs = np.array(list(tree.query_pairs(r_max)))
    M_r = np.full(len(r_bins) - 1, np.nan)
    if len(pairs) == 0:
        return M_r
    i = pairs[:, 0]
    j = pairs[:, 1]
    dx = pos[i] - pos[j]
    dx = dx - L_box * np.round(dx / L_box)
    d = np.linalg.norm(dx, axis=1)
    bin_indices = np.digitize(d, r_bins) - 1
    m_ij = m[i] * m[j]
    m_mean_sq = np.mean(m)**2
    for k in range(len(r_bins) - 1):
        mask = bin_indices == k
        if np.sum(mask) > 0:
            M_r[k] = np.mean(m_ij[mask]) / m_mean_sq
    return M_r

def main():
    data_dir = '/home/node/work/projects/pointproc_cosmology/data'
    output_dir = 'data'
    L_box = 500.0
    print('Starting Marked Correlation Analysis...')
    r_bins = np.logspace(-1, np.log10(5.0), 15)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    M_r_orig_all = []
    M_r_shuf_all = []
    for i in range(10):
        gal_file = os.path.join(data_dir, 'galaxy_catalog_' + str(i).zfill(2) + '.npy')
        shuf_file = os.path.join(output_dir, 'shuffled_galaxy_catalog_' + str(i).zfill(2) + '.npy')
        if not os.path.exists(gal_file) or not os.path.exists(shuf_file):
            print('Warning: Files for realization ' + str(i).zfill(2) + ' not found. Skipping.')
            continue
        gal_data = np.load(gal_file)
        shuf_data = np.load(shuf_file)
        pos_orig = gal_data[:, 0:3]
        pos_shuf = shuf_data[:, 0:3]
        ln_L = gal_data[:, 6]
        L = np.exp(ln_L)
        ln_M_h = gal_data[:, 7]
        mass_bins = np.linspace(np.min(ln_M_h) - 0.01, np.max(ln_M_h) + 0.01, 11)
        bin_idx = np.digitize(ln_M_h, mass_bins) - 1
        m = np.zeros_like(L)
        for k in range(10):
            mask = bin_idx == k
            if np.sum(mask) > 0:
                m[mask] = L[mask] / np.mean(L[mask])
        M_r_orig = compute_marked_correlation(pos_orig, m, r_bins, L_box)
        M_r_shuf = compute_marked_correlation(pos_shuf, m, r_bins, L_box)
        M_r_orig_all.append(M_r_orig)
        M_r_shuf_all.append(M_r_shuf)
        print('Processed realization ' + str(i).zfill(2))
    if len(M_r_orig_all) == 0:
        print('Error: No data processed.')
        sys.exit(1)
    M_r_orig_mean = np.nanmean(M_r_orig_all, axis=0)
    M_r_orig_std = np.nanstd(M_r_orig_all, axis=0)
    M_r_shuf_mean = np.nanmean(M_r_shuf_all, axis=0)
    M_r_shuf_std = np.nanstd(M_r_shuf_all, axis=0)
    print('\nMarked Correlation Function M(r) Results (1-Halo Regime):')
    print('  r_center [Mpc/h] | M(r) Original | M(r) Shuffled | Difference')
    print('-' * 65)
    for i in range(len(r_centers)):
        diff = M_r_orig_mean[i] - M_r_shuf_mean[i]
        print('  ' + '{:16.4f}'.format(r_centers[i]) + ' | ' + '{:13.4f}'.format(M_r_orig_mean[i]) + ' | ' + '{:13.4f}'.format(M_r_shuf_mean[i]) + ' | ' + '{:10.4f}'.format(diff))
    plt.figure(figsize=(8, 6))
    valid = ~np.isnan(M_r_orig_mean) & ~np.isnan(M_r_shuf_mean)
    if np.any(valid):
        plt.plot(r_centers[valid], M_r_orig_mean[valid], label='Original Catalog', color='blue', marker='o', lw=2)
        plt.fill_between(r_centers[valid], (M_r_orig_mean - M_r_orig_std)[valid], (M_r_orig_mean + M_r_orig_std)[valid], color='blue', alpha=0.2)
        plt.plot(r_centers[valid], M_r_shuf_mean[valid], label='Shuffled Catalog', color='red', marker='s', lw=2, linestyle='--')
        plt.fill_between(r_centers[valid], (M_r_shuf_mean - M_r_shuf_std)[valid], (M_r_shuf_mean + M_r_shuf_std)[valid], color='red', alpha=0.2)
    plt.axhline(1.0, color='black', linestyle=':', label='Unmarked (M=1)')
    plt.xscale('log')
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('Marked Correlation M(r)')
    plt.title('Marked Correlation Function in 1-Halo Regime')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, 'marked_correlation_1_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nPlot saved to ' + plot_filename)
    out_npz = os.path.join(output_dir, 'marked_correlation_data.npz')
    np.savez(out_npz, r_centers=r_centers, r_bins=r_bins, M_r_orig_mean=M_r_orig_mean, M_r_orig_std=M_r_orig_std, M_r_shuf_mean=M_r_shuf_mean, M_r_shuf_std=M_r_shuf_std)
    print('Saved marked correlation data to ' + out_npz)

if __name__ == '__main__':
    main()