# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from datetime import datetime
import warnings
import os
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)
plt.rcParams['text.usetex'] = False

def compute_marked_correlation():
    data_dir = 'data/'
    bins_1h = np.linspace(0.1, 5.0, 15)
    bins_2h = np.linspace(10.0, 30.0, 15)
    centers_1h = (bins_1h[:-1] + bins_1h[1:]) / 2.0
    centers_2h = (bins_2h[:-1] + bins_2h[1:]) / 2.0
    n_bins_1h = len(centers_1h)
    n_bins_2h = len(centers_2h)
    M_ratio_1h_all = []
    M_ratio_2h_all = []
    for i in range(10):
        realization_str = str(i).zfill(2)
        gal_path = '/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_' + realization_str + '.npy'
        if not os.path.exists(gal_path):
            continue
        galaxies = np.load(gal_path)
        pos = galaxies[:, 0:3]
        L = np.exp(galaxies[:, 6])
        tree = cKDTree(pos, boxsize=500.0)
        pairs = tree.query_pairs(30.0, output_type='ndarray')
        if len(pairs) == 0:
            continue
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        dx = pos[idx_i, 0] - pos[idx_j, 0]
        dy = pos[idx_i, 1] - pos[idx_j, 1]
        dz = pos[idx_i, 2] - pos[idx_j, 2]
        dx = dx - 500.0 * np.round(dx / 500.0)
        dy = dy - 500.0 * np.round(dy / 500.0)
        dz = dz - 500.0 * np.round(dz / 500.0)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        mark_prod_obs = L[idx_i] * L[idx_j]
        n_shuffles = 10
        mark_prod_shuff_sum = np.zeros(len(pairs))
        for _ in range(n_shuffles):
            L_shuff = np.random.permutation(L)
            mark_prod_shuff_sum += L_shuff[idx_i] * L_shuff[idx_j]
        mark_prod_shuff_avg = mark_prod_shuff_sum / n_shuffles
        ratio_1h = np.zeros(n_bins_1h)
        for b in range(n_bins_1h):
            mask = (dist >= bins_1h[b]) & (dist < bins_1h[b+1])
            if np.sum(mask) > 0:
                m_obs = np.mean(mark_prod_obs[mask])
                m_shuff = np.mean(mark_prod_shuff_avg[mask])
                ratio_1h[b] = m_obs / m_shuff
            else:
                ratio_1h[b] = np.nan
        ratio_2h = np.zeros(n_bins_2h)
        for b in range(n_bins_2h):
            mask = (dist >= bins_2h[b]) & (dist < bins_2h[b+1])
            if np.sum(mask) > 0:
                m_obs = np.mean(mark_prod_obs[mask])
                m_shuff = np.mean(mark_prod_shuff_avg[mask])
                ratio_2h[b] = m_obs / m_shuff
            else:
                ratio_2h[b] = np.nan
        M_ratio_1h_all.append(ratio_1h)
        M_ratio_2h_all.append(ratio_2h)
    M_ratio_1h_all = np.array(M_ratio_1h_all)
    M_ratio_2h_all = np.array(M_ratio_2h_all)
    mean_1h = np.nanmean(M_ratio_1h_all, axis=0)
    std_1h = np.nanstd(M_ratio_1h_all, axis=0)
    mean_2h = np.nanmean(M_ratio_2h_all, axis=0)
    std_2h = np.nanstd(M_ratio_2h_all, axis=0)
    df_1h = pd.DataFrame({'r_center': centers_1h, 'M_ratio_mean': mean_1h, 'M_ratio_std': std_1h})
    df_2h = pd.DataFrame({'r_center': centers_2h, 'M_ratio_mean': mean_2h, 'M_ratio_std': std_2h})
    df_1h.to_csv(os.path.join(data_dir, 'marked_correlation_1h.csv'), index=False)
    df_2h.to_csv(os.path.join(data_dir, 'marked_correlation_2h.csv'), index=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.errorbar(centers_1h, mean_1h, yerr=std_1h, fmt='-o', color='blue', capsize=4, markersize=6)
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Radial distance r [Mpc/h]')
    ax1.set_ylabel('M(r) / M_shuffled(r)')
    ax1.set_title('Marked Correlation: 1-Halo Regime (r < 5 Mpc/h)')
    ax1.grid(True, alpha=0.5, linestyle=':')
    ax2.errorbar(centers_2h, mean_2h, yerr=std_2h, fmt='-o', color='red', capsize=4, markersize=6)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Radial distance r [Mpc/h]')
    ax2.set_ylabel('M(r) / M_shuffled(r)')
    ax2.set_title('Marked Correlation: 2-Halo Regime (r > 10 Mpc/h)')
    ax2.grid(True, alpha=0.5, linestyle=':')
    plt.tight_layout()
    plot_path = os.path.join(data_dir, 'marked_correlation_' + str(int(time.time())) + '.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Marked correlation results saved to ' + data_dir)
    print('Plot saved to ' + plot_path)

if __name__ == '__main__':
    compute_marked_correlation()