# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import os

def compute_k_l_functions():
    boxsize = 500.0
    volume = boxsize**3
    n_realizations = 10
    r_bins = np.arange(1, 101, 1.0)
    K_all = np.zeros((n_realizations, len(r_bins)))
    L_all = np.zeros((n_realizations, len(r_bins)))
    for i in range(n_realizations):
        filepath = "/home/node/work/projects/pointproc_cosmology/data/halo_catalog_" + str(i).zfill(2) + ".npy"
        halo_data = np.load(filepath)
        coords = halo_data[:, :3]
        n_points = len(coords)
        tree = cKDTree(coords, boxsize=boxsize)
        counts = tree.count_neighbors(tree, r_bins)
        pairs_ij = counts - n_points
        K_r = (volume / (n_points * (n_points - 1))) * pairs_ij
        L_r = ((3.0 / (4.0 * np.pi)) * K_r)**(1.0 / 3.0)
        K_all[i] = K_r
        L_all[i] = L_r
    L_dev = L_all - r_bins
    L_dev_mean = np.mean(L_dev, axis=0)
    L_dev_std = np.std(L_dev, axis=0)
    radii_to_print = [10, 20, 50]
    print("L-function deviation from CSR (L(r) - r):")
    for r_val in radii_to_print:
        idx = np.where(np.isclose(r_bins, r_val))[0][0]
        print("  r = " + str(r_val) + " Mpc/h: Mean = " + str(L_dev_mean[idx]) + " Mpc/h, Std = " + str(L_dev_std[idx]) + " Mpc/h")
    save_path = "data/halo_K_L_functions.npz"
    np.savez(save_path, r_bins=r_bins, K_all=K_all, L_all=L_all, L_dev_mean=L_dev_mean, L_dev_std=L_dev_std)
    print("Results saved to " + save_path)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(r_bins, L_dev_mean, 'b-', label='Mean L(r) - r')
    plt.fill_between(r_bins, L_dev_mean - L_dev_std, L_dev_mean + L_dev_std, color='b', alpha=0.3, label='1 Std Dev')
    plt.axhline(0, color='k', linestyle='--', label='CSR (Poisson)')
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('L(r) - r [Mpc/h]')
    plt.title('Halo Parent Process: L-function Deviation from CSR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = "data/halo_L_function_1_" + str(timestamp) + ".png"
    plt.savefig(plot_path, dpi=300)
    print("Plot saved to " + plot_path)
    plt.close()

if __name__ == '__main__':
    compute_k_l_functions()