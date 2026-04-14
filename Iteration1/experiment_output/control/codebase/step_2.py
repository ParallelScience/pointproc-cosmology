# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import os

def compute_vpf():
    boxsize = 500.0
    volume = boxsize**3
    n_realizations = 10
    r_bins = np.linspace(1, 30, 30)
    n_spheres = 5000000
    P0_all = np.zeros((n_realizations, len(r_bins)))
    densities = np.zeros(n_realizations)
    np.random.seed(42)
    for i in range(n_realizations):
        filepath = "/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_" + str(i).zfill(2) + ".npy"
        galaxy_data = np.load(filepath)
        coords = galaxy_data[:, :3]
        n_gal = len(coords)
        densities[i] = n_gal / volume
        tree = cKDTree(coords, boxsize=boxsize)
        random_points = np.random.uniform(0, boxsize, (n_spheres, 3))
        dists, _ = tree.query(random_points, k=1, workers=-1)
        for j in range(len(r_bins)):
            P0_all[i, j] = np.sum(dists > r_bins[j]) / float(n_spheres)
    P0_mean = np.mean(P0_all, axis=0)
    P0_var = np.var(P0_all, axis=0)
    P0_std = np.sqrt(P0_var)
    mean_density = np.mean(densities)
    V_r = (4.0 / 3.0) * np.pi * r_bins**3
    P0_poisson = np.exp(-mean_density * V_r)
    normalized_residuals = (P0_mean - P0_poisson) / P0_poisson
    print("Empirical VPF vs Poisson VPF and Normalized Residuals:")
    print("r [Mpc/h] | Empirical P0 | Poisson P0 | Normalized Residual (Emp - Poiss)/Poiss")
    for j in range(len(r_bins)):
        r_str = str(np.round(r_bins[j], 1))
        emp_str = str("%e" % P0_mean[j])
        poiss_str = str("%e" % P0_poisson[j])
        res_str = str("%e" % normalized_residuals[j])
        print(r_str + " | " + emp_str + " | " + poiss_str + " | " + res_str)
    save_path = "data/vpf_results.npz"
    np.savez(save_path, r_bins=r_bins, P0_all=P0_all, P0_mean=P0_mean, P0_var=P0_var, P0_poisson=P0_poisson, normalized_residuals=normalized_residuals)
    print("Results saved to " + save_path)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(r_bins, P0_mean, 'b-', label='Empirical VPF (Mean)')
    plt.fill_between(r_bins, P0_mean - P0_std, P0_mean + P0_std, color='b', alpha=0.3, label='1 Std Dev')
    plt.plot(r_bins, P0_poisson, 'r--', label='Poisson VPF')
    plt.yscale('log')
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('P_0(r)')
    plt.title('Void Probability Function (VPF)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = "data/vpf_plot_1_" + str(timestamp) + ".png"
    plt.savefig(plot_path, dpi=300)
    print("Plot saved to " + plot_path)
    plt.close()

if __name__ == '__main__':
    compute_vpf()