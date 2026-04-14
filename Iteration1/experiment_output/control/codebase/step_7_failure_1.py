# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.spatial.distance import pdist, squareform, cdist
import time

def fit_strauss_inhomogeneous(sat_coords_rel, R_grid, gamma_grid, M=1000):
    n = len(sat_coords_rel)
    radii = np.linalg.norm(sat_coords_rel, axis=1)
    idx = np.random.randint(0, n, size=M)
    r_dummy = radii[idx]
    u_dir = np.random.randn(M, 3)
    u_dir /= np.linalg.norm(u_dir, axis=1)[:, None]
    u = u_dir * r_dummy[:, None]
    d_ss = squareform(pdist(sat_coords_rel))
    np.fill_diagonal(d_ss, np.inf)
    d_us = cdist(u, sat_coords_rel)
    best_L = -np.inf
    best_R = np.nan
    best_gamma = np.nan
    t_x_all = np.array([np.sum(d_ss < R, axis=1) for R in R_grid])
    t_u_all = np.array([np.sum(d_us < R, axis=1) for R in R_grid])
    sum_t_x_all = np.sum(t_x_all, axis=1)
    for i, R in enumerate(R_grid):
        t_u = t_u_all[i]
        sum_t_x = sum_t_x_all[i]
        max_t = np.max(t_u)
        counts = np.bincount(t_u, minlength=max_t+1)
        probs = counts / float(M)
        for gamma in gamma_grid:
            if gamma == 0:
                if sum_t_x > 0:
                    L = -np.inf
                else:
                    E_gamma_t_u = probs[0]
                    L = -n * np.log(E_gamma_t_u) if E_gamma_t_u > 0 else -np.inf
            else:
                gamma_pow = gamma ** np.arange(max_t + 1)
                E_gamma_t_u = np.sum(probs * gamma_pow)
                L = np.log(gamma) * sum_t_x - n * np.log(E_gamma_t_u) if E_gamma_t_u > 0 else -np.inf
            if L > best_L:
                best_L = L
                best_R = R
                best_gamma = gamma
    return best_R, best_gamma

def main():
    n_realizations = 10
    ln_M_min = np.log(1e13)
    data_dir = "data/"
    strauss_path = os.path.join(data_dir, "strauss_parameters.npz")
    strauss = np.load(strauss_path)
    gamma_all = strauss['gamma']
    counts = []
    for i in range(n_realizations):
        filepath = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        galaxy_data = np.load(filepath)
        massive_mask = galaxy_data[:, 7] > ln_M_min
        massive_gals = galaxy_data[massive_mask]
        unique_masses, inverse_indices = np.unique(massive_gals[:, 7], return_inverse=True)
        c = 0
        for j in range(len(unique_masses)):
            halo_gals = massive_gals[inverse_indices == j]
            central_mask = halo_gals[:, 8] == 1
            if np.sum(central_mask) != 1:
                continue
            sat_mask = halo_gals[:, 8] == 0
            sat_coords = halo_gals[sat_mask, :3]
            if len(sat_coords) < 3:
                continue
            c += 1
        counts.append(c)
    if np.sum(counts) == len(gamma_all):
        gamma_per_realization = []
        idx = 0
        for c in counts:
            gamma_per_realization.append(np.mean(gamma_all[idx:idx+c]))
            idx += c
    else:
        print("Counts do not match (" + str(np.sum(counts)) + " vs " + str(len(gamma_all)) + "), re-running Strauss parameter estimation...")
        np.random.seed(42)
        boxsize = 500.0
        R_grid = np.linspace(0.1, 2.0, 20)
        gamma_grid = np.linspace(0.0, 1.0, 21)
        gamma_per_realization = []
        for i in range(n_realizations):
            filepath = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
            galaxy_data = np.load(filepath)
            massive_mask = galaxy_data[:, 7] > ln_M_min
            massive_gals = galaxy_data[massive_mask]
            unique_masses, inverse_indices = np.unique(massive_gals[:, 7], return_inverse=True)
            gamma_list = []
            for j in range(len(unique_masses)):
                halo_gals = massive_gals[inverse_indices == j]
                central_mask = halo_gals[:, 8] == 1
                if np.sum(central_mask) != 1:
                    continue
                central_coord = halo_gals[central_mask, :3][0]
                sat_mask = halo_gals[:, 8] == 0
                sat_coords = halo_gals[sat_mask, :3]
                if len(sat_coords) < 3:
                    continue
                sat_coords_rel = sat_coords - central_coord
                sat_coords_rel = sat_coords_rel - np.round(sat_coords_rel / boxsize) * boxsize
                R, gamma = fit_strauss_inhomogeneous(sat_coords_rel, R_grid, gamma_grid)
                if not np.isnan(R):
                    gamma_list.append(gamma)
            gamma_per_realization.append(np.mean(gamma_list))
    gamma_mean = np.array(gamma_per_realization)
    vpf_path = os.path.join(data_dir, "vpf_results.npz")
    vpf_data = np.load(vpf_path)
    P0_all = vpf_data['P0_all']
    P0_poisson = vpf_data['P0_poisson']
    vpf_dev = np.mean((P0_all - P0_poisson) / P0_poisson, axis=1)
    x = vpf_dev
    y = gamma_mean
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print("Joint Analysis: VPF Deviation vs Strauss Gamma Parameter")
    print("--------------------------------------------------------")
    print("Pearson correlation coefficient: " + str(np.round(pearson_corr, 4)) + " (p-value: " + str("%e" % pearson_p) + ")")
    print("Spearman correlation coefficient: " + str(np.round(spearman_corr, 4)) + " (p-value: " + str("%e" % spearman_p) + ")")
    print("Linear Regression: gamma = " + str(np.round(slope, 4)) + " * VPF_dev + " + str(np.round(intercept, 4)))
    print("R-squared: " + str(np.round(r_value**2, 4)))
    print("\nData per realization:")
    print("Realization | VPF Deviation | Mean Gamma")
    for i in range(n_realizations):
        print(str(i).ljust(11) + " | " + str(np.round(x[i], 4)).ljust(13) + " | " + str(np.round(y[i], 4)))
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='b', label='Realizations', s=50)
    x_line = np.linspace(np.min(x) - 0.05*np.abs(np.min(x)), np.max(x) + 0.05*np.abs(np.max(x)), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', label='Linear Fit (R^2 = ' + str(np.round(r_value**2, 2)) + ')')
    plt.xlabel('VPF Deviation Metric (Mean Normalized Residual)')
    plt.ylabel('Mean Strauss Interaction Parameter (gamma)')
    plt.title('Joint Analysis: VPF Leakage vs Gibbs Repulsion')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, "joint_analysis_vpf_gamma_1_" + str(timestamp) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("\nPlot saved to " + plot_path)
    plt.close()

if __name__ == '__main__':
    main()