# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, kstest, gamma
import matplotlib.pyplot as plt
import time
import pickle

if __name__ == '__main__':
    input_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    L_box = 500.0
    all_r_norm = []
    all_r = []
    print("Loading catalogs and matching satellites to host halos...")
    for i in range(10):
        gal_path = os.path.join(input_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halo_path = os.path.join(input_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_cat = np.load(gal_path)
        halo_cat = np.load(halo_path)
        satellites = gal_cat[gal_cat[:, 8] == 0]
        halo_log_M = np.log(halo_cat[:, 6]).reshape(-1, 1)
        tree_mass = cKDTree(halo_log_M)
        dists_mass, idxs_mass = tree_mass.query(satellites[:, 7].reshape(-1, 1))
        host_halos = halo_cat[idxs_mass]
        host_pos = host_halos[:, :3]
        if halo_cat.shape[1] > 7:
            R_vir = host_halos[:, 7]
        else:
            M_vir = host_halos[:, 6]
            Omega_m = 0.311
            rho_c = 2.77536627e11
            rho_m = Omega_m * rho_c
            R_vir = (M_vir / ((4.0 / 3.0) * np.pi * 200.0 * rho_m))**(1.0 / 3.0)
        dx = np.abs(satellites[:, 0] - host_pos[:, 0])
        dy = np.abs(satellites[:, 1] - host_pos[:, 1])
        dz = np.abs(satellites[:, 2] - host_pos[:, 2])
        dx = np.minimum(dx, L_box - dx)
        dy = np.minimum(dy, L_box - dy)
        dz = np.minimum(dz, L_box - dz)
        dists = np.sqrt(dx**2 + dy**2 + dz**2)
        valid = R_vir > 0
        r_norm = dists[valid] / R_vir[valid]
        all_r_norm.extend(r_norm)
        all_r.extend(dists[valid])
    all_r_norm = np.array(all_r_norm)
    all_r = np.array(all_r)
    print("Total satellites processed across 10 realizations: " + str(len(all_r_norm)))
    print("Fitting KDE to the empirical radial distribution...")
    kde = gaussian_kde(all_r_norm, bw_method='scott')
    bins = np.linspace(0, 5, 50)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    hist_1d, _ = np.histogram(all_r_norm, bins=bins, density=True)
    rho_empirical = hist_1d / (4 * np.pi * bin_centers**2)
    pdf_theory_1 = np.exp(-bin_centers)
    rho_theory_1 = pdf_theory_1 / (4 * np.pi * bin_centers**2)
    pdf_theory_2 = 0.5 * bin_centers**2 * np.exp(-bin_centers)
    rho_theory_2 = pdf_theory_2 / (4 * np.pi * bin_centers**2)
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(bin_centers, hist_1d, 'ko', label='Empirical 1D PDF')
    axes[0].plot(bin_centers, pdf_theory_1, 'r-', label='P(r) ~ exp(-r/R_vir) (Missing Jacobian)')
    axes[0].plot(bin_centers, pdf_theory_2, 'b--', label='P(r) ~ r^2 exp(-r/R_vir) (Correct 3D Exp)')
    axes[0].plot(bin_centers, kde(bin_centers), 'g:', lw=2, label='KDE Null Model')
    axes[0].set_xlabel('r / R_vir')
    axes[0].set_ylabel('P(r / R_vir)')
    axes[0].set_title('1D Radial Probability Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(bin_centers, rho_empirical, 'ko', label='Empirical 3D Density')
    axes[1].plot(bin_centers, rho_theory_1, 'r-', label='rho(r) ~ r^-2 exp(-r/R_vir) (Missing Jacobian)')
    axes[1].plot(bin_centers, rho_theory_2, 'b--', label='rho(r) ~ exp(-r/R_vir) (Correct 3D Exp)')
    axes[1].set_xlabel('r / R_vir')
    axes[1].set_ylabel('rho(r / R_vir)')
    axes[1].set_yscale('log')
    axes[1].set_title('3D Radial Density Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if np.any(rho_empirical > 0):
        max_rho = np.max(rho_empirical[rho_empirical > 0])
        axes[1].set_ylim(1e-4, max_rho * 5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, "radial_density_profile_1_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)
    print("\n--- Goodness-of-Fit Statistics ---")
    counts, bin_edges = np.histogram(all_r_norm, bins=50, range=(0, 5))
    expected_counts_1 = len(all_r_norm) * (np.exp(-bin_edges[:-1]) - np.exp(-bin_edges[1:]))
    valid_1 = expected_counts_1 > 0
    chi2_1 = np.sum((counts[valid_1] - expected_counts_1[valid_1])**2 / expected_counts_1[valid_1])
    dof_1 = np.sum(valid_1) - 1
    cdf_2 = gamma.cdf(bin_edges, a=3)
    expected_counts_2 = len(all_r_norm) * (cdf_2[1:] - cdf_2[:-1])
    valid_2 = expected_counts_2 > 0
    chi2_2 = np.sum((counts[valid_2] - expected_counts_2[valid_2])**2 / expected_counts_2[valid_2])
    dof_2 = np.sum(valid_2) - 1
    ks_stat_1, ks_p_1 = kstest(all_r_norm, 'expon')
    ks_stat_2, ks_p_2 = kstest(all_r_norm, 'gamma', args=(3,))
    print("Model 1: Missing Jacobian (P(r) ~ exp(-r/R_vir))")
    print("  Chi-squared: " + str(round(chi2_1, 2)) + " (dof=" + str(dof_1) + ")")
    print("  KS Statistic: " + str(round(ks_stat_1, 4)) + ", p-value: " + str(ks_p_1))
    print("\nModel 2: Correct 3D Exponential (P(r) ~ r^2 exp(-r/R_vir))")
    print("  Chi-squared: " + str(round(chi2_2, 2)) + " (dof=" + str(dof_2) + ")")
    print("  KS Statistic: " + str(round(ks_stat_2, 4)) + ", p-value: " + str(ks_p_2))
    np.save(os.path.join(output_dir, "empirical_1d_pdf.npy"), np.column_stack((bin_centers, hist_1d)))
    np.save(os.path.join(output_dir, "empirical_3d_density.npy"), np.column_stack((bin_centers, rho_empirical)))
    np.save(os.path.join(output_dir, "kde_null_model_data.npy"), all_r_norm)
    with open(os.path.join(output_dir, "kde_model.pkl"), "wb") as f:
        pickle.dump(kde, f)
    print("\nComputed profiles and KDE model saved to disk.")