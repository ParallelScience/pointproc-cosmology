# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False

def safe_cov(data):
    n_samples, n_features = data.shape
    cov = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            valid = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
            if np.sum(valid) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cov[i, j] = np.cov(data[valid, i], data[valid, j])[0, 1]
            else:
                cov[i, j] = np.nan
    return cov

def main():
    data_dir = "data/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rad_prof = np.load(os.path.join(data_dir, "radial_profiles.npz"))
    mle_fits = np.load(os.path.join(data_dir, "mle_fits.npz"))
    pcf = np.load(os.path.join(data_dir, "2pcf_1halo.npz"))
    mark_corr = np.load(os.path.join(data_dir, "marked_correlation.npz"))
    mass_bins = rad_prof['mass_bins']
    n_mass_bins = len(mass_bins) - 1
    x_bins = rad_prof['x_bins']
    x_bin_centers = rad_prof['x_bin_centers']
    density_norm_x = rad_prof['density_norm_x']
    n_sats = rad_prof['n_sats']
    n_halos = rad_prof['n_halos']
    mean_density_x = np.mean(density_norm_x, axis=0)
    cov_density_x = np.array([np.cov(density_norm_x[:, b, :], rowvar=False) for b in range(n_mass_bins)])
    err_density_x = np.sqrt(np.array([np.maximum(np.diag(cov_density_x[b]), 0) for b in range(n_mass_bins)]))
    mle_lambda = mle_fits['mle_lambda']
    mean_lambda = np.mean(mle_lambda, axis=0)
    V_shell_x = (4.0 / 3.0) * np.pi * (x_bins[1:]**3 - x_bins[:-1]**3)
    chi2_full_list = []
    chi2_diag_list = []
    dof_list = []
    for b in range(n_mass_bins):
        halos_b = n_halos[:, b]
        valid_halos = halos_b > 0
        if np.sum(valid_halos) > 0:
            mean_N_sat = np.mean(n_sats[valid_halos, b] / halos_b[valid_halos])
        else:
            mean_N_sat = 0.0
        lam = mean_lambda[b]
        x1 = x_bins[:-1]
        x2 = x_bins[1:]
        prob_bin = (1.0 / (2 * lam**2)) * ( np.exp(-x1/lam)*(x1**2 + 2*lam*x1 + 2*lam**2) - np.exp(-x2/lam)*(x2**2 + 2*lam*x2 + 2*lam**2) )
        model_density = mean_N_sat * prob_bin / V_shell_x
        diff = mean_density_x[b] - model_density
        inv_cov = np.linalg.pinv(cov_density_x[b])
        chi2_full = np.dot(diff.T, np.dot(inv_cov, diff))
        chi2_full_list.append(chi2_full)
        var = np.diag(cov_density_x[b])
        valid = var > 0
        chi2_diag = np.sum((diff[valid]**2) / var[valid])
        chi2_diag_list.append(chi2_diag)
        dof_list.append(np.sum(valid) - 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for b in range(n_mass_bins):
        ax = axes[b]
        valid = mean_density_x[b] > 0
        ax.errorbar(x_bin_centers[valid], mean_density_x[b][valid], yerr=err_density_x[b][valid], fmt='o', label='Data', markersize=4, capsize=2)
        halos_b = n_halos[:, b]
        valid_halos = halos_b > 0
        if np.sum(valid_halos) > 0:
            mean_N_sat = np.mean(n_sats[valid_halos, b] / halos_b[valid_halos])
        else:
            mean_N_sat = 0.0
        lam = mean_lambda[b]
        x1 = x_bins[:-1]
        x2 = x_bins[1:]
        prob_bin = (1.0 / (2 * lam**2)) * ( np.exp(-x1/lam)*(x1**2 + 2*lam*x1 + 2*lam**2) - np.exp(-x2/lam)*(x2**2 + 2*lam*x2 + 2*lam**2) )
        model_density = mean_N_sat * prob_bin / V_shell_x
        ax.plot(x_bin_centers, model_density, 'r-', label="Exp Fit (lambda=" + str(np.round(lam, 3)) + ")")
        ax.set_yscale('log')
        ax.set_xlabel('x = r/R_vir')
        ax.set_ylabel('Normalized Density n(x)')
        ax.set_title("Mass Bin: 10^" + str(mass_bins[b]) + " - 10^" + str(mass_bins[b+1]) + " M_sun/h")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rad_plot_path = os.path.join(data_dir, "radial_profiles_1_" + timestamp + ".png")
    plt.savefig(rad_plot_path, dpi=300)
    plt.close()
    print("Plot saved to " + rad_plot_path)
    r_bin_centers_pcf = pcf['r_bin_centers']
    xi_cs = pcf['xi_cs']
    xi_ss = pcf['xi_ss']
    xi_cs_null = pcf['xi_cs_null']
    xi_ss_null = pcf['xi_ss_null']
    mean_xi_cs = np.mean(xi_cs, axis=0)
    mean_xi_ss = np.mean(xi_ss, axis=0)
    mean_xi_cs_null = np.mean(xi_cs_null, axis=0)
    mean_xi_ss_null = np.mean(xi_ss_null, axis=0)
    cov_xi_cs = pcf['cov_xi_cs']
    cov_xi_ss = pcf['cov_xi_ss']
    cov_xi_cs_null = pcf['cov_xi_cs_null']
    cov_xi_ss_null = pcf['cov_xi_ss_null']
    err_xi_cs = np.sqrt(np.maximum(np.diag(cov_xi_cs), 0))
    err_xi_ss = np.sqrt(np.maximum(np.diag(cov_xi_ss), 0))
    err_xi_cs_null = np.sqrt(np.maximum(np.diag(cov_xi_cs_null), 0))
    err_xi_ss_null = np.sqrt(np.maximum(np.diag(cov_xi_ss_null), 0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    valid_cs = mean_xi_cs > 0
    ax1.errorbar(r_bin_centers_pcf[valid_cs], mean_xi_cs[valid_cs], yerr=err_xi_cs[valid_cs], fmt='o-', label='Data (C-S)', capsize=2)
    valid_cs_null = mean_xi_cs_null > 0
    ax1.errorbar(r_bin_centers_pcf[valid_cs_null], mean_xi_cs_null[valid_cs_null], yerr=err_xi_cs_null[valid_cs_null], fmt='s--', label='Null (C-S)', capsize=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('r [Mpc/h]')
    ax1.set_ylabel('xi_cs(r)')
    ax1.set_title('Central-Satellite 2PCF (1-halo)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    valid_ss = mean_xi_ss > 0
    ax2.errorbar(r_bin_centers_pcf[valid_ss], mean_xi_ss[valid_ss], yerr=err_xi_ss[valid_ss], fmt='o-', label='Data (S-S)', capsize=2)
    valid_ss_null = mean_xi_ss_null > 0
    ax2.errorbar(r_bin_centers_pcf[valid_ss_null], mean_xi_ss_null[valid_ss_null], yerr=err_xi_ss_null[valid_ss_null], fmt='s--', label='Null (S-S)', capsize=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('r [Mpc/h]')
    ax2.set_ylabel('xi_ss(r)')
    ax2.set_title('Satellite-Satellite 2PCF (1-halo)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    pcf_plot_path = os.path.join(data_dir, "2pcf_1halo_2_" + timestamp + ".png")
    plt.savefig(pcf_plot_path, dpi=300)
    plt.close()
    print("Plot saved to " + pcf_plot_path)
    r_bin_centers_mc = mark_corr['r_bin_centers']
    M_r_base_offset = mark_corr['M_r_base_offset']
    M_r_base_L = mark_corr['M_r_base_L']
    M_r_bins_offset = mark_corr['M_r_bins_offset']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_M_r_base_offset = np.nanmean(M_r_base_offset, axis=0)
        mean_M_r_base_L = np.nanmean(M_r_base_L, axis=0)
    cov_M_r_base_offset = safe_cov(M_r_base_offset)
    cov_M_r_base_L = safe_cov(M_r_base_L)
    err_M_r_base_offset = np.sqrt(np.maximum(np.diag(cov_M_r_base_offset), 0))
    err_M_r_base_L = np.sqrt(np.maximum(np.diag(cov_M_r_base_L), 0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(r_bin_centers_mc, mean_M_r_base_offset, yerr=err_M_r_base_offset, fmt='o-', capsize=2)
    ax1.axhline(1.0, color='k', linestyle='--')
    ax1.set_xlabel('r [Mpc/h]')
    ax1.set_ylabel('M(r)')
    ax1.set_title('Marked Correlation (Luminosity Offset)')
    ax1.grid(True, alpha=0.3)
    ax2.errorbar(r_bin_centers_mc, mean_M_r_base_L, yerr=err_M_r_base_L, fmt='o-', capsize=2)
    ax2.axhline(1.0, color='k', linestyle='--')
    ax2.set_xlabel('r [Mpc/h]')
    ax2.set_ylabel('M(r)')
    ax2.set_title('Marked Correlation (Absolute Luminosity)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    mc_plot_path = os.path.join(data_dir, "marked_correlation_3_" + timestamp + ".png")
    plt.savefig(mc_plot_path, dpi=300)
    plt.close()
    print("Plot saved to " + mc_plot_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for b in range(n_mass_bins):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_M_r_b = np.nanmean(M_r_bins_offset[:, b, :], axis=0)
        cov_M_r_b = safe_cov(M_r_bins_offset[:, b, :])
        err_M_r_b = np.sqrt(np.maximum(np.diag(cov_M_r_b), 0))
        ax.errorbar(r_bin_centers_mc, mean_M_r_b, yerr=err_M_r_b, fmt='o-', label="Mass: 10^" + str(mass_bins[b]) + " - 10^" + str(mass_bins[b+1]), capsize=2)
    ax.axhline(1.0, color='k', linestyle='--')
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel('M(r)')
    ax.set_title('Marked Correlation (Luminosity Offset) by Mass Bin')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mc_mass_plot_path = os.path.join(data_dir, "marked_correlation_mass_bins_4_" + timestamp + ".png")
    plt.savefig(mc_mass_plot_path, dpi=300)
    plt.close()
    print("Plot saved to " + mc_mass_plot_path)
    cov_path = os.path.join(data_dir, "aggregated_covariances.npz")
    np.savez(cov_path, cov_density_x=cov_density_x, cov_xi_cs=cov_xi_cs, cov_xi_ss=cov_xi_ss, cov_M_r_base_offset=cov_M_r_base_offset, cov_M_r_base_L=cov_M_r_base_L)
    print("Covariance matrices saved to " + cov_path)
    print("\n" + "="*50)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*50 + "\n")
    print("1. Radial Profile Model Fits (Exponential Kernel)")
    print("-" * 40)
    for b in range(n_mass_bins):
        print("Mass Bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "):")
        print("  Mean lambda/R_vir = " + str(np.round(mean_lambda[b], 4)) + " +/- " + str(np.round(np.std(mle_lambda[:, b]), 4)))
        print("  Chi^2 (full cov)  = " + str(np.round(chi2_full_list[b], 2)))
        print("  Chi^2 (diag only) = " + str(np.round(chi2_diag_list[b], 2)) + " / " + str(dof_list[b]) + " dof")
    print("\n")
    print("2. 1-Halo 2PCF Amplitude Ratios (Data / Null) at r ~ 0.5 Mpc/h")
    print("-" * 40)
    idx_05 = np.argmin(np.abs(r_bin_centers_pcf - 0.5))
    r_val = r_bin_centers_pcf[idx_05]
    print("At r = " + str(np.round(r_val, 3)) + " Mpc/h:")
    print("  xi_cs (Data) = " + str(np.round(mean_xi_cs[idx_05], 2)) + " +/- " + str(np.round(err_xi_cs[idx_05], 2)))
    print("  xi_cs (Null) = " + str(np.round(mean_xi_cs_null[idx_05], 2)) + " +/- " + str(np.round(err_xi_cs_null[idx_05], 2)))
    if mean_xi_cs_null[idx_05] > 0:
        ratio_cs = mean_xi_cs[idx_05] / mean_xi_cs_null[idx_05]
    else:
        ratio_cs = np.nan
    print("  Ratio (Data/Null) = " + str(np.round(ratio_cs, 2)))
    print("  xi_ss (Data) = " + str(np.round(mean_xi_ss[idx_05], 2)) + " +/- " + str(np.round(err_xi_ss[idx_05], 2)))
    print("  xi_ss (Null) = " + str(np.round(mean_xi_ss_null[idx_05], 2)) + " +/- " + str(np.round(err_xi_ss_null[idx_05], 2)))
    if mean_xi_ss_null[idx_05] > 0:
        ratio_ss = mean_xi_ss[idx_05] / mean_xi_ss_null[idx_05]
    else:
        ratio_ss = np.nan
    print("  Ratio (Data/Null) = " + str(np.round(ratio_ss, 2)))
    print("\n")
    print("3. Marked Correlation Function M(r) at r ~ 0.5 Mpc/h")
    print("-" * 40)
    idx_mc_05 = np.argmin(np.abs(r_bin_centers_mc - 0.5))
    r_mc_val = r_bin_centers_mc[idx_mc_05]
    print("At r = " + str(np.round(r_mc_val, 3)) + " Mpc/h:")
    print("  M(r) [Luminosity Offset] = " + str(np.round(mean_M_r_base_offset[idx_mc_05], 4)) + " +/- " + str(np.round(err_M_r_base_offset[idx_mc_05], 4)))
    print("  M(r) [Absolute L_sat]    = " + str(np.round(mean_M_r_base_L[idx_mc_05], 4)) + " +/- " + str(np.round(err_M_r_base_L[idx_mc_05], 4)))
    print("\n")

if __name__ == '__main__':
    main()