# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
from scipy.stats import ks_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cdf_exp_profile(x):
    return 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2)

if __name__ == '__main__':
    input_data_dir = "/home/node/work/projects/pointproc_cosmology/data/"
    output_data_dir = "data/"
    rho_crit = 2.77536627e11
    L_box = 500.0
    print("--- Statistical Significance and Covariance ---\n")
    print("Computing metrics per realization...")
    ks_stats = []
    for i in range(10):
        halo_file = os.path.join(input_data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_file = os.path.join(input_data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        M_vir = halos[:, 6]
        sats = gals[gals[:, 8] == 0]
        halo_ln_M = np.log(M_vir)
        sat_ln_M = sats[:, 7]
        sort_idx = np.argsort(halo_ln_M)
        sorted_halo_ln_M = halo_ln_M[sort_idx]
        idx = np.searchsorted(sorted_halo_ln_M, sat_ln_M)
        idx = np.clip(idx, 1, len(sorted_halo_ln_M) - 1)
        left_diff = np.abs(sat_ln_M - sorted_halo_ln_M[idx - 1])
        right_diff = np.abs(sat_ln_M - sorted_halo_ln_M[idx])
        closest_sorted_idx = np.where(left_diff < right_diff, idx - 1, idx)
        matched_halo_idx = sort_idx[closest_sorted_idx]
        matched_M_vir = M_vir[matched_halo_idx]
        R_vir = (3.0 * matched_M_vir / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        dx = np.abs(sats[:, 0] - halos[matched_halo_idx, 0])
        dx = np.minimum(dx, L_box - dx)
        dy = np.abs(sats[:, 1] - halos[matched_halo_idx, 1])
        dy = np.minimum(dy, L_box - dy)
        dz = np.abs(sats[:, 2] - halos[matched_halo_idx, 2])
        dz = np.minimum(dz, L_box - dz)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        x = r / R_vir
        stat, _ = ks_1samp(x, cdf_exp_profile)
        ks_stats.append(stat)
    ks_stats = np.array(ks_stats)
    vpf_data = np.load(os.path.join(output_data_dir, "vpf_results.npz"))
    emp_P0 = vpf_data['empirical_P0']
    theo_P0 = vpf_data['theoretical_P0']
    vpf_residuals = np.mean(np.abs(emp_P0 - theo_P0), axis=1)
    rcrit_data = np.load(os.path.join(output_data_dir, "rcrit_results.npz"))
    r_centers_xi = rcrit_data['r_centers']
    xi_emp_all = rcrit_data['xi_emp_all']
    xi_dec_all = rcrit_data['xi_dec_all']
    r_crits = []
    for i in range(10):
        delta_xi = xi_emp_all[i] - xi_dec_all[i]
        rc = r_centers_xi[-1]
        for j in range(1, len(r_centers_xi)):
            if delta_xi[j-1] * delta_xi[j] < 0:
                r_prev = r_centers_xi[j-1]
                r_curr = r_centers_xi[j]
                val_prev = delta_xi[j-1]
                val_curr = delta_xi[j]
                fraction = np.abs(val_prev) / (np.abs(val_prev) + np.abs(val_curr))
                rc = r_prev + fraction * (r_curr - r_prev)
                break
        r_crits.append(rc)
    r_crits = np.array(r_crits)
    pcf3_data = np.load(os.path.join(output_data_dir, "3pcf_results.npz"))
    zeta_emp_all = pcf3_data['zeta_emp_all']
    zeta_eq = zeta_emp_all[:, 0, 7]
    zeta_sq = zeta_emp_all[:, 0, 9]
    mc_data = np.load(os.path.join(output_data_dir, "marked_correlation_results.npz"))
    M_r_all = mc_data['M_r_all']
    mean_Mr = np.nanmean(M_r_all, axis=1)
    X = np.column_stack((ks_stats, vpf_residuals, r_crits, zeta_eq, zeta_sq, mean_Mr))
    cov_matrix = np.cov(X, rowvar=False)
    corr_matrix = np.corrcoef(X, rowvar=False)
    print("\n--- Extracted Metrics per Realization ---")
    print("Realization | KS_Stat | VPF_Res | r_crit | 3PCF_Eq | 3PCF_Sq | M_r")
    for i in range(10):
        s = "     " + str(i).zfill(2) + "    | "
        s += str(np.round(ks_stats[i], 4)).rjust(7) + " | "
        s += str(np.round(vpf_residuals[i], 4)).rjust(7) + " | "
        s += str(np.round(r_crits[i], 2)).rjust(6) + " | "
        s += str(np.round(zeta_eq[i], 0)).rjust(7) + " | "
        s += str(np.round(zeta_sq[i], 0)).rjust(7) + " | "
        s += str(np.round(mean_Mr[i], 4)).rjust(5)
        print(s)
    print("\n--- Sample Covariance Matrix (10 Realizations) ---")
    header = ["KS_Stat", "VPF_Res", "r_crit", "3PCF_Eq", "3PCF_Sq", "M_r"]
    header_str = "".rjust(10) + " "
    for h in header:
        header_str += h.rjust(12) + " "
    print(header_str)
    for i in range(6):
        row_str = header[i].rjust(10) + " "
        for j in range(6):
            row_str += str(np.format_float_scientific(cov_matrix[i, j], precision=4)).rjust(12) + " "
        print(row_str)
    print("\n--- Sample Correlation Matrix ---")
    print(header_str)
    for i in range(6):
        row_str = header[i].rjust(10) + " "
        for j in range(6):
            row_str += str(np.round(corr_matrix[i, j], 4)).rjust(12) + " "
        print(row_str)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    emp_prof_data = np.load(os.path.join(output_data_dir, "empirical_satellite_profile.npz"))
    all_x_vals = emp_prof_data['all_x_vals']
    bins = np.linspace(0, 10, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_volumes = (4.0 * np.pi / 3.0) * (bins[1:]**3 - bins[:-1]**3)
    counts, _ = np.histogram(all_x_vals, bins=bins)
    n_empirical = counts / bin_volumes / len(all_x_vals)
    n_theoretical = np.exp(-bin_centers) / (8.0 * np.pi)
    ax.plot(bin_centers, n_theoretical, 'k--', label='Theoretical exp(-r/R_vir)')
    ax.plot(bin_centers, n_empirical, 'b-', label='Empirical (All Mass Bins)')
    ax.set_yscale('log')
    ax.set_xlabel('r / R_vir')
    ax.set_ylabel('Normalized Density')
    ax.set_title('Satellite Radial Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    r_bins_vpf = vpf_data['r_bins']
    emp_P0_mean = np.mean(emp_P0, axis=0)
    emp_P0_std = np.std(emp_P0, axis=0, ddof=1) / np.sqrt(10.0)
    theo_P0_mean = np.mean(theo_P0, axis=0)
    ax.errorbar(r_bins_vpf, emp_P0_mean, yerr=emp_P0_std, fmt='bo-', label='Empirical P0(r)')
    ax.plot(r_bins_vpf, theo_P0_mean, 'r--', label='Theoretical P0(r)')
    ax.set_xscale('log')
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel('P0(r)')
    ax.set_title('Void Probability Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 2]
    delta_xi = xi_emp_all - xi_dec_all
    mean_xi_emp = np.mean(xi_emp_all, axis=0)
    mean_delta_xi = np.mean(delta_xi, axis=0)
    std_delta_xi = np.std(delta_xi, axis=0, ddof=1)
    mean_xi_emp_safe = np.where(mean_xi_emp == 0, np.inf, mean_xi_emp)
    err_rel = (std_delta_xi / np.sqrt(10.0)) / np.abs(mean_xi_emp_safe)
    rel_diff = mean_delta_xi / mean_xi_emp_safe
    ax.errorbar(r_centers_xi, rel_diff, yerr=err_rel, fmt='go-', label='Delta_xi / xi_emp')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel('Relative Difference')
    ax.set_title('2PCF Decoupling Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    cos_centers = pcf3_data['cos_centers']
    zeta_dec_all = pcf3_data['zeta_dec_all']
    mean_zeta_emp = np.mean(zeta_emp_all[:, 0, :], axis=0)
    std_zeta_emp = np.std(zeta_emp_all[:, 0, :], axis=0, ddof=1) / np.sqrt(10.0)
    mean_zeta_dec = np.mean(zeta_dec_all[:, 0, :], axis=0)
    std_zeta_dec = np.std(zeta_dec_all[:, 0, :], axis=0, ddof=1) / np.sqrt(10.0)
    ax.errorbar(cos_centers, mean_zeta_emp, yerr=std_zeta_emp, fmt='bo-', label='Empirical')
    ax.errorbar(cos_centers, mean_zeta_dec, yerr=std_zeta_dec, fmt='r--', label='Decoupled')
    ax.set_yscale('symlog', linthresh=10.0)
    ax.set_xlabel('cos(theta)')
    ax.set_ylabel('zeta(r1, r2, theta)')
    ax.set_title('3PCF (r1, r2 in [1, 3] Mpc/h)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    r_centers_mc = mc_data['r_centers']
    mean_M_r_plot = np.nanmean(M_r_all, axis=0)
    std_M_r_plot = np.nanstd(M_r_all, axis=0, ddof=1) / np.sqrt(10.0)
    ax.errorbar(r_centers_mc, mean_M_r_plot, yerr=std_M_r_plot, fmt='mo-', label='Marked Correlation M(r)')
    ax.axhline(1.0, color='k', linestyle='--')
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel('M(r)')
    ax.set_title('Mass-Conditioned Marked Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)
    labels = ['KS Stat', 'VPF Res', 'r_crit', '3PCF Eq', '3PCF Sq', 'M(r)']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title('Metrics Correlation Matrix', pad=20)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = "summary_figure_1_" + timestamp + ".png"
    plot_filepath = os.path.join(output_data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    print("\nSummary figure saved to " + plot_filepath)