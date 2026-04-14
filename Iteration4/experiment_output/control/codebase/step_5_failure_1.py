# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import time

def gnfw_profile(x, A, gamma, x_s):
    return A * (x / x_s)**(-gamma) * (1.0 + x / x_s)**(gamma - 3.0)

if __name__ == '__main__':
    input_dir = '/home/node/work/projects/pointproc_cosmology/data'
    output_dir = 'data'
    L_box = 500.0
    bins_rad = np.linspace(0, 5, 50)
    bin_centers_rad = 0.5 * (bins_rad[1:] + bins_rad[:-1])
    delta_x = bins_rad[1] - bins_rad[0]
    volume_shell = 4.0 * np.pi * bin_centers_rad**2 * delta_x
    all_rho_emp = []
    gnfw_params_realization = []
    for i in range(10):
        gal_path = os.path.join(input_dir, 'galaxy_catalog_' + str(i).zfill(2) + '.npy')
        halo_path = os.path.join(input_dir, 'halo_catalog_' + str(i).zfill(2) + '.npy')
        gal_cat = np.load(gal_path)
        halo_cat = np.load(halo_path)
        satellites = gal_cat[gal_cat[:, 8] == 0]
        halo_log_M = np.log(halo_cat[:, 6]).reshape(-1, 1)
        tree_mass = cKDTree(halo_log_M)
        _, idxs_mass = tree_mass.query(satellites[:, 7].reshape(-1, 1))
        host_halos = halo_cat[idxs_mass]
        host_pos = host_halos[:, :3]
        M_vir = host_halos[:, 6]
        if halo_cat.shape[1] > 7:
            R_vir = host_halos[:, 7]
        else:
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
        M_vir_valid = M_vir[valid]
        counts, _ = np.histogram(r_norm, bins=bins_rad)
        rho_emp = counts / (len(r_norm) * volume_shell)
        all_rho_emp.append(rho_emp)
        mask_bin1 = (M_vir_valid > 1e13) & (M_vir_valid < 1e14)
        mask_bin2 = (M_vir_valid >= 1e14)
        r_bin1 = r_norm[mask_bin1]
        r_bin2 = r_norm[mask_bin2]
        params_i = []
        for r_data in [r_bin1, r_bin2]:
            c, _ = np.histogram(r_data, bins=bins_rad)
            rho_data = c / (len(r_data) * volume_shell)
            sigma_rho = np.sqrt(c) / (len(r_data) * volume_shell)
            sigma_rho[c == 0] = 1.0 / (len(r_data) * volume_shell)
            idx_1 = np.argmin(np.abs(bin_centers_rad - 1.0))
            A_guess = rho_data[idx_1] if rho_data[idx_1] > 0 else 0.1
            p0 = [A_guess, 1.0, 1.0]
            bounds = ([0.0, 0.0, 0.01], [np.inf, 2.99, 10.0])
            try:
                popt, _ = curve_fit(gnfw_profile, bin_centers_rad, rho_data, p0=p0, bounds=bounds, sigma=sigma_rho, absolute_sigma=True)
                params_i.extend([popt[0], popt[2]])
            except:
                params_i.extend([np.nan, np.nan])
        gnfw_params_realization.append(params_i)
    all_rho_emp = np.array(all_rho_emp)
    mean_rho_emp = np.mean(all_rho_emp, axis=0)
    std_rho_emp = np.std(all_rho_emp, axis=0)
    gnfw_params_realization = np.array(gnfw_params_realization)
    all_frac_res_xi = np.load(os.path.join(output_dir, 'all_frac_res_xi.npy'))
    spearman_results = np.load(os.path.join(output_dir, 'spearman_results.npy'))
    mean_frac_res_1halo = np.nanmean(all_frac_res_xi[:, :4], axis=1)
    spearman_corr = spearman_results[:, 0]
    feature_matrix = np.column_stack([mean_frac_res_1halo, spearman_corr, gnfw_params_realization[:, 0], gnfw_params_realization[:, 1], gnfw_params_realization[:, 2], gnfw_params_realization[:, 3]])
    valid_rows = ~np.isnan(feature_matrix).any(axis=1)
    feature_matrix_valid = feature_matrix[valid_rows]
    cov_matrix = np.cov(feature_matrix_valid, rowvar=False)
    corr_matrix = np.corrcoef(feature_matrix_valid, rowvar=False)
    feature_names = ['Frac Res 2PCF', 'Spearman r_s', 'A (Bin 1)', 'x_s (Bin 1)', 'A (Bin 2)', 'x_s (Bin 2)']
    radii_vpf = np.load(os.path.join(output_dir, 'radii_vpf.npy'))
    all_vpf_emp = np.load(os.path.join(output_dir, 'all_vpf_emp.npy'))
    all_vpf_syn = np.load(os.path.join(output_dir, 'all_vpf_syn.npy'))
    mean_vpf_emp = np.mean(all_vpf_emp, axis=0)
    std_vpf_emp = np.std(all_vpf_emp, axis=0)
    mean_vpf_syn = np.mean(all_vpf_syn, axis=0)
    std_vpf_syn = np.std(all_vpf_syn, axis=0)
    M_r_bin_centers = np.load(os.path.join(output_dir, 'M_r_bin_centers.npy'))
    all_M_r_all = np.load(os.path.join(output_dir, 'all_M_r_all.npy'))
    all_M_r_sat = np.load(os.path.join(output_dir, 'all_M_r_sat.npy'))
    mean_M_r_all = np.nanmean(all_M_r_all, axis=0)
    std_M_r_all = np.nanstd(all_M_r_all, axis=0)
    mean_M_r_sat = np.nanmean(all_M_r_sat, axis=0)
    std_M_r_sat = np.nanstd(all_M_r_sat, axis=0)
    gnfw_bin_centers = np.load(os.path.join(output_dir, 'gnfw_bin_centers.npy'))
    gnfw_rho_emp = np.load(os.path.join(output_dir, 'gnfw_rho_emp.npy'))
    gnfw_rho_mod = np.load(os.path.join(output_dir, 'gnfw_rho_mod.npy'))
    bin_centers_2pcf = np.load(os.path.join(output_dir, 'bin_centers_2pcf.npy'))
    mean_frac_res_xi = np.nanmean(all_frac_res_xi, axis=0)
    std_frac_res_xi = np.nanstd(all_frac_res_xi, axis=0)
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    ax.plot(bin_centers_rad, mean_rho_emp, 'k-', label='Empirical Mean')
    ax.fill_between(bin_centers_rad, mean_rho_emp - std_rho_emp, mean_rho_emp + std_rho_emp, color='k', alpha=0.2, label='+/- 1 std')
    pdf_theory = 0.5 * bin_centers_rad**2 * np.exp(-bin_centers_rad)
    rho_theory = pdf_theory / (4 * np.pi * bin_centers_rad**2)
    ax.plot(bin_centers_rad, rho_theory, 'b--', label='Theory: exp(-r/R_vir)')
    ax.set_xlabel('r / R_vir')
    ax.set_ylabel('rho(r / R_vir)')
    ax.set_yscale('log')
    ax.set_title('Mean Radial Density Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.errorbar(radii_vpf, mean_vpf_emp, yerr=std_vpf_emp, fmt='ko-', label='Empirical', capsize=3)
    ax.errorbar(radii_vpf, mean_vpf_syn, yerr=std_vpf_syn, fmt='r^--', label='Synthetic (KDE)', capsize=3)
    ax.set_xlabel('Radius (Mpc/h)')
    ax.set_ylabel('Void Probability P_0(r)')
    ax.set_yscale('log')
    ax.set_title('Void Probability Function (VPF)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 2]
    ax.errorbar(M_r_bin_centers, mean_M_r_all, yerr=std_M_r_all, fmt='s-', color='purple', label='All Galaxies', capsize=3)
    ax.errorbar(M_r_bin_centers, mean_M_r_sat, yerr=std_M_r_sat, fmt='o-', color='orange', label='Satellites Only', capsize=3)
    ax.set_xlabel('r (Mpc/h)')
    ax.set_ylabel('M(r)')
    ax.set_title('Luminosity Marked Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(gnfw_bin_centers, gnfw_rho_emp[0], 'ko', mfc='none', label='Empirical (Bin 1)')
    ax.plot(gnfw_bin_centers, gnfw_rho_mod[0], 'k-', label='gNFW Fit (Bin 1)')
    ax.plot(gnfw_bin_centers, gnfw_rho_emp[1], 'bs', mfc='none', label='Empirical (Bin 2)')
    ax.plot(gnfw_bin_centers, gnfw_rho_mod[1], 'b--', label='gNFW Fit (Bin 2)')
    ax.set_xlabel('r / R_vir')
    ax.set_ylabel('rho(r / R_vir)')
    ax.set_yscale('log')
    ax.set_title('Mass-Conditioned gNFW Fits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    short_names = ['FracRes', 'Spearman', 'A1', 'xs1', 'A2', 'xs2']
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            val = corr_matrix[i, j]
            val_str = 'NaN' if np.isnan(val) else str(round(val, 2))
            color = 'black' if np.isnan(val) or abs(val) < 0.5 else 'white'
            ax.text(j, i, val_str, ha='center', va='center', color=color, fontsize=8)
    ax.set_title('Correlation Matrix of Metrics')
    ax = axes[1, 2]
    ax.errorbar(bin_centers_2pcf, mean_frac_res_xi, yerr=std_frac_res_xi, fmt='ko-', capsize=3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('r (Mpc/h)')
    ax.set_ylabel('Delta xi / xi')
    ax.set_xscale('log')
    ax.set_title('Fractional 2PCF Residuals')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, 'summary_figure_5_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Summary figure saved to ' + plot_filename)
    np.save(os.path.join(output_dir, 'metrics_cov_matrix.npy'), cov_matrix)
    np.save(os.path.join(output_dir, 'metrics_corr_matrix.npy'), corr_matrix)
    np.save(os.path.join(output_dir, 'metrics_feature_matrix.npy'), feature_matrix_valid)