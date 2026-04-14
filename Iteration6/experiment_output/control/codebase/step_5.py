# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
import datetime

def main():
    data_dir = 'data'
    prof_data = np.load(os.path.join(data_dir, 'empirical_radial_profiles.npz'))
    mass_bins = prof_data['mass_bins']
    r_bins = prof_data['r_bins']
    r_centers = prof_data['r_centers']
    mean_profiles = prof_data['mean_profiles']
    var_profiles = prof_data['var_profiles']
    std_profiles = np.sqrt(var_profiles / 10.0)
    rec_data = np.load(os.path.join(data_dir, 'recovered_parameters.npz'))
    alpha_mle = rec_data['alpha_mle']
    alpha_std = rec_data['alpha_std']
    M_sat_fit = rec_data['M_sat_fit']
    M_sat_err = rec_data['M_sat_err']
    alpha_sat_fit = rec_data['alpha_sat_fit']
    alpha_sat_err = rec_data['alpha_sat_err']
    mean_masses = rec_data['mean_masses']
    N_sat_obs = rec_data['N_sat_obs']
    mark_data = np.load(os.path.join(data_dir, 'marked_correlation_results.npz'))
    mark_r_centers = mark_data['r_centers']
    mean_M_r = mark_data['mean_M_r']
    std_M_r = mark_data['std_M_r']
    mean_xi_r = mark_data['mean_xi_r']
    std_xi_r = mark_data['std_xi_r']
    mean_ratio = mark_data['mean_ratio']
    std_ratio = mark_data['std_ratio']
    Omega_m = 0.311
    rho_c = 2.775e11
    rho_m = Omega_m * rho_c
    R_vir_mean = (3.0 * mean_masses / (4.0 * np.pi * 200.0 * rho_m))**(1.0/3.0)
    V_shells = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    rho_th = np.zeros((3, len(r_centers)))
    for b in range(3):
        scale = alpha_mle * R_vir_mean[b]
        P_bin = np.exp(-r_bins[:-1] / scale) - np.exp(-r_bins[1:] / scale)
        rho_th[b, :] = N_sat_obs[b] * P_bin / V_shells
    residuals = mean_profiles - rho_th
    fractional_residuals = np.zeros_like(residuals)
    for b in range(3):
        valid = rho_th[b] > 0
        fractional_residuals[b][valid] = residuals[b][valid] / rho_th[b][valid]
    chi2_total = 0
    dof_total = 0
    chi2_list = []
    dof_list = []
    for b in range(3):
        valid = std_profiles[b] > 0
        chi2_b = np.sum(((mean_profiles[b][valid] - rho_th[b][valid]) / std_profiles[b][valid])**2)
        dof_b = np.sum(valid)
        chi2_total += chi2_b
        dof_total += dof_b
        chi2_list.append(chi2_b)
        dof_list.append(dof_b)
    print('==================================================')
    print('1-Halo Residual Analysis and Statistical Aggregation')
    print('==================================================')
    print('\n--- HOD Parameter Recovery ---')
    print('alpha (Radial Scale): Ground Truth = 1.0, Recovered = ' + str(np.round(alpha_mle, 4)) + ' +/- ' + str(np.round(alpha_std, 4)))
    print('M_sat (M_sun/h): Ground Truth = 1.00e+13, Recovered = ' + np.format_float_scientific(M_sat_fit, precision=2) + ' +/- ' + np.format_float_scientific(M_sat_err, precision=2))
    print('alpha_sat: Ground Truth = 1.0, Recovered = ' + str(np.round(alpha_sat_fit, 4)) + ' +/- ' + str(np.round(alpha_sat_err, 4)))
    print('\n--- Radial Profile Goodness-of-Fit ---')
    for b in range(3):
        print('Bin ' + str(b) + ' profile chi2 / dof: ' + str(np.round(chi2_list[b], 2)) + ' / ' + str(dof_list[b]))
    print('Total profile chi2 / dof: ' + str(np.round(chi2_total, 2)) + ' / ' + str(dof_total))
    print('\n--- Radial Profile Residuals ---')
    for b in range(3):
        mean_abs_res = np.mean(np.abs(residuals[b]))
        valid = rho_th[b] > 0
        mean_frac_res = np.mean(np.abs(fractional_residuals[b][valid]))
        print('Bin ' + str(b) + ': Mean Absolute Residual = ' + np.format_float_scientific(mean_abs_res, precision=3) + ', Mean Absolute Fractional Residual = ' + str(np.round(mean_frac_res, 4)))
    print('\n--- Marked Correlation Analysis ---')
    max_dev = np.max(np.abs(mean_M_r - 1.0))
    print('Max absolute deviation of M(r) from 1.0: ' + str(np.round(max_dev, 4)))
    print('Mean M(r) across bins: ' + str(np.round(np.mean(mean_M_r), 4)))
    print('\n--- M(r) / [1 + xi(r)] Ratio ---')
    for i in range(len(mark_r_centers)):
        print('r = ' + str(np.round(mark_r_centers[i], 2)) + ' Mpc/h: Ratio = ' + np.format_float_scientific(mean_ratio[i], precision=3) + ' +/- ' + np.format_float_scientific(std_ratio[i], precision=3))
    print('==================================================')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'multipanel_summary_1_' + timestamp + '.png')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    colors = ['C0', 'C1', 'C2']
    for b in range(3):
        ax.errorbar(r_centers, mean_profiles[b], yerr=std_profiles[b], fmt='o', color=colors[b], label='Bin ' + str(b) + ' Emp')
        ax.plot(r_centers, rho_th[b], '-', color=colors[b], label='Bin ' + str(b) + ' Model')
    ax.set_yscale('log')
    ax.set_xlabel('r (Mpc/h)')
    ax.set_ylabel('Density rho(r) [(Mpc/h)^-3]')
    ax.set_title('(a) Radial Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = axs[0, 1]
    color1 = 'tab:blue'
    ax2.set_xlabel('r (Mpc/h)')
    ax2.set_ylabel('M(r)', color=color1)
    ax2.errorbar(mark_r_centers, mean_M_r, yerr=std_M_r, fmt='o-', color=color1, label='M(r)')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(True, alpha=0.3)
    ax2_twin = ax2.twinx()
    color2 = 'tab:red'
    ax2_twin.set_ylabel('1 + xi(r)', color=color2)
    ax2_twin.errorbar(mark_r_centers, mean_xi_r + 1.0, yerr=std_xi_r, fmt='s--', color=color2, label='1 + xi(r)')
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2_twin.set_yscale('log')
    ax2.set_title('(b) Marked Correlation vs Unmarked Clustering')
    ax3 = axs[1, 0]
    for b in range(3):
        valid = rho_th[b] > 0
        yerr = np.zeros_like(std_profiles[b])
        yerr[valid] = std_profiles[b][valid] / rho_th[b][valid]
        ax3.errorbar(r_centers[valid], fractional_residuals[b][valid], yerr=yerr[valid], fmt='o-', color=colors[b], label='Bin ' + str(b))
    ax3.axhline(0, color='k', linestyle='--')
    ax3.set_xlabel('r (Mpc/h)')
    ax3.set_ylabel('Fractional Residual (Emp - Model) / Model')
    ax3.set_title('(c) Radial Profile Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    frac_res_all = []
    for b in range(3):
        valid = rho_th[b] > 0
        frac_res_all.extend(fractional_residuals[b][valid])
    if len(frac_res_all) > 0:
        min_val = max(-5.0, np.min(frac_res_all) - 0.5)
        max_val = min(5.0, np.max(frac_res_all) + 0.5)
        ax3.set_ylim(min_val, max_val)
    ax4 = axs[1, 1]
    ax4.axis('off')
    ax4.set_title('(d) HOD Parameter Recovery Summary')
    table_data = [['Parameter', 'Ground Truth', 'Recovered Value'], ['alpha (Radial Scale)', '1.0', str(np.round(alpha_mle, 4)) + ' +/- ' + str(np.round(alpha_std, 4))], ['M_sat (M_sun/h)', '1.00e+13', np.format_float_scientific(M_sat_fit, precision=2) + ' +/- ' + np.format_float_scientific(M_sat_err, precision=2)], ['alpha_sat', '1.0', str(np.round(alpha_sat_fit, 4)) + ' +/- ' + str(np.round(alpha_sat_err, 4))]]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
    results_filename = os.path.join(data_dir, 'final_residual_analysis.npz')
    np.savez(results_filename, rho_th=rho_th, residuals=residuals, fractional_residuals=fractional_residuals, chi2_total=chi2_total, dof_total=dof_total)
    print('Numerical results saved to ' + results_filename)

if __name__ == '__main__':
    main()