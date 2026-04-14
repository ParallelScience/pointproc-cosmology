# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def main():
    data_dir = "data/"
    emp_sat_prof = np.load(os.path.join(data_dir, "empirical_satellite_profiles.npz"))
    ks_stats = emp_sat_prof['ks_stats']
    mass_bins_prof = emp_sat_prof['mass_bins']
    r_norm_centers = emp_sat_prof['r_norm_centers']
    empirical_profiles = emp_sat_prof['empirical_profiles']
    theo_vpf = np.load(os.path.join(data_dir, "theoretical_vpf.npz"))
    r_bins_vpf = theo_vpf['r_bins']
    vpf_mean_theo = theo_vpf['vpf_mean']
    res_vpf = np.load(os.path.join(data_dir, "residual_vpf_strauss.npz"))
    global_residuals = res_vpf['global_residuals']
    mass_bin_residuals = res_vpf['mass_bin_residuals']
    mass_bins_vpf = res_vpf['mass_bins']
    marked_corr = np.load(os.path.join(data_dir, "marked_corr_and_transition.npz"))
    r_centers_M = marked_corr['r_centers_M']
    all_M_r = marked_corr['all_M_r']
    r_centers_2pcf = marked_corr['r_centers_2pcf']
    all_xi_emp = marked_corr['all_xi_emp']
    all_xi_dec = marked_corr['all_xi_dec']
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs[0, 0]
    mean_prof = np.mean(empirical_profiles, axis=0)
    std_prof = np.std(empirical_profiles, axis=0)
    colors = ['C0', 'C1', 'C2']
    for i in range(len(mass_bins_prof)-1):
        m_low = mass_bins_prof[i]
        m_high = mass_bins_prof[i+1]
        label = str(m_low) + " - " + str(m_high) + " M_sun/h"
        ax.errorbar(r_norm_centers, mean_prof[i], yerr=std_prof[i], fmt='o-', color=colors[i], label=label, markersize=4)
    theo_shape = np.exp(-r_norm_centers)
    scale = 1.0
    for idx, val in enumerate(mean_prof[0]):
        if val > 0:
            scale = val / theo_shape[idx]
            break
    ax.plot(r_norm_centers, scale * theo_shape, 'k--', label="Theoretical slope ~ exp(-r/R_vir)")
    ax.set_yscale('log')
    ax.set_xlabel('Normalized Distance r / R_vir')
    ax.set_ylabel('Density')
    ax.set_title('Satellite Radial Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axs[0, 1]
    realizations = np.arange(len(ks_stats))
    ax.bar(realizations, ks_stats, color='skyblue', edgecolor='black')
    mean_ks = np.mean(ks_stats)
    ax.axhline(mean_ks, color='red', linestyle='--', label='Mean: ' + str(np.round(mean_ks, 4)))
    ax.set_xlabel('Realization ID')
    ax.set_ylabel('KS D-statistic')
    ax.set_title('KS Test D-statistics (Emp vs Theo CDF)')
    ax.set_xticks(realizations)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax = axs[0, 2]
    vpf_emp = vpf_mean_theo + global_residuals
    mean_vpf_emp = np.mean(vpf_emp, axis=0)
    std_vpf_emp = np.std(vpf_emp, axis=0)
    mean_vpf_theo = np.mean(vpf_mean_theo, axis=0)
    std_vpf_theo = np.std(vpf_mean_theo, axis=0)
    ax.errorbar(r_bins_vpf, mean_vpf_emp, yerr=std_vpf_emp, fmt='o-', label='Empirical VPF', color='blue', markersize=4)
    ax.errorbar(r_bins_vpf, mean_vpf_theo, yerr=std_vpf_theo, fmt='s--', label='Theoretical (MC) VPF', color='orange', markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel('Radius r (Mpc/h)')
    ax.set_ylabel('Void Probability P_0(r)')
    ax.set_title('Global Void Probability Function')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax_res = ax.twinx()
    mean_global_res = np.mean(global_residuals, axis=0)
    std_global_res = np.std(global_residuals, axis=0)
    ax_res.errorbar(r_bins_vpf, mean_global_res, yerr=std_global_res, fmt='^-', color='red', alpha=0.5, label='Residual (Emp - Theo)', markersize=4)
    ax_res.set_ylabel('Residual VPF', color='red')
    ax_res.tick_params(axis='y', labelcolor='red')
    ax_res.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax_res.legend(loc='upper right')
    ax = axs[1, 0]
    mean_mass_res = np.mean(mass_bin_residuals, axis=0)
    std_mass_res = np.std(mass_bin_residuals, axis=0)
    for i in range(len(mass_bins_vpf)-1):
        m_low = mass_bins_vpf[i]
        m_high = mass_bins_vpf[i+1]
        label = str(m_low) + " - " + str(m_high) + " M_sun/h"
        ax.errorbar(r_bins_vpf, mean_mass_res[i], yerr=std_mass_res[i], fmt='o-', label=label, markersize=4)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Radius r (Mpc/h)')
    ax.set_ylabel('Residual VPF (Empirical - Theoretical)')
    ax.set_title('Residual VPF by Halo Mass Bin')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axs[1, 1]
    mean_M_r = np.mean(all_M_r, axis=0)
    std_M_r = np.std(all_M_r, axis=0)
    ax.errorbar(r_centers_M, mean_M_r, yerr=std_M_r, fmt='o-', color='purple', markersize=4)
    ax.axhline(1.0, color='black', linestyle='--', label='Unmarked baseline (M=1)')
    ax.set_xlabel('Distance r (Mpc/h)')
    ax.set_ylabel('Marked Correlation M(r)')
    ax.set_title('Luminosity Marked Correlation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axs[1, 2]
    mean_xi_emp = np.mean(all_xi_emp, axis=0)
    std_xi_emp = np.std(all_xi_emp, axis=0)
    mean_xi_dec = np.mean(all_xi_dec, axis=0)
    std_xi_dec = np.std(all_xi_dec, axis=0)
    ax.errorbar(r_centers_2pcf, mean_xi_emp, yerr=std_xi_emp, fmt='o-', label='Empirical 2PCF', color='blue', markersize=4)
    ax.errorbar(r_centers_2pcf, mean_xi_dec, yerr=std_xi_dec, fmt='s--', label='Decoupled 2PCF', color='green', markersize=4)
    ax.set_xlabel('Distance r (Mpc/h)')
    ax.set_ylabel('2-Point Correlation Function xi(r)')
    ax.set_title('2PCF in Transition Regime (5-10 Mpc/h)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_filename = "spatial_statistics_summary_" + str(int(time.time())) + ".png"
    plot_filepath = os.path.join(data_dir, plot_filename)
    fig.savefig(plot_filepath, dpi=300)
    print("Plot saved to " + plot_filepath)
    print("\n=== Summary of Plotted Data ===")
    print("1. Satellite Radial Profiles (Mean Density at r/R_vir = 1.0):")
    idx_1 = np.argmin(np.abs(r_norm_centers - 1.0))
    for i in range(len(mass_bins_prof)-1):
        print("   Mass bin " + str(mass_bins_prof[i]) + " - " + str(mass_bins_prof[i+1]) + ": " + str(np.round(mean_prof[i, idx_1], 4)) + " +/- " + str(np.round(std_prof[i, idx_1], 4)))
    print("\n2. KS Test D-statistics:")
    print("   Mean: " + str(np.round(np.mean(ks_stats), 4)) + " +/- " + str(np.round(np.std(ks_stats), 4)))
    print("\n3. Global VPF (Empirical vs Theoretical) at r = 5.0 Mpc/h:")
    idx_5 = np.argmin(np.abs(r_bins_vpf - 5.0))
    print("   Empirical: " + str(np.round(mean_vpf_emp[idx_5], 4)) + " +/- " + str(np.round(std_vpf_emp[idx_5], 4)))
    print("   Theoretical: " + str(np.round(mean_vpf_theo[idx_5], 4)) + " +/- " + str(np.round(std_vpf_theo[idx_5], 4)))
    print("   Residual: " + str(np.round(mean_global_res[idx_5], 4)) + " +/- " + str(np.round(std_global_res[idx_5], 4)))
    print("\n4. Residual VPF by Mass Bin at r = 5.0 Mpc/h:")
    for i in range(len(mass_bins_vpf)-1):
        print("   Mass bin " + str(mass_bins_vpf[i]) + " - " + str(mass_bins_vpf[i+1]) + ": " + str(np.round(mean_mass_res[i, idx_5], 4)) + " +/- " + str(np.round(std_mass_res[i, idx_5], 4)))
    print("\n5. Marked Correlation M(r) at r = 1.25 Mpc/h:")
    idx_M = np.argmin(np.abs(r_centers_M - 1.25))
    print("   M(r): " + str(np.round(mean_M_r[idx_M], 4)) + " +/- " + str(np.round(std_M_r[idx_M], 4)))
    print("\n6. 2PCF in Transition Regime at r = 7.25 Mpc/h:")
    idx_2pcf = np.argmin(np.abs(r_centers_2pcf - 7.25))
    print("   Empirical: " + str(np.round(mean_xi_emp[idx_2pcf], 4)) + " +/- " + str(np.round(std_xi_emp[idx_2pcf], 4)))
    print("   Decoupled: " + str(np.round(mean_xi_dec[idx_2pcf], 4)) + " +/- " + str(np.round(std_xi_dec[idx_2pcf], 4)))

if __name__ == '__main__':
    main()