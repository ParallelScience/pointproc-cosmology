# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['text.usetex'] = False

def run_step_3():
    data_dir = 'data/'
    sat_path = os.path.join(data_dir, 'master_satellites.csv')
    halo_path = os.path.join(data_dir, 'halo_satellite_counts.csv')
    mle_path = os.path.join(data_dir, 'mle_hod_parameters.csv')
    if not os.path.exists(sat_path) or not os.path.exists(halo_path) or not os.path.exists(mle_path):
        print('Error: Required data files not found.')
        return
    df_sat = pd.read_csv(sat_path)
    df_halo = pd.read_csv(halo_path)
    mle_df = pd.read_csv(mle_path)
    mean_log_M_sat = mle_df['log_M_sat'].mean()
    mean_alpha_sat = mle_df['alpha_sat'].mean()
    M_sat_fit = 10**mean_log_M_sat
    alpha_sat_fit = mean_alpha_sat
    print('Using Best-Fit HOD Parameters: log(M_sat) = ' + str(round(mean_log_M_sat, 3)) + ', alpha_sat = ' + str(round(mean_alpha_sat, 3)))
    mass_bins = [13.0, 13.5, 14.0, 14.5, 15.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    r_bins = np.linspace(0, 5, 25)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    shell_volumes = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    print('\n--- Radial Profile Summary ---')
    for i in range(len(mass_bins) - 1):
        m_low = mass_bins[i]
        m_high = mass_bins[i+1]
        mask_halo = (np.log10(df_halo['M_vir']) >= m_low) & (np.log10(df_halo['M_vir']) < m_high)
        halos_in_bin = df_halo[mask_halo]
        N_halos = len(halos_in_bin)
        if N_halos == 0:
            continue
        mask_sat = (np.log10(df_sat['M_vir']) >= m_low) & (np.log10(df_sat['M_vir']) < m_high)
        sats_in_bin = df_sat[mask_sat]
        N_sats = len(sats_in_bin)
        mean_R_vir = halos_in_bin['R_vir'].mean()
        print('Mass Bin ' + str(m_low) + '-' + str(m_high) + ': ' + str(N_halos) + ' halos, ' + str(N_sats) + ' satellites, Mean R_vir = ' + str(round(mean_R_vir, 3)) + ' Mpc/h')
        counts, _ = np.histogram(sats_in_bin['r'], bins=r_bins)
        empirical_density = counts / (N_halos * shell_volumes)
        err_density = np.sqrt(counts) / (N_halos * shell_volumes)
        r_fine = np.linspace(0, 5, 100)
        n_th_fine = np.zeros_like(r_fine)
        n_th_actual_fine = np.zeros_like(r_fine)
        M_vir_arr = halos_in_bin['M_vir'].values
        R_vir_arr = halos_in_bin['R_vir'].values
        N_sat_actual = halos_in_bin['N_sat'].values
        N_sat_expected = np.where(M_vir_arr > M_sat_fit, (M_vir_arr / M_sat_fit)**alpha_sat_fit, 0.0)
        for j, r_val in enumerate(r_fine):
            density_contributions = (N_sat_expected / (8.0 * np.pi * R_vir_arr**3)) * np.exp(-r_val / R_vir_arr)
            n_th_fine[j] = np.sum(density_contributions) / N_halos
            density_contributions_actual = (N_sat_actual / (8.0 * np.pi * R_vir_arr**3)) * np.exp(-r_val / R_vir_arr)
            n_th_actual_fine[j] = np.sum(density_contributions_actual) / N_halos
        ax = axes[i]
        valid = counts > 0
        ax.errorbar(r_centers[valid], empirical_density[valid], yerr=err_density[valid], fmt='o', label='Empirical', color='black', markersize=5, capsize=3)
        ax.plot(r_fine, n_th_fine, '-', label='Best-Fit HOD Exp Model', color='red', linewidth=2)
        ax.plot(r_fine, n_th_actual_fine, '--', label='Actual N_sat Exp Model', color='blue', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Radial distance r [Mpc/h]')
        ax.set_ylabel('Satellite Density n(r) [h^3/Mpc^3]')
        ax.set_title('Mass Bin: 10^' + str(m_low) + ' - 10^' + str(m_high) + ' M_sun/h')
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.set_xlim(0, 5)
        if np.any(valid):
            min_val = np.min(empirical_density[valid])
            max_val = np.max(empirical_density[valid])
            ax.set_ylim(bottom=max(min_val*0.5, 1e-7), top=max_val*2)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'satellite_radial_profiles_1_' + timestamp + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('\nDiagnostic plot saved to ' + plot_path)

if __name__ == '__main__':
    run_step_3()