# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import os

def gnfw_profile(x, A, gamma, x_s):
    return A * (x / x_s)**(-gamma) * (1.0 + x / x_s)**(gamma - 3.0)

if __name__ == '__main__':
    input_dir = '/home/node/work/projects/pointproc_cosmology/data'
    output_dir = 'data'
    L_box = 500.0
    x_bin1 = []
    x_bin2 = []
    print('Loading catalogs and matching satellites to host halos for mass binning...')
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
        mask_bin1 = (M_vir_valid > 1e13) & (M_vir_valid < 1e14)
        mask_bin2 = (M_vir_valid >= 1e14)
        x_bin1.extend(r_norm[mask_bin1])
        x_bin2.extend(r_norm[mask_bin2])
    x_bin1 = np.array(x_bin1)
    x_bin2 = np.array(x_bin2)
    print('Total satellites in Bin 1 (10^13 < M < 10^14): ' + str(len(x_bin1)))
    print('Total satellites in Bin 2 (M >= 10^14): ' + str(len(x_bin2)))
    bins = np.linspace(0, 5, 50)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    delta_x = bins[1] - bins[0]
    fit_params = []
    rho_emp_list = []
    rho_mod_list = []
    sigma_rho_list = []
    for bin_idx, (x_data, bin_name) in enumerate(zip([x_bin1, x_bin2], ['10^13 < M < 10^14', 'M >= 10^14'])):
        counts, _ = np.histogram(x_data, bins=bins)
        N_tot = len(x_data)
        volume_shell = 4.0 * np.pi * bin_centers**2 * delta_x
        rho_empirical = counts / (N_tot * volume_shell)
        sigma_counts = np.sqrt(counts)
        sigma_counts[counts == 0] = 1.0
        sigma_rho = sigma_counts / (N_tot * volume_shell)
        idx_1 = np.argmin(np.abs(bin_centers - 1.0))
        A_guess = rho_empirical[idx_1]
        if A_guess <= 0:
            A_guess = 0.1
        p0 = [A_guess, 1.0, 1.0]
        bounds = ([0.0, 0.0, 0.01], [np.inf, 2.99, 10.0])
        try:
            popt, pcov = curve_fit(gnfw_profile, bin_centers, rho_empirical, p0=p0, bounds=bounds, sigma=sigma_rho, absolute_sigma=True)
            A_fit, gamma_fit, x_s_fit = popt
            rho_model = gnfw_profile(bin_centers, A_fit, gamma_fit, x_s_fit)
            chi2 = np.sum(((rho_empirical - rho_model) / sigma_rho)**2)
            dof = len(bin_centers) - 3
            red_chi2 = chi2 / dof
            fit_params.append([A_fit, gamma_fit, x_s_fit, chi2, dof, red_chi2])
            rho_emp_list.append(rho_empirical)
            rho_mod_list.append(rho_model)
            sigma_rho_list.append(sigma_rho)
            print('\n--- Mass Bin: ' + bin_name + ' ---')
            print('Best-fit inner slope (gamma): ' + str(round(gamma_fit, 4)))
            print('Best-fit scale radius (R_s/R_vir): ' + str(round(x_s_fit, 4)))
            print('Normalization (A): ' + str(round(A_fit, 4)))
            print('Reduced Chi-squared: ' + str(round(red_chi2, 4)) + ' (Chi2: ' + str(round(chi2, 2)) + ', dof: ' + str(dof) + ')')
        except Exception as e:
            print('Fit failed for mass bin ' + bin_name + ': ' + str(e))
    np.save(os.path.join(output_dir, 'gnfw_fit_params.npy'), np.array(fit_params))
    np.save(os.path.join(output_dir, 'gnfw_rho_emp.npy'), np.array(rho_emp_list))
    np.save(os.path.join(output_dir, 'gnfw_rho_mod.npy'), np.array(rho_mod_list))
    np.save(os.path.join(output_dir, 'gnfw_sigma_rho.npy'), np.array(sigma_rho_list))
    np.save(os.path.join(output_dir, 'gnfw_bin_centers.npy'), bin_centers)
    print('\nFitted gNFW parameters and goodness-of-fit statistics saved to disk.')