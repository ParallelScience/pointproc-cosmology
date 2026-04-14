# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_jackknife_variance(data_matrix):
    total_sum = np.nansum(data_matrix, axis=0)
    total_count = np.sum(~np.isnan(data_matrix), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_val = total_sum / total_count
        val_i = np.nan_to_num(data_matrix)
        count_i = (~np.isnan(data_matrix)).astype(int)
        jk_means = (total_sum - val_i) / (total_count - count_i)
        jk_means[np.isinf(jk_means)] = np.nan
        jk_mean_of_means = np.nanmean(jk_means, axis=0)
        jk_var = ((total_count - 1) / total_count) * np.nansum((jk_means - jk_mean_of_means)**2, axis=0)
        jk_var[total_count <= 1] = np.nan
    return mean_val, jk_var

def main():
    data_dir = 'data/'
    halo_kl = np.load(os.path.join(data_dir, 'halo_K_L_functions.npz'))
    r_bins_kl = halo_kl['r_bins']
    L_all = halo_kl['L_all']
    L_dev_all = L_all - r_bins_kl
    mean_L_dev, var_L_dev = compute_jackknife_variance(L_dev_all)
    vpf = np.load(os.path.join(data_dir, 'vpf_results.npz'))
    r_bins_vpf = vpf['r_bins']
    P0_all = vpf['P0_all']
    mean_P0, var_P0 = compute_jackknife_variance(P0_all)
    j_func = np.load(os.path.join(data_dir, 'j_function_results.npz'))
    r_bins_j = j_func['r_bins']
    J_orig_all = j_func['J_orig_all']
    mean_J, var_J = compute_jackknife_variance(J_orig_all)
    mark = np.load(os.path.join(data_dir, 'mark_independence_results.npz'))
    bin_centers_mark = mark['bin_centers']
    sat_L_profiles = mark['sat_L_profiles']
    mean_mark, var_mark = compute_jackknife_variance(sat_L_profiles)
    strauss = np.load(os.path.join(data_dir, 'strauss_parameters.npz'))
    R_all = strauss['R']
    gamma_all = strauss['gamma']
    gibbs_data = np.column_stack((R_all, gamma_all))
    mean_gibbs, var_gibbs = compute_jackknife_variance(gibbs_data)
    np.set_printoptions(threshold=sys.maxsize, linewidth=150)
    print('Jackknife Variances (Diagonal of Covariance Matrix):')
    print('\n1. L-function Deviation (L(r) - r) Variances:')
    print(np.array2string(var_L_dev, separator=', '))
    print('\n2. VPF P_0(r) Variances:')
    print(np.array2string(var_P0, separator=', '))
    print('\n3. J-function J(r) Variances:')
    print(np.array2string(var_J, separator=', '))
    print('\n4. Mark Correlation (Satellite Luminosity) Variances:')
    print(np.array2string(var_mark, separator=', '))
    print('\n5. Gibbs Parameters Variances (R, gamma):')
    print(np.array2string(var_gibbs, separator=', '))
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    with np.errstate(invalid='ignore'):
        yerr_L = np.sqrt(var_L_dev)
    axs[0, 0].errorbar(r_bins_kl, mean_L_dev, yerr=yerr_L, fmt='b-', label='Mean L(r) - r', capsize=3)
    axs[0, 0].axhline(0, color='k', linestyle='--', label='CSR')
    axs[0, 0].set_xlabel('r [Mpc/h]')
    axs[0, 0].set_ylabel('L(r) - r [Mpc/h]')
    axs[0, 0].set_title('Halo Parent Process: L-function Deviation')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    with np.errstate(invalid='ignore'):
        yerr_P0 = np.sqrt(var_P0)
    axs[0, 1].errorbar(r_bins_vpf, mean_P0, yerr=yerr_P0, fmt='b-', label='Empirical VPF', capsize=3)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlabel('r [Mpc/h]')
    axs[0, 1].set_ylabel('P_0(r)')
    axs[0, 1].set_title('Void Probability Function (VPF)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which='both', linestyle='--', alpha=0.7)
    with np.errstate(invalid='ignore'):
        yerr_J = np.sqrt(var_J)
    axs[1, 0].errorbar(r_bins_j, mean_J, yerr=yerr_J, fmt='b-', label='J(r) Original', capsize=3)
    axs[1, 0].axhline(1, color='k', linestyle='--', label='CSR')
    axs[1, 0].set_xlabel('r [Mpc/h]')
    axs[1, 0].set_ylabel('J(r)')
    axs[1, 0].set_title('J-function')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    valid_bins = ~np.isnan(mean_mark)
    with np.errstate(invalid='ignore'):
        yerr_mark = np.sqrt(var_mark)
    axs[1, 1].errorbar(bin_centers_mark[valid_bins], mean_mark[valid_bins], yerr=yerr_mark[valid_bins], fmt='b-o', label='Satellite Luminosity', capsize=3)
    axs[1, 1].set_xlabel('Normalized Halo-centric Distance (r / R_vir)')
    axs[1, 1].set_ylabel('Mean Luminosity Mark (ln L)')
    axs[1, 1].set_title('Spatial-Mark Independence')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/summary_statistics_1_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    print('\nPlot saved to ' + plot_path)
    plt.close()

if __name__ == '__main__':
    main()