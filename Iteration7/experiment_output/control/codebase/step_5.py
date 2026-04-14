# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['text.usetex'] = False

def run_step_5():
    data_dir = 'data/'
    sat_path = os.path.join(data_dir, 'master_satellites.csv')
    halo_path = os.path.join(data_dir, 'halo_satellite_counts.csv')
    
    if not os.path.exists(sat_path) or not os.path.exists(halo_path):
        print('Error: Required data files not found.')
        return
        
    df_sat = pd.read_csv(sat_path)
    df_halo = pd.read_csv(halo_path)
    
    df_halo = df_halo[(df_halo['M_vir'] >= 1e13) & (df_halo['M_vir'] <= 1e15)]
    M_vir_h = df_halo['M_vir'].values
    R_vir_h = df_halo['R_vir'].values
    
    M_bins = np.logspace(13, 15, 21)
    halo_mass_idx = np.digitize(M_vir_h, M_bins) - 1
    valid_halo = (halo_mass_idx >= 0) & (halo_mass_idx < 20)
    M_vir_h = M_vir_h[valid_halo]
    R_vir_h = R_vir_h[valid_halo]
    halo_mass_idx = halo_mass_idx[valid_halo]
    
    r_bins_fit = np.linspace(0, 5, 21)
    r_bins_trans = np.linspace(5, 10, 21)
    
    df_sat_fit = df_sat[(df_sat['r'] >= 0) & (df_sat['r'] <= 5) & (df_sat['M_vir'] >= 1e13) & (df_sat['M_vir'] <= 1e15)]
    O_jk_fit, _, _ = np.histogram2d(df_sat_fit['M_vir'], df_sat_fit['r'], bins=[M_bins, r_bins_fit])
    
    df_sat_trans = df_sat[(df_sat['r'] > 5) & (df_sat['r'] <= 10) & (df_sat['M_vir'] >= 1e13) & (df_sat['M_vir'] <= 1e15)]
    O_jk_trans, _, _ = np.histogram2d(df_sat_trans['M_vir'], df_sat_trans['r'], bins=[M_bins, r_bins_trans])
    
    n_h = 5000.0 / 500.0**3
    def S_sel(r):
        return np.exp(- (4.0 * np.pi / 3.0) * r**3 * n_h)
        
    def compute_E_jk(log_M_sat, alpha_sat, alpha_0, beta, r_bins):
        M_sat = 10**log_M_sat
        N_sat = (M_vir_h / M_sat)**alpha_sat
        alpha_c = alpha_0 * (M_vir_h / 1e13)**beta
        
        N_bins = len(r_bins) - 1
        E_jk = np.zeros((20, N_bins))
        
        for k in range(N_bins):
            r_fine = np.linspace(r_bins[k], r_bins[k+1], 6)
            r_mid = (r_fine[:-1] + r_fine[1:]) / 2
            dr = r_fine[1] - r_fine[0]
            
            integral = np.zeros(len(M_vir_h))
            for r in r_mid:
                val = (alpha_c**3 * r**2 / (2.0 * R_vir_h**3)) * np.exp(-alpha_c * r / R_vir_h) * S_sel(r)
                integral += val * dr
                
            lambda_ik = N_sat * integral
            E_jk[:, k] = np.bincount(halo_mass_idx, weights=lambda_ik, minlength=20)
            
        return E_jk

    def nll_const_alpha(params):
        log_M_sat, alpha_sat, alpha_0 = params
        beta = 0.0
        E_jk = compute_E_jk(log_M_sat, alpha_sat, alpha_0, beta, r_bins_fit)
        eps = 1e-10
        E_jk = np.clip(E_jk, eps, None)
        ll = np.sum(O_jk_fit * np.log(E_jk) - E_jk)
        return -ll

    def nll_mass_dep_alpha(params):
        log_M_sat, alpha_sat, alpha_0, beta = params
        E_jk = compute_E_jk(log_M_sat, alpha_sat, alpha_0, beta, r_bins_fit)
        eps = 1e-10
        E_jk = np.clip(E_jk, eps, None)
        ll = np.sum(O_jk_fit * np.log(E_jk) - E_jk)
        return -ll

    print('Fitting Constant-alpha Model...')
    init_const = [13.0, 1.0, 1.0]
    bounds_const = [(12.0, 14.0), (0.5, 2.0), (0.1, 5.0)]
    res_const = minimize(nll_const_alpha, init_const, bounds=bounds_const, method='L-BFGS-B')
    
    print('Fitting Mass-Dependent alpha Model...')
    init_mass_dep = [res_const.x[0], res_const.x[1], res_const.x[2], 0.0]
    bounds_mass_dep = [(12.0, 14.0), (0.5, 2.0), (0.1, 5.0), (-2.0, 2.0)]
    res_mass_dep = minimize(nll_mass_dep_alpha, init_mass_dep, bounds=bounds_mass_dep, method='L-BFGS-B')
    
    k_const = 3
    k_mass_dep = 4
    aic_const = 2 * k_const + 2 * res_const.fun
    aic_mass_dep = 2 * k_mass_dep + 2 * res_mass_dep.fun
    delta_aic = aic_const - aic_mass_dep
    
    print('\n--- Model Comparison Results ---')
    print('Constant-alpha Model:')
    print('  log_M_sat = ' + str(round(res_const.x[0], 3)) + ', alpha_sat = ' + str(round(res_const.x[1], 3)) + ', alpha_0 = ' + str(round(res_const.x[2], 3)))
    print('  NLL = ' + str(round(res_const.fun, 2)) + ', AIC = ' + str(round(aic_const, 2)))
    
    print('\nMass-Dependent alpha Model:')
    print('  log_M_sat = ' + str(round(res_mass_dep.x[0], 3)) + ', alpha_sat = ' + str(round(res_mass_dep.x[1], 3)) + ', alpha_0 = ' + str(round(res_mass_dep.x[2], 3)) + ', beta = ' + str(round(res_mass_dep.x[3], 3)))
    print('  NLL = ' + str(round(res_mass_dep.fun, 2)) + ', AIC = ' + str(round(aic_mass_dep, 2)))
    
    print('\nDelta AIC (Constant - Mass-Dependent) = ' + str(round(delta_aic, 2)))
    if delta_aic > 10:
        print('  -> Decisive evidence in favor of Mass-Dependent model.')
    elif delta_aic > 2:
        print('  -> Meaningful evidence in favor of Mass-Dependent model.')
    elif delta_aic < -10:
        print('  -> Decisive evidence in favor of Constant-alpha model.')
    elif delta_aic < -2:
        print('  -> Meaningful evidence in favor of Constant-alpha model.')
    else:
        print('  -> No strong preference between models.')
        
    if delta_aic > 0:
        best_params = res_mass_dep.x
        beta_best = best_params[3]
        model_name = 'Mass-Dependent'
    else:
        best_params = res_const.x
        beta_best = 0.0
        model_name = 'Constant-alpha'
        
    E_jk_trans = compute_E_jk(best_params[0], best_params[1], best_params[2], beta_best, r_bins_trans)
    
    O_trans_sum = np.sum(O_jk_trans, axis=0)
    E_trans_sum = np.sum(E_jk_trans, axis=0)
    
    print('\n--- Transition Region (r = 5-10 Mpc/h) Summary ---')
    print('Total Observed Satellites: ' + str(int(np.sum(O_trans_sum))))
    print('Total Expected Satellites (' + model_name + '): ' + str(round(np.sum(E_trans_sum), 1)))
    
    valid = E_trans_sum > 1e-5
    frac_residuals = np.zeros_like(O_trans_sum)
    frac_residuals[valid] = (O_trans_sum[valid] - E_trans_sum[valid]) / E_trans_sum[valid]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    M_plot = np.logspace(13, 15, 100)
    alpha_const_plot = np.full_like(M_plot, res_const.x[2])
    alpha_mass_dep_plot = res_mass_dep.x[2] * (M_plot / 1e13)**res_mass_dep.x[3]
    
    axes[0].plot(M_plot, alpha_const_plot, '--', label='Constant-alpha', color='blue', linewidth=2)
    axes[0].plot(M_plot, alpha_mass_dep_plot, '-', label='Mass-Dependent', color='red', linewidth=2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Halo Mass M_vir [M_sun/h]')
    axes[0].set_ylabel('Concentration Parameter alpha(M)')
    axes[0].set_title('Recovered Concentration-Mass Relation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)
    
    r_centers_trans = (r_bins_trans[:-1] + r_bins_trans[1:]) / 2
    
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].plot(r_centers_trans[valid], frac_residuals[valid], 'o-', color='purple', markersize=6, linewidth=2)
    axes[1].set_xlabel('Radial distance r [Mpc/h]')
    axes[1].set_ylabel('Fractional Residual (O - E) / E')
    axes[1].set_title('Transition Region Residuals (' + model_name + ' Model)')
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'model_comparison_residuals_1_' + timestamp + '.png'
    plot_path = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print('\nPlot saved to ' + plot_path)
    
    df_res = pd.DataFrame({
        'r_center': r_centers_trans,
        'Observed': O_trans_sum,
        'Expected': E_trans_sum,
        'Residual': O_trans_sum - E_trans_sum,
        'Frac_Residual': frac_residuals
    })
    res_path = os.path.join(data_dir, 'transition_residuals.csv')
    df_res.to_csv(res_path, index=False)
    print('Residuals data saved to ' + res_path)

if __name__ == '__main__':
    run_step_5()