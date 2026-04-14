# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import minimize
import scipy.special as sc

def run_step_2():
    data_dir = "data/"
    sat_path = os.path.join(data_dir, "master_satellites.csv")
    if not os.path.exists(sat_path):
        print("Error: master_satellites.csv not found.")
        return
    df_sat = pd.read_csv(sat_path)
    Omega_m = 0.311
    rho_c = 2.77536627e11
    rho_m = Omega_m * rho_c
    Delta_m = 200.0
    n_h = 5000.0 / 500.0**3
    M0 = 5e13
    A_mass = 5000.0 / (M0**0.3 * sc.gamma(0.3))
    def R_vir(M):
        return (3.0 * M / (4.0 * np.pi * Delta_m * rho_m))**(1.0/3.0)
    def n_mass(M):
        return A_mass * M**(-0.7) * np.exp(-M / M0)
    def S_sel(r):
        return np.exp(- (4.0 * np.pi / 3.0) * r**3 * n_h)
    def integrand_r(r, Rvir):
        return (r**2 / (2.0 * Rvir**3)) * np.exp(-r / Rvir) * S_sel(r)
    M_grid = np.logspace(13, 15, 200)
    Rvir_grid = R_vir(M_grid)
    r_bins = np.linspace(0, 5, 21)
    N_rad_bins = len(r_bins) - 1
    I_k_grid = np.zeros((N_rad_bins, len(M_grid)))
    for i, R in enumerate(Rvir_grid):
        for k in range(N_rad_bins):
            r_fine = np.linspace(r_bins[k], r_bins[k+1], 50)
            I_k_grid[k, i] = simpson(integrand_r(r_fine, R), x=r_fine)
    splines_H_k = []
    for k in range(N_rad_bins):
        H_k = n_mass(M_grid) * I_k_grid[k, :]
        splines_H_k.append(CubicSpline(M_grid, H_k))
    M_bins = np.logspace(13, 15, 21)
    N_mass_bins = len(M_bins) - 1
    def compute_E_jk(M_sat, alpha_sat, n_realizations=1):
        E_jk = np.zeros((N_mass_bins, N_rad_bins))
        for j in range(N_mass_bins):
            M_start = max(M_bins[j], M_sat)
            M_end = M_bins[j+1]
            if M_start >= M_end:
                continue
            M_fine = np.linspace(M_start, M_end, 50)
            for k in range(N_rad_bins):
                H_k_val = splines_H_k[k](M_fine)
                integrand = H_k_val * (M_fine / M_sat)**alpha_sat
                E_jk[j, k] = simpson(integrand, x=M_fine) * n_realizations
        return E_jk
    df_sat_filtered = df_sat[(df_sat['r'] <= 5.0) & (df_sat['M_vir'] >= 1e13) & (df_sat['M_vir'] <= 1e15)]
    results = []
    for real_id in range(10):
        df_real = df_sat_filtered[df_sat_filtered['realization_id'] == real_id]
        O_jk, _, _ = np.histogram2d(df_real['M_vir'], df_real['r'], bins=[M_bins, r_bins])
        def nll(params):
            log_M_sat, alpha_sat = params
            M_sat = 10**log_M_sat
            E_jk = compute_E_jk(M_sat, alpha_sat, n_realizations=1)
            eps = 1e-10
            E_jk = np.clip(E_jk, eps, None)
            ll = np.sum(O_jk * np.log(E_jk) - E_jk)
            return -ll
        res = minimize(nll, [13.0, 1.0], bounds=[(12.0, 14.0), (0.5, 1.5)], method='L-BFGS-B')
        log_M_sat_fit, alpha_sat_fit = res.x
        results.append({'realization_id': real_id, 'log_M_sat': log_M_sat_fit, 'M_sat': 10**log_M_sat_fit, 'alpha_sat': alpha_sat_fit, 'nll': res.fun})
    results_df = pd.DataFrame(results)
    O_jk_joint, _, _ = np.histogram2d(df_sat_filtered['M_vir'], df_sat_filtered['r'], bins=[M_bins, r_bins])
    def nll_joint(params):
        log_M_sat, alpha_sat = params
        M_sat = 10**log_M_sat
        E_jk = compute_E_jk(M_sat, alpha_sat, n_realizations=10)
        eps = 1e-10
        E_jk = np.clip(E_jk, eps, None)
        ll = np.sum(O_jk_joint * np.log(E_jk) - E_jk)
        return -ll
    res_joint = minimize(nll_joint, [13.0, 1.0], bounds=[(12.0, 14.0), (0.5, 1.5)], method='L-BFGS-B')
    log_M_sat_joint, alpha_sat_joint = res_joint.x
    out_path = os.path.join(data_dir, "mle_hod_parameters.csv")
    results_df.to_csv(out_path, index=False)
    E_jk_joint = compute_E_jk(10**log_M_sat_joint, alpha_sat_joint, n_realizations=10)
    np.save(os.path.join(data_dir, "expected_counts_joint.npy"), E_jk_joint)
    np.save(os.path.join(data_dir, "observed_counts_joint.npy"), O_jk_joint)
if __name__ == '__main__':
    run_step_2()