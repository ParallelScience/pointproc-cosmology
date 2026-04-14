# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import pickle

def compute_2pcf_optimized(pos_d, tree_r, rr_counts, Nr, bins, boxsize):
    tree_d = cKDTree(pos_d, boxsize=boxsize)
    Nd = float(len(pos_d))
    dd_counts = tree_d.count_neighbors(tree_d, bins, cumulative=True)
    dd = np.diff(dd_counts).astype(float)
    dr_counts = tree_d.count_neighbors(tree_r, bins, cumulative=True)
    dr = np.diff(dr_counts).astype(float)
    rr = np.diff(rr_counts).astype(float)
    f_dd = Nd * (Nd - 1.0)
    f_rr = Nr * (Nr - 1.0)
    f_dr = Nd * Nr
    rr_norm = rr / f_rr
    xi = np.zeros_like(dd, dtype=float)
    mask = rr_norm > 0
    xi[mask] = (dd[mask] / f_dd - 2.0 * dr[mask] / f_dr + rr_norm[mask]) / rr_norm[mask]
    return xi

def compute_vpf_optimized(pos_d, random_centers, radii, boxsize):
    tree_data = cKDTree(pos_d, boxsize=boxsize)
    dists, _ = tree_data.query(random_centers, k=1)
    vpf = []
    num_spheres = float(len(random_centers))
    for r in radii:
        empty_count = np.sum(dists > r)
        vpf.append(float(empty_count) / num_spheres)
    return np.array(vpf)

if __name__ == '__main__':
    np.random.seed(42)
    input_dir = '/home/node/work/projects/pointproc_cosmology/data'
    output_dir = 'data'
    L_box = 500.0
    with open(os.path.join(output_dir, 'kde_model.pkl'), 'rb') as f:
        kde = pickle.load(f)
    bins_2pcf = np.array([0.1, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0])
    bin_centers_2pcf = 0.5 * (bins_2pcf[:-1] + bins_2pcf[1:])
    radii_vpf = np.linspace(1, 30, 15)
    all_xi_emp = []
    all_xi_syn = []
    all_vpf_emp = []
    all_vpf_syn = []
    print('Starting Refined 1-Halo Structural Analysis...')
    for i in range(10):
        print('Processing realization ' + str(i) + '...')
        gal_path = os.path.join(input_dir, 'galaxy_catalog_' + str(i).zfill(2) + '.npy')
        halo_path = os.path.join(input_dir, 'halo_catalog_' + str(i).zfill(2) + '.npy')
        gal_cat = np.load(gal_path)
        halo_cat = np.load(halo_path)
        pos_emp = gal_cat[:, :3]
        N_gal = len(pos_emp)
        N_rand = 5 * N_gal
        randoms = np.random.uniform(0, L_box, (N_rand, 3))
        tree_r = cKDTree(randoms, boxsize=L_box)
        Nr = float(N_rand)
        rr_counts = tree_r.count_neighbors(tree_r, bins_2pcf, cumulative=True)
        num_spheres = 200000
        random_centers = np.random.uniform(0, L_box, (num_spheres, 3))
        xi_emp = compute_2pcf_optimized(pos_emp, tree_r, rr_counts, Nr, bins_2pcf, L_box)
        vpf_emp = compute_vpf_optimized(pos_emp, random_centers, radii_vpf, L_box)
        all_xi_emp.append(xi_emp)
        all_vpf_emp.append(vpf_emp)
        is_central = gal_cat[:, 8] == 1
        centrals = gal_cat[is_central]
        satellites = gal_cat[~is_central]
        halo_log_M = np.log(halo_cat[:, 6]).reshape(-1, 1)
        tree_mass = cKDTree(halo_log_M)
        _, idxs_mass = tree_mass.query(satellites[:, 7].reshape(-1, 1))
        host_halos = halo_cat[idxs_mass]
        host_pos = host_halos[:, :3]
        M_vir = host_halos[:, 6]
        Omega_m = 0.311
        rho_c = 2.77536627e11
        rho_m = Omega_m * rho_c
        R_vir = (M_vir / ((4.0 / 3.0) * np.pi * 200.0 * rho_m))**(1.0 / 3.0)
        r_norm_syn = np.abs(kde.resample(len(satellites))[0])
        r_new = r_norm_syn * R_vir
        cos_theta = np.random.uniform(-1, 1, len(satellites))
        phi = np.random.uniform(0, 2 * np.pi, len(satellites))
        sin_theta = np.sqrt(1 - cos_theta**2)
        dx = r_new * sin_theta * np.cos(phi)
        dy = r_new * sin_theta * np.sin(phi)
        dz = r_new * cos_theta
        sat_pos_syn = np.zeros((len(satellites), 3))
        sat_pos_syn[:, 0] = (host_pos[:, 0] + dx) % L_box
        sat_pos_syn[:, 1] = (host_pos[:, 1] + dy) % L_box
        sat_pos_syn[:, 2] = (host_pos[:, 2] + dz) % L_box
        pos_syn = np.vstack([centrals[:, :3], sat_pos_syn])
        xi_syn = compute_2pcf_optimized(pos_syn, tree_r, rr_counts, Nr, bins_2pcf, L_box)
        vpf_syn = compute_vpf_optimized(pos_syn, random_centers, radii_vpf, L_box)
        all_xi_syn.append(xi_syn)
        all_vpf_syn.append(vpf_syn)
    mean_xi_emp = np.mean(all_xi_emp, axis=0)
    mean_xi_syn = np.mean(all_xi_syn, axis=0)
    mean_vpf_emp = np.mean(all_vpf_emp, axis=0)
    mean_vpf_syn = np.mean(all_vpf_syn, axis=0)
    frac_res_xi = np.zeros_like(mean_xi_emp)
    mask = np.abs(mean_xi_syn) > 1e-6
    frac_res_xi[mask] = (mean_xi_emp[mask] - mean_xi_syn[mask]) / mean_xi_syn[mask]
    frac_res_xi[~mask] = np.nan
    print('\n--- 2PCF Fractional Residuals (Empirical vs KDE Synthetic) ---')
    print('Bin Centers (Mpc/h) | Empirical xi | Synthetic xi | Fractional Residual (Delta xi / xi)')
    print('-' * 85)
    for i in range(len(bin_centers_2pcf)):
        res_str = str(round(frac_res_xi[i], 4)) if not np.isnan(frac_res_xi[i]) else 'NaN'
        print(str(round(bin_centers_2pcf[i], 2)).ljust(19) + ' | ' + str(round(mean_xi_emp[i], 4)).ljust(12) + ' | ' + str(round(mean_xi_syn[i], 4)).ljust(12) + ' | ' + res_str)
    print('\n--- VPF (Void Probability Function) ---')
    print('Radius (Mpc/h) | Empirical VPF | Synthetic VPF')
    print('-' * 50)
    for i in range(len(radii_vpf)):
        print(str(round(radii_vpf[i], 2)).ljust(14) + ' | ' + str(round(mean_vpf_emp[i], 6)).ljust(13) + ' | ' + str(round(mean_vpf_syn[i], 6)))
    np.save(os.path.join(output_dir, 'all_xi_emp.npy'), np.array(all_xi_emp))
    np.save(os.path.join(output_dir, 'all_xi_syn.npy'), np.array(all_xi_syn))
    np.save(os.path.join(output_dir, 'all_vpf_emp.npy'), np.array(all_vpf_emp))
    np.save(os.path.join(output_dir, 'all_vpf_syn.npy'), np.array(all_vpf_syn))
    syn_arr = np.array(all_xi_syn)
    emp_arr = np.array(all_xi_emp)
    all_frac_res_xi = np.zeros_like(emp_arr)
    mask_all = np.abs(syn_arr) > 1e-6
    all_frac_res_xi[mask_all] = (emp_arr[mask_all] - syn_arr[mask_all]) / syn_arr[mask_all]
    all_frac_res_xi[~mask_all] = np.nan
    np.save(os.path.join(output_dir, 'all_frac_res_xi.npy'), all_frac_res_xi)
    np.save(os.path.join(output_dir, 'mean_xi_emp.npy'), np.column_stack((bin_centers_2pcf, mean_xi_emp)))
    np.save(os.path.join(output_dir, 'mean_xi_syn.npy'), np.column_stack((bin_centers_2pcf, mean_xi_syn)))
    np.save(os.path.join(output_dir, 'frac_res_xi.npy'), np.column_stack((bin_centers_2pcf, frac_res_xi)))
    np.save(os.path.join(output_dir, 'mean_vpf_emp.npy'), np.column_stack((radii_vpf, mean_vpf_emp)))
    np.save(os.path.join(output_dir, 'mean_vpf_syn.npy'), np.column_stack((radii_vpf, mean_vpf_syn)))
    np.save(os.path.join(output_dir, 'bins_2pcf.npy'), bins_2pcf)
    np.save(os.path.join(output_dir, 'bin_centers_2pcf.npy'), bin_centers_2pcf)
    np.save(os.path.join(output_dir, 'radii_vpf.npy'), radii_vpf)
    print('\nVPF, 2PCF, and residual metrics saved to disk.')