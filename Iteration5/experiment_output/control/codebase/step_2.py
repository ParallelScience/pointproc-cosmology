# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree

def get_auto_pairs(pos, bins, boxsize):
    tree = cKDTree(pos, boxsize=boxsize)
    counts = tree.count_neighbors(tree, bins, cumulative=True)
    return np.diff(counts) / 2.0

def get_cross_pairs(pos1, pos2, bins, boxsize):
    tree1 = cKDTree(pos1, boxsize=boxsize)
    tree2 = cKDTree(pos2, boxsize=boxsize)
    counts = tree1.count_neighbors(tree2, bins, cumulative=True)
    return np.diff(counts).astype(float)

def sci_fmt(val):
    if np.isnan(val):
        return "nan"
    if val == 0:
        return "0.00e+00"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10**exponent)
    return str(round(mantissa, 2)) + "e" + str(exponent)

def main():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    print("Starting Large-Scale Bias Estimation...\n")
    r_bins = np.linspace(20, 80, 13)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    L_box = 500.0
    M_sat = 1e13
    max_M = 0.0
    for i in range(10):
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        if os.path.exists(halo_file):
            halo_data = np.load(halo_file)
            max_M = max(max_M, np.max(halo_data[:, 6]))
    if max_M < M_sat:
        max_M = M_sat * 100
    mass_bins = np.logspace(np.log10(M_sat), np.log10(max_M * 1.01), 4)
    print("Defined Mass Bins (M_sun/h):")
    for k in range(3):
        print("  Bin " + str(k+1) + ": " + sci_fmt(mass_bins[k]) + " - " + sci_fmt(mass_bins[k+1]))
    print("Radial bins for bias estimation: " + str(r_bins[0]) + " to " + str(r_bins[-1]) + " Mpc/h (12 bins)\n")
    N_R = 100000
    np.random.seed(42)
    rand_pos = np.random.uniform(0, L_box, (N_R, 3))
    print("Computing RR pairs for random catalog (N_R = " + str(N_R) + ")...\n")
    RR_counts = get_auto_pairs(rand_pos, r_bins, L_box)
    RR_norm = RR_counts / (N_R * (N_R - 1) / 2.0)
    results_bias = np.zeros((10, 3))
    corr_dict = {'r_centers': r_centers, 'mass_bins': mass_bins, 'r_bins': r_bins}
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(halo_file):
            print("Warning: Files for realization " + str(i).zfill(2) + " not found. Skipping.")
            results_bias[i, :] = np.nan
            continue
        gal_data = np.load(gal_file)
        halo_data = np.load(halo_file)
        gal_pos = gal_data[:, 0:3]
        halo_pos = halo_data[:, 0:3]
        halo_M = halo_data[:, 6]
        N_g = len(gal_pos)
        D_g_R = get_cross_pairs(gal_pos, rand_pos, r_bins, L_box)
        D_g_R_norm = D_g_R / (N_g * N_R)
        print("Realization " + str(i).zfill(2) + ":")
        for k in range(3):
            mask = (halo_M >= mass_bins[k]) & (halo_M < mass_bins[k+1])
            halo_pos_k = halo_pos[mask]
            N_h = len(halo_pos_k)
            if N_h < 2:
                xi_hh = np.full(len(r_centers), np.nan)
                xi_gh = np.full(len(r_centers), np.nan)
                b_val = np.nan
                print("  Mass Bin " + str(k+1) + ": " + str(N_h) + " halos -> Not enough halos for correlation.")
            else:
                DD_hh = get_auto_pairs(halo_pos_k, r_bins, L_box)
                DR_hh = get_cross_pairs(halo_pos_k, rand_pos, r_bins, L_box)
                DD_hh_norm = DD_hh / (N_h * (N_h - 1) / 2.0)
                DR_hh_norm = DR_hh / (N_h * N_R)
                xi_hh = (DD_hh_norm - 2 * DR_hh_norm + RR_norm) / RR_norm
                D_g_D_h = get_cross_pairs(gal_pos, halo_pos_k, r_bins, L_box)
                D_g_D_h_norm = D_g_D_h / (N_g * N_h)
                xi_gh = (D_g_D_h_norm - D_g_R_norm - DR_hh_norm + RR_norm) / RR_norm
                valid_mask = (xi_hh > 0) & (xi_gh > 0)
                if np.any(valid_mask):
                    ratio_r = np.sqrt(xi_gh[valid_mask] / xi_hh[valid_mask])
                    b_val = np.mean(ratio_r)
                else:
                    b_val = np.nan
                print("  Mass Bin " + str(k+1) + ": " + str(N_h) + " halos, b(M) = " + str(round(b_val, 4)))
            results_bias[i, k] = b_val
            corr_dict["xi_hh_" + str(i).zfill(2) + "_" + str(k)] = xi_hh
            corr_dict["xi_gh_" + str(i).zfill(2) + "_" + str(k)] = xi_gh
    print("\nLarge-Scale Bias Estimates Summary (r > 20 Mpc/h):")
    for k in range(3):
        b_vals = results_bias[:, k]
        valid_b = b_vals[np.isfinite(b_vals)]
        if len(valid_b) > 0:
            mean_b = np.mean(valid_b)
            var_b = np.var(valid_b)
        else:
            mean_b = np.nan
            var_b = np.nan
        print("Mass Bin " + str(k+1) + " [" + sci_fmt(mass_bins[k]) + " - " + sci_fmt(mass_bins[k+1]) + " M_sun/h]:")
        print("  Mean Bias: " + str(round(mean_b, 4)))
        print("  Variance:  " + sci_fmt(var_b))
        print("  Values across realizations: " + str(np.round(b_vals, 4).tolist()))
    out_bias = os.path.join(output_dir, "bias_estimates.npy")
    out_corr = os.path.join(output_dir, "large_scale_bias_correlations.npz")
    np.save(out_bias, results_bias)
    np.savez(out_corr, **corr_dict)
    print("\nSaved bias estimates to " + out_bias)
    print("Saved correlation functions to " + out_corr)

if __name__ == '__main__':
    main()