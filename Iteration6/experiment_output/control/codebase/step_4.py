# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import os

def compute_marked_correlation():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    r_bins = np.linspace(0.0, 5.0, 11)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    V_box = 500.0**3
    V_shells = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    M_r_all = []
    xi_r_all = []
    ratio_all = []
    for i in range(10):
        gal_path = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        gal_cat = np.load(gal_path)
        sat_mask = gal_cat[:, 8] == 0
        sat_cat = gal_cat[sat_mask]
        pos = sat_cat[:, 0:3]
        L = np.exp(sat_cat[:, 6])
        mean_L = np.mean(L)
        m = L / mean_L
        N_sat = len(pos)
        tree = cKDTree(pos, boxsize=500.0)
        pairs = tree.query_pairs(5.0, output_type='ndarray')
        if len(pairs) > 0:
            idx1 = pairs[:, 0]
            idx2 = pairs[:, 1]
            dx = np.abs(pos[idx1, 0] - pos[idx2, 0])
            dy = np.abs(pos[idx1, 1] - pos[idx2, 1])
            dz = np.abs(pos[idx1, 2] - pos[idx2, 2])
            dx = np.minimum(dx, 500.0 - dx)
            dy = np.minimum(dy, 500.0 - dy)
            dz = np.minimum(dz, 500.0 - dz)
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            mark_products = m[idx1] * m[idx2]
            DD, _ = np.histogram(dists, bins=r_bins)
            sum_marks, _ = np.histogram(dists, bins=r_bins, weights=mark_products)
            valid = DD > 0
            M_r = np.zeros_like(DD, dtype=float)
            M_r[valid] = sum_marks[valid] / DD[valid]
            RR = (N_sat * (N_sat - 1) / 2.0) * (V_shells / V_box)
            one_plus_xi = np.zeros_like(DD, dtype=float)
            valid_RR = RR > 0
            one_plus_xi[valid_RR] = DD[valid_RR] / RR[valid_RR]
            xi_r = one_plus_xi - 1.0
            ratio = np.zeros_like(DD, dtype=float)
            valid_ratio = one_plus_xi > 0
            ratio[valid_ratio] = M_r[valid_ratio] / one_plus_xi[valid_ratio]
            M_r_all.append(M_r)
            xi_r_all.append(xi_r)
            ratio_all.append(ratio)
        else:
            M_r_all.append(np.zeros(len(r_centers)))
            xi_r_all.append(np.zeros(len(r_centers)))
            ratio_all.append(np.zeros(len(r_centers)))
    M_r_all = np.array(M_r_all)
    xi_r_all = np.array(xi_r_all)
    ratio_all = np.array(ratio_all)
    mean_M_r = np.mean(M_r_all, axis=0)
    std_M_r = np.std(M_r_all, axis=0, ddof=1) / np.sqrt(10)
    mean_xi_r = np.mean(xi_r_all, axis=0)
    std_xi_r = np.std(xi_r_all, axis=0, ddof=1) / np.sqrt(10)
    mean_ratio = np.mean(ratio_all, axis=0)
    std_ratio = np.std(ratio_all, axis=0, ddof=1) / np.sqrt(10)
    print("--- Marked Correlation Analysis (1-halo regime, r < 5 Mpc/h) ---")
    for k in range(len(r_centers)):
        r_val = np.round(r_centers[k], 2)
        m_val = np.round(mean_M_r[k], 4)
        m_err = np.round(std_M_r[k], 4)
        xi_val = np.round(mean_xi_r[k] + 1.0, 2)
        xi_err = np.round(std_xi_r[k], 2)
        rat_val = np.format_float_scientific(mean_ratio[k], precision=3)
        rat_err = np.format_float_scientific(std_ratio[k], precision=3)
        print(str(r_val) + " | " + str(m_val) + " +/- " + str(m_err) + " | " + str(xi_val) + " +/- " + str(xi_err) + " | " + str(rat_val) + " +/- " + str(rat_err))
    output_path = os.path.join(output_dir, "marked_correlation_results.npz")
    np.savez(output_path, r_bins=r_bins, r_centers=r_centers, M_r_all=M_r_all, xi_r_all=xi_r_all, ratio_all=ratio_all, mean_M_r=mean_M_r, std_M_r=std_M_r, mean_xi_r=mean_xi_r, std_xi_r=std_xi_r, mean_ratio=mean_ratio, std_ratio=std_ratio)
    print("\nMarked correlation results saved to " + output_path)

if __name__ == '__main__':
    compute_marked_correlation()