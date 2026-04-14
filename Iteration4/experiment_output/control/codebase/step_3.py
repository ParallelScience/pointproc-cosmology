# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

def compute_marked_correlation(pos, marks, bins, boxsize):
    tree = cKDTree(pos, boxsize=boxsize)
    max_dist = bins[-1]
    pairs_set = tree.query_pairs(max_dist)
    pairs = np.array(list(pairs_set))
    if len(pairs) == 0:
        return np.full(len(bins) - 1, np.nan)
    idx1 = pairs[:, 0]
    idx2 = pairs[:, 1]
    dx = np.abs(pos[idx1, 0] - pos[idx2, 0])
    dy = np.abs(pos[idx1, 1] - pos[idx2, 1])
    dz = np.abs(pos[idx1, 2] - pos[idx2, 2])
    dx = np.minimum(dx, boxsize - dx)
    dy = np.minimum(dy, boxsize - dy)
    dz = np.minimum(dz, boxsize - dz)
    dists = np.sqrt(dx**2 + dy**2 + dz**2)
    marks_prod = marks[idx1] * marks[idx2]
    mean_mark = np.mean(marks)
    M_r = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        mask = (dists >= bins[b]) & (dists < bins[b+1])
        if np.sum(mask) > 0:
            M_r[b] = np.mean(marks_prod[mask]) / (mean_mark**2)
        else:
            M_r[b] = np.nan
    return M_r

if __name__ == '__main__':
    input_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    L_box = 500.0
    bins = np.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    all_M_r_all = []
    all_M_r_sat = []
    spearman_results = []
    print("Computing Marked Correlation Function M(r) and Spearman rank correlation...")
    print("Bins (Mpc/h): " + str(bins))
    for i in range(10):
        gal_path = os.path.join(input_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        gal_cat = np.load(gal_path)
        pos = gal_cat[:, :3]
        ln_L = gal_cat[:, 6]
        L = np.exp(ln_L)
        is_central = gal_cat[:, 8] == 1
        M_r_all = compute_marked_correlation(pos, L, bins, L_box)
        all_M_r_all.append(M_r_all)
        sat_mask = ~is_central
        pos_sat = pos[sat_mask]
        L_sat = L[sat_mask]
        M_r_sat = compute_marked_correlation(pos_sat, L_sat, bins, L_box)
        all_M_r_sat.append(M_r_sat)
        tree_sat = cKDTree(pos_sat, boxsize=L_box)
        k = 5
        dists_k, _ = tree_sat.query(pos_sat, k=k+1)
        d_k = dists_k[:, k]
        rho_k = 1.0 / (d_k**3 + 1e-10)
        corr, p_val = spearmanr(L_sat, rho_k)
        spearman_results.append((corr, p_val))
    all_M_r_all = np.array(all_M_r_all)
    all_M_r_sat = np.array(all_M_r_sat)
    mean_M_r_all = np.nanmean(all_M_r_all, axis=0)
    mean_M_r_sat = np.nanmean(all_M_r_sat, axis=0)
    print("\n--- Marked Correlation Function M(r) ---")
    print("r_bin_center (Mpc/h) | M(r) All Galaxies | M(r) Satellites Only")
    print("-" * 65)
    for b in range(len(bin_centers)):
        print(str(round(bin_centers[b], 2)).ljust(20) + " | " + str(round(mean_M_r_all[b], 4)).ljust(17) + " | " + str(round(mean_M_r_sat[b], 4)))
    print("\n--- Spearman Rank Correlation (Luminosity vs Local Satellite Density) ---")
    print("Realization | Spearman r_s | p-value")
    print("-" * 45)
    for i, (corr, p_val) in enumerate(spearman_results):
        print(str(i).ljust(11) + " | " + str(round(corr, 4)).ljust(12) + " | " + str(p_val))
    np.save(os.path.join(output_dir, "M_r_bins.npy"), bins)
    np.save(os.path.join(output_dir, "M_r_bin_centers.npy"), bin_centers)
    np.save(os.path.join(output_dir, "all_M_r_all.npy"), all_M_r_all)
    np.save(os.path.join(output_dir, "all_M_r_sat.npy"), all_M_r_sat)
    np.save(os.path.join(output_dir, "spearman_results.npy"), np.array(spearman_results))
    print("\nMarked correlation results saved to disk.")