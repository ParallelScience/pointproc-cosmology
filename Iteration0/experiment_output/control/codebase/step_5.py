# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import warnings

def compute_marked_correlation_multiple(pos, marks_dict, r_bins, boxsize=500.0):
    tree = cKDTree(pos, boxsize=boxsize)
    pairs_list = list(tree.query_pairs(r=r_bins[-1]))
    res = {k: np.full(len(r_bins)-1, np.nan) for k in marks_dict.keys()}
    counts = np.zeros(len(r_bins)-1)
    if len(pairs_list) == 0:
        return res, counts
    pairs = np.array(list(pairs_list))
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    dx = pos[i_idx] - pos[j_idx]
    dx = dx - boxsize * np.round(dx / boxsize)
    d = np.linalg.norm(dx, axis=1)
    bin_indices = np.digitize(d, r_bins) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(r_bins) - 1)
    bin_indices = bin_indices[valid]
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    mean_m_sq_dict = {k: np.mean(v)**2 for k, v in marks_dict.items()}
    for b in range(len(r_bins) - 1):
        mask = (bin_indices == b)
        counts[b] = np.sum(mask)
        if counts[b] > 0:
            for k, marks in marks_dict.items():
                m_i = marks[i_idx[mask]]
                m_j = marks[j_idx[mask]]
                m_ij = m_i * m_j
                res[k][b] = np.mean(m_ij) / mean_m_sq_dict[k]
    return res, counts

def main():
    data_dir = "data/"
    mass_bins = [13.0, 13.5, 14.0, 14.5, 15.5]
    n_mass_bins = len(mass_bins) - 1
    r_bins = np.linspace(0, 5, 26)
    r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    n_bins = len(r_bin_centers)
    all_M_r_base_offset = np.zeros((10, n_bins))
    all_M_r_base_L = np.zeros((10, n_bins))
    all_M_r_bins_offset = np.zeros((10, n_mass_bins, n_bins))
    all_M_r_bins_L = np.zeros((10, n_mass_bins, n_bins))
    all_counts_base = np.zeros((10, n_bins))
    all_counts_bins = np.zeros((10, n_mass_bins, n_bins))
    print("Computing marked correlation function M(r) for satellites...\n")
    for i in range(10):
        gal_path = os.path.join(data_dir, "matched_catalog_" + str(i).zfill(2) + ".npz")
        data = np.load(gal_path)
        galaxies = data['galaxies']
        halos = data['halos']
        matched_halo_idx = data['matched_halo_idx']
        halo_M_vir = halos[:, 6]
        halo_logM = np.log10(halo_M_vir)
        is_sat = galaxies[:, 8] == 0
        is_cen = galaxies[:, 8] == 1
        cen_halo_idx = matched_halo_idx[is_cen]
        cen_ln_L = np.full(len(halos), np.nan)
        cen_ln_L[cen_halo_idx] = galaxies[is_cen, 6]
        sat_halo_idx = matched_halo_idx[is_sat]
        sat_pos = galaxies[is_sat, :3]
        sat_ln_L = galaxies[is_sat, 6]
        sat_cen_ln_L = cen_ln_L[sat_halo_idx]
        valid_sat = ~np.isnan(sat_cen_ln_L)
        sat_pos = sat_pos[valid_sat]
        sat_halo_idx = sat_halo_idx[valid_sat]
        sat_ln_L = sat_ln_L[valid_sat]
        sat_cen_ln_L = sat_cen_ln_L[valid_sat]
        marks_dict = {'offset': np.exp(sat_ln_L - sat_cen_ln_L), 'L_sat': np.exp(sat_ln_L)}
        sat_halo_logM = halo_logM[sat_halo_idx]
        baseline_mask = sat_halo_logM >= 13.0
        pos_base = sat_pos[baseline_mask]
        marks_base_dict = {k: v[baseline_mask] for k, v in marks_dict.items()}
        if len(pos_base) > 1:
            res_base, counts_base = compute_marked_correlation_multiple(pos_base, marks_base_dict, r_bins)
            all_M_r_base_offset[i] = res_base['offset']
            all_M_r_base_L[i] = res_base['L_sat']
            all_counts_base[i] = counts_base
        for b in range(n_mass_bins):
            mask_b = (sat_halo_logM >= mass_bins[b]) & (sat_halo_logM < mass_bins[b+1])
            pos_b = sat_pos[mask_b]
            marks_b_dict = {k: v[mask_b] for k, v in marks_dict.items()}
            if len(pos_b) > 1:
                res_b, counts_b = compute_marked_correlation_multiple(pos_b, marks_b_dict, r_bins)
                all_M_r_bins_offset[i, b] = res_b['offset']
                all_M_r_bins_L[i, b] = res_b['L_sat']
                all_counts_bins[i, b] = counts_b
        print("Realization " + str(i).zfill(2) + " completed.")
    out_path = os.path.join(data_dir, "marked_correlation.npz")
    np.savez(out_path, r_bins=r_bins, r_bin_centers=r_bin_centers, mass_bins=mass_bins, M_r_base_offset=all_M_r_base_offset, M_r_base_L=all_M_r_base_L, M_r_bins_offset=all_M_r_bins_offset, M_r_bins_L=all_M_r_bins_L, counts_base=all_counts_base, counts_bins=all_counts_bins)
    print("\nSaved marked correlation results to " + out_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_M_r_base_offset = np.nanmean(all_M_r_base_offset, axis=0)
        mean_M_r_base_L = np.nanmean(all_M_r_base_L, axis=0)
    print("\nSummary of Marked Correlation M(r) for Baseline (all satellites in M >= 10^13):")
    print("r_centers (Mpc/h):      " + str(np.round(r_bin_centers[:10], 3)))
    print("M(r) [offset mark]:     " + str(np.round(mean_M_r_base_offset[:10], 4)))
    print("M(r) [L_sat mark]:      " + str(np.round(mean_M_r_base_L[:10], 4)))
    print("Pair counts (mean):     " + str(np.round(np.mean(all_counts_base, axis=0)[:10], 1)))
    print("\nSummary of Marked Correlation M(r) [offset mark] by Mass Bin:")
    for b in range(n_mass_bins):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_M_r_b = np.nanmean(all_M_r_bins_offset[:, b, :], axis=0)
        print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "):")
        print("    M(r):               " + str(np.round(mean_M_r_b[:10], 4)))

if __name__ == '__main__':
    main()