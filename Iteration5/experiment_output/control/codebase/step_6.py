# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import cKDTree
matplotlib.rcParams['text.usetex'] = False
def compute_decomposed_pairs(pos, is_central, halo_idx, r_bins, L_box):
    tree = cKDTree(pos, boxsize=L_box)
    r_max = r_bins[-1]
    pairs = np.array(list(tree.query_pairs(r_max)))
    counts = {k: np.zeros(len(r_bins)-1) for k in ['1h_cc', '1h_cs', '1h_ss', '2h_cc', '2h_cs', '2h_ss', 'tot']}
    if len(pairs) == 0:
        return counts
    i = pairs[:, 0]
    j = pairs[:, 1]
    dx = pos[i] - pos[j]
    dx = dx - L_box * np.round(dx / L_box)
    d = np.linalg.norm(dx, axis=1)
    bin_indices = np.digitize(d, r_bins) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < len(r_bins) - 1)
    i = i[valid_mask]
    j = j[valid_mask]
    bin_indices = bin_indices[valid_mask]
    c_i = is_central[i]
    c_j = is_central[j]
    h_i = halo_idx[i]
    h_j = halo_idx[j]
    same_halo = (h_i == h_j) & (~np.isnan(h_i)) & (~np.isnan(h_j))
    is_cc = c_i & c_j
    is_ss = (~c_i) & (~c_j)
    is_cs = (c_i & ~c_j) | (~c_i & c_j)
    mask_1h_cc = same_halo & is_cc
    mask_1h_cs = same_halo & is_cs
    mask_1h_ss = same_halo & is_ss
    mask_2h_cc = (~same_halo) & is_cc
    mask_2h_cs = (~same_halo) & is_cs
    mask_2h_ss = (~same_halo) & is_ss
    counts['1h_cc'] = np.bincount(bin_indices[mask_1h_cc], minlength=len(r_bins)-1)
    counts['1h_cs'] = np.bincount(bin_indices[mask_1h_cs], minlength=len(r_bins)-1)
    counts['1h_ss'] = np.bincount(bin_indices[mask_1h_ss], minlength=len(r_bins)-1)
    counts['2h_cc'] = np.bincount(bin_indices[mask_2h_cc], minlength=len(r_bins)-1)
    counts['2h_cs'] = np.bincount(bin_indices[mask_2h_cs], minlength=len(r_bins)-1)
    counts['2h_ss'] = np.bincount(bin_indices[mask_2h_ss], minlength=len(r_bins)-1)
    counts['tot'] = np.bincount(bin_indices, minlength=len(r_bins)-1)
    return counts
def main():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    L_box = 500.0
    print("Starting 2PCF Decomposition...\n")
    r_bins = np.logspace(-1, 1.7, 25)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    V_shell = (4 * np.pi / 3) * (r_bins[1:]**3 - r_bins[:-1]**3)
    V_box = L_box**3
    xi_components = {'1h_cs': [], '1h_ss': [], '2h_cc': [], '2h_cs': [], '2h_ss': [], 'tot': []}
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        assoc_file = os.path.join(output_dir, "galaxy_halo_association_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(assoc_file):
            print("Warning: Files for realization " + str(i).zfill(2) + " not found. Skipping.")
            continue
        gal_data = np.load(gal_file)
        assoc_data = np.load(assoc_file)
        pos = gal_data[:, 0:3]
        is_central = gal_data[:, 8].astype(bool)
        halo_idx = assoc_data[:, 0]
        counts = compute_decomposed_pairs(pos, is_central, halo_idx, r_bins, L_box)
        if np.sum(counts['1h_cc']) > 0:
            print("Warning: Found " + str(int(np.sum(counts['1h_cc']))) + " 1-halo CC pairs in realization " + str(i) + "!")
        N_c = np.sum(is_central)
        N_s = np.sum(~is_central)
        N_tot = len(is_central)
        RR_cc = 0.5 * N_c * (N_c - 1) * V_shell / V_box
        RR_cs = N_c * N_s * V_shell / V_box
        RR_ss = 0.5 * N_s * (N_s - 1) * V_shell / V_box
        RR_tot = 0.5 * N_tot * (N_tot - 1) * V_shell / V_box
        xi_1h_cs = counts['1h_cs'] / RR_tot
        xi_1h_ss = counts['1h_ss'] / RR_tot
        xi_2h_cc = (counts['2h_cc'] - RR_cc) / RR_tot
        xi_2h_cs = (counts['2h_cs'] - RR_cs) / RR_tot
        xi_2h_ss = (counts['2h_ss'] - RR_ss) / RR_tot
        xi_tot = (counts['tot'] - RR_tot) / RR_tot
        xi_components['1h_cs'].append(xi_1h_cs)
        xi_components['1h_ss'].append(xi_1h_ss)
        xi_components['2h_cc'].append(xi_2h_cc)
        xi_components['2h_cs'].append(xi_2h_cs)
        xi_components['2h_ss'].append(xi_2h_ss)
        xi_components['tot'].append(xi_tot)
        print("Processed realization " + str(i).zfill(2))
    if len(xi_components['tot']) == 0:
        print("Error: No data processed.")
        sys.exit(1)
    mean_xi = {k: np.mean(xi_components[k], axis=0) for k in xi_components}
    print("\n2PCF Decomposition Results (Mean across realizations):")
    print("  r [Mpc/h] |     Total |    1h_cs |    1h_ss |    2h_cc |    2h_cs |    2h_ss")
    print("-" * 85)
    for i in range(len(r_centers)):
        print("  " + "{:9.4f}".format(r_centers[i]) + " | " + "{:9.2f}".format(mean_xi['tot'][i]) + " | " + "{:8.2f}".format(mean_xi['1h_cs'][i]) + " | " + "{:8.2f}".format(mean_xi['1h_ss'][i]) + " | " + "{:8.2f}".format(mean_xi['2h_cc'][i]) + " | " + "{:8.2f}".format(mean_xi['2h_cs'][i]) + " | " + "{:8.2f}".format(mean_xi['2h_ss'][i]))
    plt.figure(figsize=(10, 7))
    plt.plot(r_centers, mean_xi['tot'], label='Total xi(r)', color='black', lw=3)
    plt.plot(r_centers, mean_xi['1h_cs'], label='1-halo Central-Satellite', color='blue', marker='o', lw=2)
    plt.plot(r_centers, mean_xi['1h_ss'], label='1-halo Satellite-Satellite', color='cyan', marker='s', lw=2)
    plt.plot(r_centers, mean_xi['2h_cc'], label='2-halo Central-Central', color='red', marker='^', lw=2)
    plt.plot(r_centers, mean_xi['2h_cs'], label='2-halo Central-Satellite', color='orange', marker='v', lw=2)
    plt.plot(r_centers, mean_xi['2h_ss'], label='2-halo Satellite-Satellite', color='magenta', marker='d', lw=2)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-2)
    plt.xlabel('r [Mpc/h]', fontsize=12)
    plt.ylabel('xi(r) Contributions', fontsize=12)
    plt.title('Decomposition of the Two-Point Correlation Function', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, "2pcf_decomposition_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("\nPlot saved to " + plot_filename)
    out_npz = os.path.join(output_dir, "2pcf_decomposition_data.npz")
    np.savez(out_npz, r_centers=r_centers, r_bins=r_bins, **mean_xi)
    print("Saved 2PCF decomposition data to " + out_npz)
if __name__ == '__main__':
    main()