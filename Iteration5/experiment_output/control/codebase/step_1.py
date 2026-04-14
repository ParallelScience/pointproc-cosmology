# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.spatial import cKDTree

def main():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    Omega_m = 0.311
    rho_crit = 2.77536627e11
    rho_m = Omega_m * rho_crit
    Delta = 200.0
    total_satellites = 0
    total_unmatched_satellites = 0
    total_halos = 0
    print("Starting Data Pre-processing and Halo-Galaxy Association...\n")
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(halo_file):
            print("Warning: Files for realization " + str(i).zfill(2) + " not found. Skipping.")
            continue
        gal_data = np.load(gal_file)
        halo_data = np.load(halo_file)
        gal_pos = gal_data[:, 0:3]
        gal_ln_M = gal_data[:, 7]
        is_central = gal_data[:, 8].astype(bool)
        is_sat = ~is_central
        halo_pos = halo_data[:, 0:3]
        halo_M = halo_data[:, 6]
        halo_ln_M = np.log(halo_M)
        R_vir = (halo_M / ((4.0 / 3.0) * np.pi * Delta * rho_m))**(1.0 / 3.0)
        tree = cKDTree(halo_pos, boxsize=500.0)
        distances, indices = tree.query(gal_pos, k=50)
        diff = np.abs(halo_ln_M[indices] - gal_ln_M[:, None])
        best_match = np.argmin(diff, axis=1)
        matched_halo_idx = indices[np.arange(len(gal_data)), best_match]
        min_diff = diff[np.arange(len(gal_data)), best_match]
        matched_mask = min_diff < 1e-3
        sat_matched_mask = matched_mask[is_sat]
        num_sat = np.sum(is_sat)
        num_sat_matched = np.sum(sat_matched_mask)
        num_sat_unmatched = num_sat - num_sat_matched
        total_satellites += num_sat
        total_unmatched_satellites += num_sat_unmatched
        total_halos += len(halo_data)
        mean_sat_per_halo = num_sat / len(halo_data)
        assoc_data = np.zeros((len(gal_data), 5))
        assoc_data[:, 0] = matched_halo_idx
        assoc_data[:, 1:4] = halo_pos[matched_halo_idx]
        assoc_data[:, 4] = R_vir[matched_halo_idx]
        assoc_data[~matched_mask, :] = np.nan
        out_file = os.path.join(output_dir, "galaxy_halo_association_" + str(i).zfill(2) + ".npy")
        np.save(out_file, assoc_data)
        print("Realization " + str(i).zfill(2) + ":")
        print("  Total galaxies: " + str(len(gal_data)))
        print("  Satellites: " + str(num_sat))
        print("  Satellites matched: " + str(num_sat_matched))
        print("  Satellites unmatched: " + str(num_sat_unmatched))
        print("  Mean satellites per halo: " + str(round(mean_sat_per_halo, 4)))
        print("  Saved association data to " + out_file + "\n")
    print("Overall Summary:")
    print("  Total satellites across all realizations: " + str(total_satellites))
    print("  Total matched satellites: " + str(total_satellites - total_unmatched_satellites))
    fraction_unmatched = 0.0
    if total_satellites > 0:
        fraction_unmatched = total_unmatched_satellites / total_satellites
    print("  Fraction unmatched: " + str(round(fraction_unmatched, 6)))
    global_mean = 0.0
    if total_halos > 0:
        global_mean = total_satellites / total_halos
    print("  Global mean satellites per halo: " + str(round(global_mean, 4)))

if __name__ == '__main__':
    main()