# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
import os

def main():
    data_dir = "data/"
    Omega_m = 0.311
    x = Omega_m - 1.0
    Delta_c = 18 * np.pi**2 + 82 * x - 39 * x**2
    rho_c = 2.7754e11
    mass_bins = [13.0, 13.5, 14.0, 14.5, 15.5]
    print("Starting spatial matching of galaxies to halos...\n")
    for i in range(10):
        gal_path = "/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_" + str(i).zfill(2) + ".npy"
        halo_path = "/home/node/work/projects/pointproc_cosmology/data/halo_catalog_" + str(i).zfill(2) + ".npy"
        galaxies = np.load(gal_path)
        halos = np.load(halo_path)
        gal_pos = galaxies[:, :3]
        gal_ln_M_h = galaxies[:, 7]
        is_central = galaxies[:, 8].astype(bool)
        halo_pos = halos[:, :3]
        halo_M_vir = halos[:, 6]
        halo_ln_M = np.log(halo_M_vir)
        R_vir = (halo_M_vir / ((4.0 / 3.0) * np.pi * Delta_c * rho_c))**(1.0 / 3.0)
        tree = cKDTree(halo_pos, boxsize=500.0)
        distances, indices = tree.query(gal_pos, k=50)
        k1_mass_diff = np.abs(halo_ln_M[indices[:, 0]] - gal_ln_M_h)
        k1_correct = k1_mass_diff < 1e-3
        n_k1_correct = np.sum(k1_correct)
        final_matched_idx = np.zeros(len(galaxies), dtype=int)
        final_matched_idx[k1_correct] = indices[k1_correct, 0]
        incorrect_indices = np.where(~k1_correct)[0]
        for j in incorrect_indices:
            mass_diffs = np.abs(halo_ln_M[indices[j]] - gal_ln_M_h[j])
            valid = np.where(mass_diffs < 1e-3)[0]
            if len(valid) > 0:
                final_matched_idx[j] = indices[j, valid[0]]
            else:
                final_matched_idx[j] = np.argmin(np.abs(halo_ln_M - gal_ln_M_h[j]))
        n_galaxies = len(galaxies)
        n_satellites = np.sum(~is_central)
        sat_fraction = n_satellites / n_galaxies
        unique_hosts = np.unique(final_matched_idx)
        log10_M_unique = np.log10(halo_M_vir[unique_hosts])
        bin_means = []
        bin_counts = []
        for b in range(len(mass_bins) - 1):
            mask = (log10_M_unique >= mass_bins[b]) & (log10_M_unique < mass_bins[b + 1])
            if np.any(mask):
                bin_means.append(np.mean(halo_M_vir[unique_hosts][mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_means.append(0.0)
                bin_counts.append(0)
        print("Realization " + str(i).zfill(2) + ":")
        print("  Total galaxies: " + str(n_galaxies))
        print("  Matched correctly by k=1 KD-Tree: " + str(n_k1_correct) + " (" + str(n_k1_correct / n_galaxies * 100) + "%)")
        print("  Satellite fraction: " + str(sat_fraction * 100) + "%")
        print("  Mean halo mass per bin (for matched host halos with M >= 10^13):")
        for b in range(len(mass_bins) - 1):
            print("    Bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b + 1]) + "): " + str(bin_counts[b]) + " halos, Mean Mass = " + str(bin_means[b]) + " M_sun/h")
        print("-" * 40)
        out_path = os.path.join(data_dir, "matched_catalog_" + str(i).zfill(2) + ".npz")
        np.savez(out_path, galaxies=galaxies, halos=halos, halo_R_vir=R_vir, matched_halo_idx=final_matched_idx)
    print("Matching complete. All matched datasets saved to data/ directory.")

if __name__ == '__main__':
    main()