# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.stats import ks_2samp

def sci_fmt(val):
    if np.isnan(val):
        return "nan"
    if val == 0:
        return "0.00e+00"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10**exponent)
    return str(round(mantissa, 4)) + "e" + str(exponent)

def main():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    L_box = 500.0
    print("Starting Assembly Bias Isolation via Shuffling...\n")
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        assoc_file = os.path.join(output_dir, "galaxy_halo_association_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(assoc_file):
            print("Warning: Files for realization " + str(i).zfill(2) + " not found. Skipping.")
            continue
        gal_data = np.load(gal_file)
        assoc_data = np.load(assoc_file)
        is_central = gal_data[:, 8].astype(bool)
        is_sat = ~is_central
        gal_pos = gal_data[:, 0:3]
        halo_pos = assoc_data[:, 1:4]
        sat_pos = gal_pos[is_sat]
        sat_halo_pos = halo_pos[is_sat]
        dx = sat_pos - sat_halo_pos
        dx = dx - L_box * np.round(dx / L_box)
        r_orig = np.linalg.norm(dx, axis=1)
        np.random.seed(42 + i)
        u = np.random.uniform(-1, 1, size=len(r_orig))
        phi = np.random.uniform(0, 2 * np.pi, size=len(r_orig))
        sin_theta = np.sqrt(1 - u**2)
        dx_new = r_orig * sin_theta * np.cos(phi)
        dy_new = r_orig * sin_theta * np.sin(phi)
        dz_new = r_orig * u
        new_sat_pos = sat_halo_pos + np.column_stack((dx_new, dy_new, dz_new))
        new_sat_pos = new_sat_pos % L_box
        dx_check = new_sat_pos - sat_halo_pos
        dx_check = dx_check - L_box * np.round(dx_check / L_box)
        r_new = np.linalg.norm(dx_check, axis=1)
        ks_stat, p_value = ks_2samp(r_orig, r_new)
        shuffled_gal_data = gal_data.copy()
        shuffled_gal_data[is_sat, 0:3] = new_sat_pos
        out_file = os.path.join(output_dir, "shuffled_galaxy_catalog_" + str(i).zfill(2) + ".npy")
        np.save(out_file, shuffled_gal_data)
        print("Realization " + str(i).zfill(2) + ":")
        print("  Satellites shuffled: " + str(np.sum(is_sat)))
        print("  KS-test statistic: " + sci_fmt(ks_stat) + ", p-value: " + str(round(p_value, 4)))
        print("  Max absolute difference in radial distance: " + sci_fmt(np.max(np.abs(r_orig - r_new))) + " Mpc/h")
        print("  Saved shuffled catalog to " + out_file + "\n")

if __name__ == '__main__':
    main()