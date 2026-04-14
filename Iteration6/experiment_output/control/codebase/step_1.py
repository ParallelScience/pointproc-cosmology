# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os
from scipy.spatial import cKDTree

def process_catalogs():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    all_satellites = []
    for i in range(10):
        gal_path = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halo_path = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_cat = np.load(gal_path)
        halo_cat = np.load(halo_path)
        is_central = gal_cat[:, 8].astype(int)
        sat_mask = (is_central == 0)
        sat_cat = gal_cat[sat_mask]
        sat_pos = sat_cat[:, 0:3]
        sat_ln_M_h = sat_cat[:, 7]
        sat_ln_L = sat_cat[:, 6]
        halo_pos = halo_cat[:, 0:3]
        halo_M_vir = halo_cat[:, 6]
        halo_ln_M = np.log(halo_M_vir)
        if halo_cat.shape[1] > 7:
            halo_R_vir = halo_cat[:, 7]
        else:
            Omega_m = 0.311
            rho_c = 2.775e11
            rho_m = Omega_m * rho_c
            halo_R_vir = (3.0 * halo_M_vir / (4.0 * np.pi * 200.0 * rho_m))**(1.0/3.0)
        tree = cKDTree(halo_pos, boxsize=500.0)
        k_neighbors = min(50, len(halo_pos))
        dists, idxs = tree.query(sat_pos, k=k_neighbors)
        matched_halo_idx = np.full(len(sat_pos), -1, dtype=int)
        for j in range(len(sat_pos)):
            s_mass = sat_ln_M_h[j]
            for k in range(k_neighbors):
                h_idx = idxs[j, k]
                if abs(halo_ln_M[h_idx] - s_mass) < 1e-4:
                    matched_halo_idx[j] = h_idx
                    break
            if matched_halo_idx[j] == -1:
                mass_matches = np.where(np.abs(halo_ln_M - s_mass) < 1e-4)[0]
                if len(mass_matches) > 0:
                    dx_fallback = np.abs(sat_pos[j, 0] - halo_pos[mass_matches, 0])
                    dy_fallback = np.abs(sat_pos[j, 1] - halo_pos[mass_matches, 1])
                    dz_fallback = np.abs(sat_pos[j, 2] - halo_pos[mass_matches, 2])
                    dx_fallback = np.minimum(dx_fallback, 500.0 - dx_fallback)
                    dy_fallback = np.minimum(dy_fallback, 500.0 - dy_fallback)
                    dz_fallback = np.minimum(dz_fallback, 500.0 - dz_fallback)
                    d_fallback = dx_fallback**2 + dy_fallback**2 + dz_fallback**2
                    best = np.argmin(d_fallback)
                    matched_halo_idx[j] = mass_matches[best]
        valid_match = matched_halo_idx != -1
        matched_sats = np.sum(valid_match)
        sat_pos_matched = sat_pos[valid_match]
        sat_ln_L_matched = sat_ln_L[valid_match]
        h_idx_matched = matched_halo_idx[valid_match]
        h_pos_matched = halo_pos[h_idx_matched]
        h_M_vir_matched = halo_M_vir[h_idx_matched]
        h_R_vir_matched = halo_R_vir[h_idx_matched]
        dx = sat_pos_matched[:, 0] - h_pos_matched[:, 0]
        dy = sat_pos_matched[:, 1] - h_pos_matched[:, 1]
        dz = sat_pos_matched[:, 2] - h_pos_matched[:, 2]
        dx = (dx + 250.0) % 500.0 - 250.0
        dy = (dy + 250.0) % 500.0 - 250.0
        dz = (dz + 250.0) % 500.0 - 250.0
        r_sat = np.sqrt(dx**2 + dy**2 + dz**2)
        qualifying_halo_mask = halo_M_vir > 1e13
        num_qualifying_halos = np.sum(qualifying_halo_mask)
        sat_filter = (h_M_vir_matched > 1e13) & (r_sat < 5.0)
        n_filtered = np.sum(sat_filter)
        realization_data = np.zeros(n_filtered, dtype=[('halo_mass', 'f8'), ('r_vir', 'f8'), ('r_sat', 'f8'), ('dx', 'f8'), ('dy', 'f8'), ('dz', 'f8'), ('ln_L', 'f8'), ('realization_id', 'i4')])
        realization_data['halo_mass'] = h_M_vir_matched[sat_filter]
        realization_data['r_vir'] = h_R_vir_matched[sat_filter]
        realization_data['r_sat'] = r_sat[sat_filter]
        realization_data['dx'] = dx[sat_filter]
        realization_data['dy'] = dy[sat_filter]
        realization_data['dz'] = dz[sat_filter]
        realization_data['ln_L'] = sat_ln_L_matched[sat_filter]
        realization_data['realization_id'] = i
        all_satellites.append(realization_data)
        print("Realization " + str(i) + ":")
        print("  Matched satellites: " + str(matched_sats) + " / " + str(len(sat_pos)))
        print("  Halos with M > 10^13 M_sun/h: " + str(num_qualifying_halos))
        sats_in_qual = np.sum(h_M_vir_matched > 1e13)
        mean_sats = sats_in_qual / num_qualifying_halos if num_qualifying_halos > 0 else 0
        print("  Mean satellites per qualifying halo: " + str(round(mean_sats, 4)))
        print("  Satellites in 1-halo regime (r < 5 Mpc/h): " + str(n_filtered))
        print("-" * 40)
    final_data = np.concatenate(all_satellites)
    output_path = os.path.join(output_dir, "processed_satellites.npy")
    np.save(output_path, final_data)
    print("Processed data saved to " + output_path)
    print("Total satellites across all realizations: " + str(len(final_data)))

if __name__ == '__main__':
    process_catalogs()