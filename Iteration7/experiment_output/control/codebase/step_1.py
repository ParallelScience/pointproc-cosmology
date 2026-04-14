# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree

def load_and_process_data():
    Omega_m = 0.311
    rho_c = 2.77536627e11
    rho_m = Omega_m * rho_c
    Delta_m = 200.0
    L_box = 500.0
    master_satellites = []
    halo_counts = []
    for i in range(10):
        realization_str = str(i).zfill(2)
        gal_path = "/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_" + realization_str + ".npy"
        halo_path = "/home/node/work/projects/pointproc_cosmology/data/halo_catalog_" + realization_str + ".npy"
        if not os.path.exists(gal_path) or not os.path.exists(halo_path):
            print("Warning: Data for realization " + realization_str + " not found.")
            continue
        galaxies = np.load(gal_path)
        halos = np.load(halo_path)
        halo_x = halos[:, 0]
        halo_y = halos[:, 1]
        halo_z = halos[:, 2]
        halo_M = halos[:, 6]
        halo_Rvir = (3.0 * halo_M / (4.0 * np.pi * Delta_m * rho_m))**(1.0/3.0)
        is_central = galaxies[:, 8].astype(int)
        satellites = galaxies[is_central == 0]
        sat_x = satellites[:, 0]
        sat_y = satellites[:, 1]
        sat_z = satellites[:, 2]
        sat_lnL = satellites[:, 6]
        sat_lnMh = satellites[:, 7]
        halo_lnM = np.log(halo_M)
        tree = cKDTree(halo_lnM.reshape(-1, 1))
        distances, halo_indices = tree.query(sat_lnMh.reshape(-1, 1))
        dx = sat_x - halo_x[halo_indices]
        dy = sat_y - halo_y[halo_indices]
        dz = sat_z - halo_z[halo_indices]
        dx = dx - L_box * np.round(dx / L_box)
        dy = dy - L_box * np.round(dy / L_box)
        dz = dz - L_box * np.round(dz / L_box)
        r_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        L_sat = np.exp(sat_lnL)
        df_sats_realization = pd.DataFrame({'realization_id': np.full(len(satellites), i), 'halo_id': halo_indices, 'M_vir': halo_M[halo_indices], 'R_vir': halo_Rvir[halo_indices], 'r': r_dist, 'L': L_sat})
        master_satellites.append(df_sats_realization)
        unique_halos, counts = np.unique(halo_indices, return_counts=True)
        sat_counts = np.zeros(len(halos), dtype=int)
        sat_counts[unique_halos] = counts
        df_halos_realization = pd.DataFrame({'realization_id': np.full(len(halos), i), 'halo_id': np.arange(len(halos)), 'M_vir': halo_M, 'R_vir': halo_Rvir, 'N_sat': sat_counts})
        halo_counts.append(df_halos_realization)
    if len(master_satellites) == 0:
        print("Error: No data loaded.")
        return
    df_satellites = pd.concat(master_satellites, ignore_index=True)
    df_halos = pd.concat(halo_counts, ignore_index=True)
    df_halos_target = df_halos[df_halos['M_vir'] >= 1e13].copy()
    log_M = np.log10(df_halos_target['M_vir'])
    bins = np.linspace(13.0, log_M.max() + 0.1, 21)
    df_halos_target['mass_bin'] = pd.cut(log_M, bins=bins, include_lowest=True)
    binned_counts = df_halos_target.groupby('mass_bin', observed=False).agg(logM_center=('M_vir', lambda x: np.mean(np.log10(x)) if len(x) > 0 else np.nan), mean_M_vir=('M_vir', 'mean'), mean_R_vir=('R_vir', 'mean'), N_halos=('M_vir', 'count'), N_sats=('N_sat', 'sum')).reset_index()
    binned_counts = binned_counts.dropna(subset=['logM_center'])
    binned_counts['mass_bin'] = binned_counts['mass_bin'].astype(str)
    data_dir = "data/"
    sat_path = os.path.join(data_dir, "master_satellites.csv")
    halo_path = os.path.join(data_dir, "halo_satellite_counts.csv")
    binned_path = os.path.join(data_dir, "mass_binned_satellite_counts.csv")
    df_satellites.to_csv(sat_path, index=False)
    df_halos.to_csv(halo_path, index=False)
    binned_counts.to_csv(binned_path, index=False)
    print("Saved master satellites dataset to " + sat_path)
    print("Saved halo satellite counts to " + halo_path)
    print("Saved mass-binned satellite counts to " + binned_path)
    print("\n--- Summary Statistics ---")
    print("Total satellites processed: " + str(len(df_satellites)))
    print("Total halos processed: " + str(len(df_halos)))
    print("Total halos with M_vir >= 10^13: " + str(len(df_halos_target)))
    print("\nMass-Binned Satellite Counts (M_vir >= 10^13):")
    print(binned_counts.to_string(index=False))

if __name__ == '__main__':
    load_and_process_data()