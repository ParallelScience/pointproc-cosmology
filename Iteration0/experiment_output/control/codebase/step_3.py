# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np

def main():
    data_dir = "data/"
    mass_bins = [13.0, 13.5, 14.0, 14.5, 15.5]
    n_mass_bins = len(mass_bins) - 1
    mle_lambda = np.zeros((10, n_mass_bins))
    mle_sigma = np.zeros((10, n_mass_bins))
    ll_exp = np.zeros((10, n_mass_bins))
    ll_gauss = np.zeros((10, n_mass_bins))
    ll_null = np.zeros((10, n_mass_bins))
    aic_exp = np.zeros((10, n_mass_bins))
    aic_gauss = np.zeros((10, n_mass_bins))
    aic_null = np.zeros((10, n_mass_bins))
    print("Fitting models to satellite radial distributions and generating null catalogs...\n")
    for i in range(10):
        in_path = os.path.join(data_dir, "matched_catalog_" + str(i).zfill(2) + ".npz")
        data = np.load(in_path)
        galaxies = data['galaxies']
        halos = data['halos']
        halo_R_vir = data['halo_R_vir']
        matched_halo_idx = data['matched_halo_idx']
        is_sat = galaxies[:, 8] == 0
        sat_galaxies = galaxies[is_sat]
        sat_halo_idx = matched_halo_idx[is_sat]
        sat_pos = sat_galaxies[:, :3]
        halo_pos = halos[sat_halo_idx, :3]
        dx = sat_pos - halo_pos
        dx = dx - 500.0 * np.round(dx / 500.0)
        r = np.linalg.norm(dx, axis=1)
        R_vir_sat = halo_R_vir[sat_halo_idx]
        x = r / R_vir_sat
        halo_M_vir = halos[:, 6]
        halo_logM = np.log10(halo_M_vir)
        sat_halo_logM = halo_logM[sat_halo_idx]
        if i == 0:
            print("Realization " + str(i).zfill(2) + " Model Fits:")
        for b in range(n_mass_bins):
            mask = (sat_halo_logM >= mass_bins[b]) & (sat_halo_logM < mass_bins[b+1])
            x_b = x[mask]
            x_b = np.maximum(x_b, 1e-6)
            N = len(x_b)
            if N > 0:
                lam = np.mean(x_b) / 3.0
                mle_lambda[i, b] = lam
                sig = np.sqrt(np.mean(x_b**2) / 3.0)
                mle_sigma[i, b] = sig
                ll_e = np.sum(2 * np.log(x_b) - x_b / lam - 3 * np.log(lam) - np.log(2))
                ll_exp[i, b] = ll_e
                ll_g = np.sum(2 * np.log(x_b) - (x_b**2) / (2 * sig**2) - 3 * np.log(sig) + np.log(np.sqrt(2/np.pi)))
                ll_gauss[i, b] = ll_g
                x_max = np.max(x_b)
                ll_n = np.sum(np.log(3) + 2 * np.log(x_b) - 3 * np.log(x_max))
                ll_null[i, b] = ll_n
                aic_exp[i, b] = 2 * 1 - 2 * ll_e
                aic_gauss[i, b] = 2 * 1 - 2 * ll_g
                aic_null[i, b] = 2 * 1 - 2 * ll_n
                if i == 0:
                    print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "): N=" + str(N))
                    print("    Exp fit: lambda/R_vir = " + str(np.round(lam, 4)) + ", AIC = " + str(np.round(aic_exp[i, b], 1)))
                    print("    Gau fit: sigma/R_vir  = " + str(np.round(sig, 4)) + ", AIC = " + str(np.round(aic_gauss[i, b], 1)))
                    print("    Nul fit: x_max        = " + str(np.round(x_max, 4)) + ", AIC = " + str(np.round(aic_null[i, b], 1)))
            else:
                if i == 0:
                    print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "): N=0")
        np.random.seed(42 + i)
        u = np.random.rand(len(sat_galaxies))
        r_new = R_vir_sat * u**(1.0/3.0)
        cos_theta = np.random.uniform(-1, 1, len(sat_galaxies))
        phi = np.random.uniform(0, 2*np.pi, len(sat_galaxies))
        sin_theta = np.sqrt(1 - cos_theta**2)
        dx_new = r_new * sin_theta * np.cos(phi)
        dy_new = r_new * sin_theta * np.sin(phi)
        dz_new = r_new * cos_theta
        new_sat_pos = halo_pos + np.column_stack((dx_new, dy_new, dz_new))
        new_sat_pos = new_sat_pos % 500.0
        null_galaxies = galaxies.copy()
        null_galaxies[is_sat, :3] = new_sat_pos
        out_path = os.path.join(data_dir, "null_catalog_" + str(i).zfill(2) + ".npy")
        np.save(out_path, null_galaxies)
    res_path = os.path.join(data_dir, "mle_fits.npz")
    np.savez(res_path, mass_bins=mass_bins, mle_lambda=mle_lambda, mle_sigma=mle_sigma, ll_exp=ll_exp, ll_gauss=ll_gauss, ll_null=ll_null, aic_exp=aic_exp, aic_gauss=aic_gauss, aic_null=aic_null)
    print("\nSaved MLE fits to data/mle_fits.npz")
    print("Saved null catalogs to data/null_catalog_XX.npy")
    print("\nMean AIC across 10 realizations:")
    for b in range(n_mass_bins):
        print("  Mass bin [" + str(mass_bins[b]) + ", " + str(mass_bins[b+1]) + "):")
        print("    Exponential: " + str(np.round(np.mean(aic_exp[:, b]), 1)))
        print("    Gaussian:    " + str(np.round(np.mean(aic_gauss[:, b]), 1)))
        print("    Uniform:     " + str(np.round(np.mean(aic_null[:, b]), 1)))
        print("    Mean lambda/R_vir: " + str(np.round(np.mean(mle_lambda[:, b]), 4)) + " +/- " + str(np.round(np.std(mle_lambda[:, b]), 4)))

if __name__ == '__main__':
    main()