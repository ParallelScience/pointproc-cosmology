# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

matplotlib.rcParams['text.usetex'] = False

def exponential_profile(x, A, alpha):
    return A * np.exp(-x / alpha)

def sci_fmt(val):
    if np.isnan(val):
        return "nan"
    if np.isinf(val):
        return "inf" if val > 0 else "-inf"
    if val == 0:
        return "0.00e+00"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10**exponent)
    return str(round(mantissa, 2)) + "e" + str(exponent)

def main():
    data_dir = "/home/node/work/projects/pointproc_cosmology/data"
    output_dir = "data"
    print("Starting Radial Profile Diagnostic and Statistical Aggregation...\n")
    corr_file = os.path.join(output_dir, "large_scale_bias_correlations.npz")
    if not os.path.exists(corr_file):
        print("Error: large_scale_bias_correlations.npz not found.")
        sys.exit(1)
    corr_data = np.load(corr_file)
    mass_bins = corr_data['mass_bins']
    x_bins = np.linspace(0, 5, 21)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    V_x_shell = (4 * np.pi / 3) * (x_bins[1:]**3 - x_bins[:-1]**3)
    profiles = {k: [] for k in range(3)}
    for i in range(10):
        gal_file = os.path.join(data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        assoc_file = os.path.join(output_dir, "galaxy_halo_association_" + str(i).zfill(2) + ".npy")
        halo_file = os.path.join(data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        if not os.path.exists(gal_file) or not os.path.exists(assoc_file) or not os.path.exists(halo_file):
            continue
        gal_data = np.load(gal_file)
        assoc_data = np.load(assoc_file)
        halo_data = np.load(halo_file)
        is_sat = ~gal_data[:, 8].astype(bool)
        valid_sat = ~np.isnan(assoc_data[is_sat, 0])
        sat_pos = gal_data[is_sat, 0:3][valid_sat]
        sat_halo_pos = assoc_data[is_sat, 1:4][valid_sat]
        sat_R_vir = assoc_data[is_sat, 4][valid_sat]
        sat_halo_idx = assoc_data[is_sat, 0][valid_sat].astype(int)
        halo_M = halo_data[:, 6]
        sat_halo_M = halo_M[sat_halo_idx]
        dx = sat_pos - sat_halo_pos
        dx = dx - 500.0 * np.round(dx / 500.0)
        r = np.linalg.norm(dx, axis=1)
        x = r / sat_R_vir
        for k in range(3):
            mask_h = (halo_M >= mass_bins[k]) & (halo_M < mass_bins[k+1])
            N_h = np.sum(mask_h)
            mask_s = (sat_halo_M >= mass_bins[k]) & (sat_halo_M < mass_bins[k+1])
            x_s = x[mask_s]
            counts, _ = np.histogram(x_s, bins=x_bins)
            if N_h > 0:
                rho_x = counts / (N_h * V_x_shell)
            else:
                rho_x = np.zeros_like(x_centers)
            profiles[k].append(rho_x)
    results = {}
    for k in range(3):
        prof_k = np.array(profiles[k])
        mean_prof = np.mean(prof_k, axis=0)
        std_prof = np.std(prof_k, axis=0)
        try:
            popt, pcov = curve_fit(exponential_profile, x_centers, mean_prof, p0=[max(mean_prof[0], 1e-5), 1.0], bounds=(0, np.inf))
            A_fit, alpha_fit = popt
        except:
            A_fit, alpha_fit = np.nan, np.nan
        n_boot = 1000
        alpha_boot = []
        np.random.seed(42 + k)
        for _ in range(n_boot):
            idx = np.random.choice(len(prof_k), len(prof_k), replace=True)
            boot_prof = np.mean(prof_k[idx], axis=0)
            try:
                popt_b, _ = curve_fit(exponential_profile, x_centers, boot_prof, p0=[max(boot_prof[0], 1e-5), 1.0], bounds=(0, np.inf))
                alpha_boot.append(popt_b[1])
            except:
                pass
        alpha_err = np.std(alpha_boot) if len(alpha_boot) > 0 else np.nan
        results[k] = {'mean_prof': mean_prof, 'std_prof': std_prof, 'A': A_fit, 'alpha': alpha_fit, 'alpha_err': alpha_err}
    print("==================================================")
    print("STATISTICAL AGGREGATION & DIAGNOSTIC SUMMARY")
    print("==================================================")
    print("\n--- 1. Radial Profile Fits (rho(x) ~ exp(-x / alpha), x = r/R_vir) ---")
    for k in range(3):
        print("  Mass Bin " + str(k+1) + " [" + sci_fmt(mass_bins[k]) + " - " + sci_fmt(mass_bins[k+1]) + " M_sun/h]:")
        print("    r_scale / R_vir (alpha) = " + str(round(results[k]['alpha'], 4)) + " +/- " + str(round(results[k]['alpha_err'], 4)) + " (Theory: 1.0)")
    bias_file = os.path.join(output_dir, "bias_estimates.npy")
    if os.path.exists(bias_file):
        bias_data = np.load(bias_file)
        mean_bias = np.nanmean(bias_data, axis=0)
        std_bias = np.nanstd(bias_data, axis=0)
        print("\n--- 2. Large-Scale Halo Bias b(M) (r > 20 Mpc/h) ---")
        for k in range(3):
            print("  Mass Bin " + str(k+1) + ": b = " + str(round(mean_bias[k], 4)) + " +/- " + str(round(std_bias[k], 4)))
    pk_file = os.path.join(output_dir, "power_spectra.npz")
    if os.path.exists(pk_file):
        pk_data = np.load(pk_file)
        b_eff = pk_data['b_eff']
        P_shot = pk_data['P_shot_gal']
        print("\n--- 3. Power Spectrum Validation ---")
        print("  Effective Halo Bias (k < 0.05 h/Mpc): " + str(round(float(b_eff), 4)))
        print("  Mean Galaxy Shot Noise: " + sci_fmt(float(P_shot)) + " (Mpc/h)^3")
    mc_file = os.path.join(output_dir, "marked_correlation_data.npz")
    if os.path.exists(mc_file):
        mc_data = np.load(mc_file)
        r_mc = mc_data['r_centers']
        M_orig = mc_data['M_r_orig_mean']
        M_shuf = mc_data['M_r_shuf_mean']
        mask_1h = r_mc < 5.0
        mean_diff = np.nanmean(M_orig[mask_1h] - M_shuf[mask_1h])
        print("\n--- 4. Marked Correlation (1-Halo Regime, r < 5 Mpc/h) ---")
        print("  Mean difference (Original - Shuffled): " + str(round(mean_diff, 4)))
    pcf_file = os.path.join(output_dir, "2pcf_decomposition_data.npz")
    if os.path.exists(pcf_file):
        pcf_data = np.load(pcf_file)
        r_pcf = pcf_data['r_centers']
        xi_tot = pcf_data['tot']
        xi_1h_cs = pcf_data['1h_cs']
        xi_1h_ss = pcf_data['1h_ss']
        print("\n--- 5. 2PCF Decomposition Highlights ---")
        idx_1 = np.argmin(np.abs(r_pcf - 1.0))
        idx_5 = np.argmin(np.abs(r_pcf - 5.0))
        print("  At r ~ " + str(round(r_pcf[idx_1], 2)) + " Mpc/h:")
        print("    Total xi: " + str(round(xi_tot[idx_1], 2)))
        print("    1h_cs fraction: " + str(round(xi_1h_cs[idx_1] / xi_tot[idx_1], 4)))
        print("    1h_ss fraction: " + str(round(xi_1h_ss[idx_1] / xi_tot[idx_1], 4)))
        print("  At r ~ " + str(round(r_pcf[idx_5], 2)) + " Mpc/h:")
        print("    Total xi: " + str(round(xi_tot[idx_5], 2)))
        print("    1h_cs fraction: " + str(round(xi_1h_cs[idx_5] / xi_tot[idx_5], 4)))
        print("    1h_ss fraction: " + str(round(xi_1h_ss[idx_5] / xi_tot[idx_5], 4)))
    print("==================================================\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for k in range(3):
        ax = axes[k]
        ax.errorbar(x_centers, results[k]['mean_prof'], yerr=results[k]['std_prof']/np.sqrt(10), fmt='o', color='blue', label='Empirical Mean', markersize=5)
        x_plot = np.linspace(0, 5, 100)
        if not np.isnan(results[k]['alpha']):
            ax.plot(x_plot, exponential_profile(x_plot, results[k]['A'], results[k]['alpha']), 'r-', lw=2, label="Fit: alpha=" + str(round(results[k]['alpha'], 3)) + " +/- " + str(round(results[k]['alpha_err'], 3)))
            ax.plot(x_plot, exponential_profile(x_plot, results[k]['A'], 1.0), 'k--', lw=2, label='Theory (alpha=1.0)')
        ax.set_yscale('log')
        ax.set_xlabel('r / R_vir', fontsize=12)
        if k == 0:
            ax.set_ylabel('Density rho(r/R_vir)', fontsize=12)
        ax.set_title('Mass Bin ' + str(k+1) + ' [' + sci_fmt(mass_bins[k]) + ' - ' + sci_fmt(mass_bins[k+1]) + ']', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir, "radial_profiles_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)
    out_npz = os.path.join(output_dir, "radial_profile_data.npz")
    np.savez(out_npz, x_centers=x_centers, x_bins=x_bins, prof_0=results[0]['mean_prof'], err_0=results[0]['std_prof'], alpha_0=results[0]['alpha'], prof_1=results[1]['mean_prof'], err_1=results[1]['std_prof'], alpha_1=results[1]['alpha'], prof_2=results[2]['mean_prof'], err_2=results[2]['std_prof'], alpha_2=results[2]['alpha'])
    print("Saved radial profile data to " + out_npz)

if __name__ == '__main__':
    main()