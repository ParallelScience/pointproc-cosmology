# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid

def compute_f(y, z):
    """
    Computes the fraction of an exponential profile (scale R_v) 
    that falls inside a sphere of radius r at distance x from the center.
    y = x / R_v
    z = r / R_v
    Valid for x >= r (i.e., y >= z).
    """
    def P(u, y, z):
        return -u**3 + (2*y - 3)*u**2 + (z**2 - y**2 + 4*y - 6)*(u + 1)
    u_plus = y + z
    u_minus = y - z
    A_plus = -np.exp(-u_plus) * P(u_plus, y, z)
    A_minus = -np.exp(-u_minus) * P(u_minus, y, z)
    return (A_plus - A_minus) / (8.0 * y)

if __name__ == '__main__':
    np.seterr(under='ignore')
    input_data_dir = "/home/node/work/projects/pointproc_cosmology/data/"
    output_data_dir = "data/"
    rho_crit = 2.77536627e11
    L_box = 500.0
    r_bins = np.logspace(np.log10(1.0), np.log10(30.0), 15)
    empirical_P0_all = []
    theoretical_P0_all = []
    print("--- Analytical VPF Modeling ---\n")
    print("Radius bins (Mpc/h): " + str(np.round(r_bins, 2)) + "\n")
    for i in range(10):
        halo_file = os.path.join(input_data_dir, "halo_catalog_" + str(i).zfill(2) + ".npy")
        gal_file = os.path.join(input_data_dir, "galaxy_catalog_" + str(i).zfill(2) + ".npy")
        halos = np.load(halo_file)
        gals = np.load(gal_file)
        gal_positions = gals[:, :3]
        np.random.seed(42 + i)
        random_centers = np.random.uniform(0, L_box, (10000, 3))
        tree = cKDTree(gal_positions, boxsize=L_box)
        distances, _ = tree.query(random_centers, k=1)
        emp_p0 = np.array([np.sum(distances > r) / 10000.0 for r in r_bins])
        empirical_P0_all.append(emp_p0)
        M_vir = halos[:, 6]
        R_vir = (3.0 * M_vir / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
        M_min = 3e11
        M_sat = 1e13
        mask_min = M_vir > M_min
        N_h_min = np.sum(mask_min)
        mask_sat = M_vir > M_sat
        M_sat_halos = M_vir[mask_sat]
        R_sat_halos = R_vir[mask_sat]
        N_sat_mean = M_sat_halos / M_sat
        V_r = (4.0 * np.pi / 3.0) * r_bins**3
        theo_p0 = np.zeros_like(r_bins)
        for j, r in enumerate(r_bins):
            integral_sum = 0.0
            if len(M_sat_halos) > 0:
                N_x = 200
                x = r + np.linspace(0, 15, N_x)[None, :] * R_sat_halos[:, None]
                y = x / R_sat_halos[:, None]
                z = r / R_sat_halos[:, None]
                f_val = compute_f(y, z)
                f_val = np.clip(f_val, 0, None)
                mu = N_sat_mean[:, None] * f_val
                integrand = 4.0 * np.pi * x**2 * (1.0 - np.exp(-mu))
                integral_vals = trapezoid(integrand, x=x, axis=1)
                integral_sum = np.sum(integral_vals)
            ln_P0 = - (N_h_min * V_r[j] + integral_sum) / (L_box**3)
            theo_p0[j] = np.exp(ln_P0)
        theoretical_P0_all.append(theo_p0)
        print("Realization " + str(i).zfill(2) + ":")
        print("  Empirical P0:   " + str(np.round(emp_p0, 4)))
        print("  Theoretical P0: " + str(np.round(theo_p0, 4)))
        print("  Max Diff:       " + str(np.round(np.max(np.abs(emp_p0 - theo_p0)), 4)) + "\n")
    empirical_P0_all = np.array(empirical_P0_all)
    theoretical_P0_all = np.array(theoretical_P0_all)
    output_file = os.path.join(output_data_dir, "vpf_results.npz")
    np.savez(output_file, r_bins=r_bins, empirical_P0=empirical_P0_all, theoretical_P0=theoretical_P0_all)
    print("Results saved to " + output_file + "\n")
    mean_emp = np.mean(empirical_P0_all, axis=0)
    mean_theo = np.mean(theoretical_P0_all, axis=0)
    std_emp = np.std(empirical_P0_all, axis=0)
    print("--- Summary across 10 realizations ---")
    print("r (Mpc/h) | Mean Emp P0 | Std Emp P0 | Mean Theo P0 | Diff")
    print("-" * 65)
    for j, r in enumerate(r_bins):
        diff = mean_emp[j] - mean_theo[j]
        print(str(np.round(r, 2)).rjust(9) + " | " + str(np.round(mean_emp[j], 4)).rjust(11) + " | " + str(np.round(std_emp[j], 4)).rjust(10) + " | " + str(np.round(mean_theo[j], 4)).rjust(12) + " | " + str(np.round(diff, 4)).rjust(8))