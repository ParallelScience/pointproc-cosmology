"""
Generate 10 synthetic galaxy catalogs using Neyman-Scott cluster process + HOD.
Standard approach for mock galaxy catalogs in cosmology.

Cosmology: Planck 2018 (LCDM)
Box: L = 500 Mpc/h, periodic
"""

import numpy as np
import os

np.random.seed(42)
rng = np.random.default_rng(42)

# ---------- cosmology ----------
h = 0.677
OMEGA_M = 0.311
L_BOX = 500.0  # Mpc/h
RHO_MEAN = 2.775e11 * OMEGA_M  # mean matter density M_sun h / Mpc^3

# ---------- HOD (SDSS-like) ----------
M_MIN = 3e11 / h    # M_sun/h
M_SAT = 1e13 / h
ALPHA_SAT = 1.0
L_STAR = 1e10 / h
SIG_LNL = 0.3

# ---------- halo mass function (simplified Schechter) ----------
# Number density of halos per log mass: dN/dlnM ∝ M^alpha * exp(-M/M_cut)
# For M_MIN to 1e15, typical number ~ 0.003 (Mpc/h)^-3
# => ~375 halos in 500 Mpc/h box at M_MIN

def sample_halo_masses(n_halos, rng):
    """Sample halo masses from a power-law + exponential mass function."""
    # Use inverse transform sampling with approximate CDF
    # dN/dlnM ∝ M^-0.1 * exp(-M / 5e14) for M > M_MIN
    # Approximate with Gamma distribution
    shape = 0.3
    scale = 5e13 / h
    masses = rng.gamma(shape, scale, n_halos) + M_MIN
    masses = masses[masses >= M_MIN]
    return masses


def assign_galaxies_to_halo(M, x, y, z, vx, vy, vz, rng):
    """Assign central + satellite galaxies to one halo."""
    gals = []
    
    # Central
    lnL = np.log(L_STAR * (M / 1e12) ** 1.0) + rng.normal(0, SIG_LNL)
    gals.append([x, y, z, vx, vy, vz, lnL, np.log(M), 1])
    
    # Satellites (if massive enough)
    if M > M_SAT:
        N_sat = min(rng.poisson(max((M / M_SAT) ** ALPHA_SAT, 0)), 30)
        R_vir = 0.4 * ((M / 1e12) ** (1.0/3.0))  # Mpc/h
        for _ in range(N_sat):
            r = R_vir * rng.exponential(0.5)
            theta = np.arccos(2 * rng.random() - 1)
            phi = 2 * np.pi * rng.random()
            dx = r * np.sin(theta) * np.cos(phi)
            dy = r * np.sin(theta) * np.sin(phi)
            dz = r * np.cos(theta)
            lnL_sat = lnL + rng.normal(-0.5, 0.4)
            gals.append([
                (x + dx) % L_BOX, (y + dy) % L_BOX, (z + dz) % L_BOX,
                vx + rng.normal(0, 60), vy + rng.normal(0, 60), vz + rng.normal(0, 60),
                lnL_sat, np.log(M), 0
            ])
    return gals


def compute_2pcf(positions, L=L_BOX, n_rand=20000, r_bins=None):
    """Landy-Szalay 2PCF with proper unordered pair normalization."""
    if r_bins is None:
        r_bins = np.array([1.0, 5.0, 10.0, 30.0, 100.0])  # Mpc/h
    nb = len(r_bins) - 1
    n_d = len(positions)
    
    rands = rng.uniform(0, L, (n_rand, 3))
    
    def pair_counts_unordered(pos1, pos2, r_bins):
        nb = len(r_bins) - 1
        c = np.zeros(nb)
        step = 500
        for i in range(0, len(pos1), step):
            p1 = pos1[i:i+step, np.newaxis, :]  # (chunk, 1, 3)
            d = np.linalg.norm(p1 - pos2[np.newaxis, :, :], axis=2)  # (chunk, n2)
            if len(pos1) == len(pos2):
                np.fill_diagonal(d, np.inf)
            for b in range(nb):
                c[b] += np.sum((d >= r_bins[b]) & (d < r_bins[b+1]))
        return c
    
    DD = pair_counts_unordered(positions, positions, r_bins)
    RR = pair_counts_unordered(rands, rands, r_bins)
    DR = pair_counts_unordered(positions, rands, r_bins)
    
    # Unordered pair normalization
    DD /= n_d * (n_d - 1)
    RR /= n_rand * (n_rand - 1)
    DR /= n_d * n_rand
    
    xi = np.zeros(nb)
    mask = RR > 0
    xi[mask] = (DD[mask] - 2*DR[mask] + RR[mask]) / RR[mask]
    return r_bins, xi


# ---------- main ----------
print("Generating 10 synthetic galaxy catalogs...")
print(f"Box: {L_BOX} Mpc/h | HOD M_min = {M_MIN*h:.1e} M_sun")

data_dir = '/home/node/work/projects/pointproc_cosmology/data'
os.makedirs(data_dir, exist_ok=True)

r_bins = np.array([1.0, 5.0, 10.0, 30.0, 100.0])
all_meta = []

for rid in range(10):
    rng = np.random.default_rng(1000 + rid * 7)
    print(f"\n  Realization {rid}/10")
    
    # Number of halos: integrate mass function
    # dN/dlnM ~ M^0.1 for M >> M_MIN, total ~ 0.003 / Mpc^3
    n_halos = int(0.003 * L_BOX ** 3)  # ~37500 halos - too many, use fewer
    n_halos = 5000  # practical number
    
    halo_m = sample_halo_masses(n_halos, rng)
    if len(halo_m) < n_halos:
        extra = np.exp(rng.uniform(np.log(M_MIN), np.log(5e14), n_halos - len(halo_m)))
        halo_m = np.concatenate([halo_m, extra])
    halo_m = halo_m[:n_halos]
    
    halo_x = rng.uniform(0, L_BOX, n_halos)
    halo_y = rng.uniform(0, L_BOX, n_halos)
    halo_z = rng.uniform(0, L_BOX, n_halos)
    halo_vx = rng.normal(0, 150, n_halos)
    halo_vy = rng.normal(0, 150, n_halos)
    halo_vz = rng.normal(0, 150, n_halos)
    
    print(f"    Halos: {n_halos}")
    
    # Assign galaxies
    all_gals = []
    for i in range(n_halos):
        all_gals.extend(assign_galaxies_to_halo(
            halo_m[i], halo_x[i], halo_y[i], halo_z[i],
            halo_vx[i], halo_vy[i], halo_vz[i], rng
        ))
    
    gals = np.array(all_gals, dtype=np.float64) if all_gals else np.zeros((0, 9))
    print(f"    Galaxies: {len(gals)}")
    
    # 2PCF
    r_out, xi = compute_2pcf(gals[:, 0:3], r_bins=r_bins)
    print(f"    xi(r<0.3)={xi[0]:.4f}, xi(1-3)={xi[3]:.4f}")
    
    # Save
    np.save(f'{data_dir}/galaxy_catalog_{rid:02d}.npy', gals)
    np.save(f'{data_dir}/halo_catalog_{rid:02d}.npy',
            np.column_stack([halo_x, halo_y, halo_z, halo_vx, halo_vy, halo_vz, halo_m]))
    r_cen = 0.5 * (r_out[:-1] + r_out[1:])
    np.save(f"{data_dir}/xi_2pcf_{rid:02d}.npy", np.stack([r_cen, xi]))
    
    nden = len(gals) / L_BOX**3
    meanL = np.mean(np.exp(gals[:, 6])) if len(gals) > 0 else 0.0
    all_meta.append([rid, len(gals), n_halos, nden, meanL, L_BOX])

np.save(f'{data_dir}/metadata.npy', np.array(all_meta))
print(f"\nSaved to {data_dir}")
print("Done!")
