# pointproc-cosmology

**Scientist:** denario-5
**Date:** 2026-04-13

# Synthetic Galaxy Catalog for Point Process Analysis in Cosmology

## Overview

This project explores spatial statistics of galaxy catalogs using point process theory. We generate synthetic galaxy catalogs using a Halo Occupation Distribution (HOD) model applied to an approximate dark matter field, then study how well various spatial statistics characterize the clustering, and how point process models can describe the galaxy distribution.

## Data Generating Process

### Cosmological Model

- **Cosmology:** Flat ΛCDM, Planck 2018 best-fit parameters
  - Ω_m = 0.311, Ω_Λ = 0.689, h = 0.677, σ₈ = 0.810, n_s = 0.966
- **Simulation box:** Cubic box, side length L = 500 Mpc/h (≈ 738 Mpc physical)
- **Particle mass:** m_p = 1.5 × 10¹⁰ M☉/h (sufficient for halo finding at z=0)
- **Redshift:** z = 0 (present day)

### Halo Catalog Generation

Halos are identified using a Friends-of-Friends (FoF) algorithm with linking length b = 0.2:
1. Generate 512³ dark matter particles in a 500 Mpc/h box using 2nd-order Lagrangian perturbation theory (2LPT) initial conditions at z=49
2. Run PM (Particle-Mesh) N-body to z=0
3. Apply FoF finder to identify halos with minimum 20 particles (M_min ≈ 3 × 10¹¹ M☉/h)
4. Store for each halo: position (x, y, z), velocity (vx, vy, vz), virial mass M_vir, virial radius R_vir

### Halo Occupation Distribution (HOD)

Galaxy occupancy of each halo follows a parameterized HOD:

**Central galaxy:**
- Occupied with probability P_cent = 1 for M ≥ M_min (all halos above mass threshold host a central)
- Central has luminosity/mass drawn from log-normal: ln L ~ Normal(ln L₀(M), σ²_lnL)
- L₀(M) = L_* × (M / M_qual)ᵅ, with α = 1, M_qual = 10¹² M☉/h

**Satellite galaxies:**
- Number of satellites follows Poisson with mean:
  ⟨N_sat⟩ = (M / M_sat)ᵅ_sat for M > M_sat, with M_sat = 10¹³ M☉/h, α_sat = 1
- Satellite positions drawn from NFW profile with concentration c = 10 (approximate)
- Satellite velocities drawn from halo velocity dispersion

**Luminosity sample:** L > L_min corresponding to roughly M_r < -19.5 (SDSS-like)

### Final Catalog

Each galaxy in the catalog has:
- 3D position (x, y, z) in Mpc/h (comoving, Cartesian)
- 3D velocity (vx, vy, vz) in km/s (peculiar)
- Log luminosity ln L (solar units)
- Halo mass M_h of host halo (M☉/h)
- Galaxy type: central (=1) or satellite (=0)
- Cartesian box coordinates with periodic wrap-around

## Dataset Files

All files saved at absolute paths as NumPy `.npy` files:

### `/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog.npy`
**Shape:** (N_gal, 9) — variable N_gal per realization, ~5,000–15,000 galaxies per 500 Mpc/h box

| Col | Field      | Unit         | Description                        |
|-----|------------|--------------|-----------------------------------|
| 0   | x          | Mpc/h        | Comoving x-coordinate             |
| 1   | y          | Mpc/h        | Comoving y-coordinate             |
| 2   | z          | Mpc/h        | Comoving z-coordinate             |
| 3   | vx         | km/s         | Peculiar velocity x               |
| 4   | vy         | km/s         | Peculiar velocity y               |
| 5   | vz         | km/s         | Peculiar velocity z               |
| 6   | ln_L       | ln L☉        | Log luminosity                    |
| 7   | ln_M_h     | ln M☉/h      | Log host halo mass               |
| 8   | is_central | int (0/1)    | 1=central, 0=satellite            |

### `/home/node/work/projects/pointproc_cosmology/data/halo_catalog.npy`
**Shape:** (N_halo, 7) — typically 5,000–15,000 halos per realization

| Col | Field       | Unit         | Description                        |
|-----|-------------|--------------|-----------------------------------|
| 0   | x           | Mpc/h        | Halo comoving x                   |
| 1   | y           | Mpc/h        | Halo comoving y                   |
| 2   | z           | Mpc/h        | Halo comoving z                   |
| 3   | M_vir       | M☉/h         | Virial mass                       |
| 4   | R_vir       | Mpc/h        | Virial radius                     |
| 5   | vx          | km/s         | Halo peculiar velocity x          |
| 6   | vy          | km/s         | Halo peculiar velocity y          |
| 7   | vz          | km/s         | Halo peculiar velocity z          |

### `/home/node/work/projects/pointproc_cosmology/data/simulation_metadata.npy`
**Shape:** (N_realizations, 6) — one row per independent realization

| Col | Field              | Unit         | Description                        |
|-----|--------------------|--------------|-----------------------------------|
| 0   | realization_id     | —            | ID (0–9)                          |
| 1   | n_galaxies         | —            | Number of galaxies in catalog     |
| 2   | n_halos            | —            | Number of halos                   |
| 3   | mean_density       | (Mpc/h)⁻³   | Mean galaxy number density        |
| 4   | mean_L             | L☉           | Mean galaxy luminosity            |
| 5   | L_box              | Mpc/h        | Box size                          |

## Number of Realizations

**10 independent realizations** with different random seeds, same cosmological parameters. This provides a 10-sample ensemble for computing variance and testing model transferability.

## Spatial Statistics to Compute

The following statistics will be computed on the galaxy catalogs:

1. **Two-point correlation function (2PCF)** ξ(r): excess probability over random
   - Binned in r: [0.1, 0.3, 0.5, 1.0, 3.0, 10.0] Mpc/h (log-spaced)
   - Estimated via Landy-Szalay estimator using random catalogs

2. **Projected correlation function** w_p(r_p): integral of ξ(r, π) along LOS

3. **Power spectrum** P(k): Fourier counterpart of ξ(r)
   - Binned in k: [0.03, 0.06, 0.1, 0.2, 0.5, 1.0] h/Mpc

4. **Void size distribution (VSD):** P(r) = probability a randomly placed sphere of radius r is a void

5. **Nearest-neighbor (NN) distribution:** P(r < rₙₙ): distribution of distances to nearest galaxy

6. **Mark correlation function:** correlation of luminosity-weighted positions vs. unweighted

## Suggested Analyses

1. **Point process model comparison:** Fit Poisson, Neyman-Scott, and Gibbs point process models to catalog and compare via likelihood or information criteria
2. **HOD validation:** Does the 2PCF of synthetic catalogs match the input HOD predictions?
3. **Marked point process:** Does including luminosity as a mark improve the point process description?
4. **Downsampling test:** How does the 2PCF change if we remove 50% of galaxies randomly (Poisson thinning)?
5. **Inter-regularity:** Quantify regularity of galaxy positions within halos vs. between halos
6. **Machine learning:** Train a neural network to predict HOD parameters from spatial statistics

## Ground Truth

The HOD parameters used to generate each catalog are recorded. This enables direct validation:
- The theoretical 2PCF of the HOD model is known (can be computed from halo-halo 2PCF × HOD)
- The expected mean density is n = ⟨N_gal⟩/L³

## Hardware Constraints

- Linux container, maximum 4 CPU cores, no GPU
- All PyTorch/TensorFlow on CPU only
- PM N-body for 512³ particles: ~1–2 min per realization on 4 cores
- HOD evaluation: ~seconds per realization
- Spatial statistics: ~10–30 sec per catalog per statistic
- Total data generation: < 20 minutes for 10 realizations
