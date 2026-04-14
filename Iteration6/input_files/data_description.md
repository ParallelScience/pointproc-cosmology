# Synthetic Galaxy Catalog for Point Process Analysis in Cosmology

## Overview

This project explores spatial statistics of galaxy catalogs using point process theory. We generate 10 independent synthetic galaxy catalogs using a Halo Occupation Distribution (HOD) model, then study how various spatial statistics characterize the clustering, and evaluate how well different point process models describe the galaxy distribution.

## Data Generating Process

### Cosmological Model

- **Cosmology:** Flat ΛCDM, Planck 2018 best-fit parameters (Ω_m = 0.311, Ω_Λ = 0.689, h = 0.677, σ₈ = 0.810, n_s = 0.966)
- **Simulation box:** Cubic, side length L = 500 Mpc/h (comoving, periodic boundaries)
- **Redshift:** z = 0 (present day)

### Halo Catalog (Neyman-Scott Parent Process)

Dark matter halos are modeled as a **Neyman-Scott cluster process** — a Poisson point process of halo centers with a gamma-distributed mass function:

    dN/dlnM ∝ M^0.3 × exp(−M / 5×10¹³ M☉/h)

Each realization contains 5,000 halos drawn from this mass function, distributed uniformly in the 500 Mpc/h box. Halo velocities are drawn from a Gaussian with σ_v ≈ 150 km/s (approximate velocity dispersion).

Halo fields in the saved files: x, y, z (Mpc/h), vx, vy, vz (km/s), M_vir (M☉/h), R_vir (Mpc/h)

### Halo Occupation Distribution (HOD)

Each halo is populated with galaxies as follows:

**Central galaxy** (always present if M > M_min = 3×10¹¹ M☉/h):
- Log luminosity ln(L/L★) drawn from Normal(α×ln(M/M_qual), σ²) with α=1, M_qual=10¹² M☉/h, σ=0.3
- L★ = 10¹⁰ L☉/h

**Satellite galaxies** (if M > M_sat = 10¹³ M☉/h):
- Number follows Poisson(⟨N_sat⟩) with ⟨N_sat⟩ = (M/M_sat)^α_sat, α_sat = 1.0
- Positions drawn from exponential radial profile with scale R_vir
- Luminosities drawn from log-normal centered 0.5 dex below central, σ=0.4 dex

This produces ~11,500–12,400 galaxies per realization, giving a mean number density n ≈ 9.4×10⁻⁵ (Mpc/h)⁻³ (SDSS-like).

## Dataset Files

All files are NumPy `.npy` arrays at absolute paths:

### `/home/node/work/projects/pointproc_cosmology/data/galaxy_catalog_XX.npy`
**Shape:** (N_gal, 9) — N_gal ≈ 11,500–12,400 per realization

| Col | Field       | Unit      | Description                        |
|-----|-------------|-----------|-----------------------------------|
| 0   | x           | Mpc/h     | Comoving x-coordinate (periodic)  |
| 1   | y           | Mpc/h     | Comoving y-coordinate             |
| 2   | z           | Mpc/h     | Comoving z-coordinate             |
| 3   | vx          | km/s      | Peculiar velocity x              |
| 4   | vy          | km/s      | Peculiar velocity y              |
| 5   | vz          | km/s      | Peculiar velocity z              |
| 6   | ln_L        | ln L☉     | Natural log of luminosity         |
| 7   | ln_M_h      | ln M☉/h   | Natural log of host halo mass     |
| 8   | is_central  | int 0/1   | 1=central galaxy, 0=satellite    |

### `/home/node/work/projects/pointproc_cosmology/data/halo_catalog_XX.npy`
**Shape:** (5000, 7)

| Col | Field  | Unit       | Description          |
|-----|--------|------------|---------------------|
| 0   | x      | Mpc/h      | Halo x              |
| 1   | y      | Mpc/h      | Halo y              |
| 2   | z      | Mpc/h      | Halo z              |
| 3   | vx     | km/s       | Halo velocity x     |
| 4   | vy     | km/s       | Halo velocity y     |
| 5   | vz     | km/s       | Halo velocity z     |
| 6   | M_vir  | M☉/h       | Virial mass         |

### `/home/node/work/projects/pointproc_cosmology/data/xi_2pcf_XX.npy`
**Shape:** (2, 4) — [r_bin_centers (Mpc/h), xi(r) values]

2PCF bins: [1, 5, 10, 30, 100] Mpc/h → centers: [3, 7.5, 20, 65] Mpc/h

### `/home/node/work/projects/pointproc_cosmology/data/metadata.npy`
**Shape:** (10, 6) — one row per realization

| Col | Field          | Unit           | Description                    |
|-----|----------------|----------------|-------------------------------|
| 0   | realization_id | —              | 0–9                           |
| 1   | n_galaxies     | —              | Number of galaxies            |
| 2   | n_halos        | —              | Number of halos (5000)        |
| 3   | mean_density   | (Mpc/h)⁻³     | Galaxy number density         |
| 4   | mean_L         | L☉             | Mean galaxy luminosity        |
| 5   | L_box          | Mpc/h          | Box size (500)                |

## Observed 2PCF Properties

The pre-computed 2PCF shows:
- **ξ(r) ≈ 60–65** at r = 1–5 Mpc/h: strong 1-halo (satellite-satellite) term
- **ξ(r) ≈ 0.3–0.4** at r = 5–10 Mpc/h: 1-halo to 2-halo transition
- **ξ(r) ≈ 0** at r = 10–30 Mpc/h: essentially uncorrelated

## Spatial Statistics to Compute

1. **Two-point correlation function (2PCF)** ξ(r): Landy-Szalay estimator
2. **Projected correlation function** w_p(r_p): integral of ξ(r, π) along LOS
3. **Power spectrum** P(k): Fourier counterpart of ξ(r)
4. **Void probability function (VPF):** P₀(r) = prob. a sphere of radius r is empty
5. **Nearest-neighbor distribution:** P(r < rₙₙ)
6. **Mark correlation function:** Luminosity-weighted vs. unweighted clustering
7. **Inter-particle distance distribution:** G(r) and R(r) functions

## Point Process Models to Compare

1. **Poisson point process** (null model): uniform random
2. **Neyman-Scott process** (Thomas cluster): parent halos + satellite offspring
3. **Gibbs point process** (pairwise interaction): soft-core / hard-core models
4. **Log-Gaussian Cox process (LGCP):** log-normal intensity field
5. **Inhomogeneous Poisson** with density n(r) inferred from data

## Suggested Research Analyses

1. **Model selection via information criteria:** Compare point process models using AIC/BIC on the 10 realizations
2. **HOD validation:** Does the 2PCF of synthetic catalogs match the known input HOD?
3. **Marked point process:** Does adding luminosity as a mark improve the model (marked vs. unmarked 2PCF)?
4. **Spatial dependence of the intensity:** Test whether n(r) varies with distance from box center
5. **Downsampling / thinning:** How does the 2PCF change under random Poisson thinning?
6. **Machine learning:** Train a neural network to predict HOD parameters (M_min, α_sat) from spatial statistics
7. **Inter-regularity:** Quantify regularity of galaxy positions within halos (Neyman-Scott vs. Gibbs)

## Ground Truth

The HOD parameters are fully specified above. Key predictions:
- Mean galaxy density: n ≈ 9.4×10⁻⁵ (Mpc/h)⁻³
- 1-halo term dominates at r < 5 Mpc/h (satellite-satellite pairs in same halo)
- Central-satellite and satellite-satellite pairs produce distinct 2PCF shapes
- The HOD model produces a quasi-linear 2PCF at r > 10 Mpc/h

## Hardware Constraints

- Linux container, maximum 4 CPU cores, no GPU
- All NumPy/SciPy on CPU; PyTorch (if used) on `device='cpu'`
- Data loading: ~2 sec per realization
- Spatial statistics (pair counting): ~10–30 sec per catalog
- Total data generation: < 5 minutes for 10 realizations
