

Iteration 0:
# Summary: Satellite Radial Profile and Marked Point Process Analysis

### 1. Dataset & Methodology
*   **Data:** 10 synthetic galaxy catalogs (HOD model, $\Lambda$CDM, $L=500$ Mpc/h).
*   **Method:** KD-Tree spatial matching of galaxies to host halos ($M \ge 10^{13} M_\odot/h$).
*   **Analysis:** MLE fitting of radial density profiles $n(r|M_{vir})$ and 1-halo 2PCF ($\xi_{cs}, \xi_{ss}$) using Landy-Szalay estimator. Marked correlation $M(r)$ evaluated using absolute luminosity vs. luminosity offset ($\ln L_{sat} - \ln L_{cen}$).

### 2. Key Findings
*   **Spatial Kernel:** Satellite distribution is best described by a 3D exponential kernel with scale $\lambda \approx 0.33 R_{vir}$ (concentration $c \approx 3$). This is stable across two orders of magnitude in halo mass, confirming self-similarity.
*   **1-Halo Clustering:** Strong small-scale amplification ($\xi \sim 10^5$ at $r=0.1$ Mpc/h) relative to uniform null. $\xi_{cs}$ and $\xi_{ss}$ show distinct geometric signatures consistent with the auto-convolution of the exponential kernel.
*   **Marked Correlation:** Absolute luminosity is strongly confounded by host halo mass. Using the luminosity offset as a mark reveals strict spatial-mark independence ($M(r) \approx 1.0$), confirming the HOD implementation is a separable marked Neyman-Scott process.

### 3. Limitations & Uncertainties
*   **Boundary Effects:** $\chi^2$ statistics indicate deviations from the analytical exponential model, likely due to truncation at $R_{vir}$.
*   **Model Selection:** Exponential kernel is statistically superior to Gaussian/Uniform, but finite-size effects in the outer halo shells remain a source of systematic bias.

### 4. Decisions for Future Experiments
*   **Concentration Parameter:** Future models must account for the $c \approx 3$ concentration inherent in the current HOD spatial implementation.
*   **Conditioning:** Any analysis of luminosity-spatial correlations must use the luminosity offset to avoid mass-confounding.
*   **Next Steps:** The dataset is validated for ML-based HOD parameter inference. Future work should investigate the impact of varying the concentration parameter $c$ or introducing dynamical friction (non-independent marks) to test the limits of the current point process model.
        