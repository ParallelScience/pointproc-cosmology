**Title: Quantifying Assembly Bias via Cross-Marked Correlation of Halo-Centric Velocity Dispersion and Luminosity**

**Hypothesis:** The current HOD implementation, while spatially robust, implicitly assumes that satellite luminosity and kinematics are independent of the host halo's local environment. I hypothesize that by constructing a cross-marked correlation function $M_{v,L}(r)$—where the marks are the local velocity dispersion $\sigma_v(r)$ and the luminosity offset $\Delta \ln L$—we can detect "Assembly Bias" signatures that are invisible to standard 2PCF or simple luminosity-marked correlations. 

**Rationale:** Previous iterations confirmed that the 1-halo regime is dominated by a radial profile bias and that the 2-halo regime suffers from a lack of gravitational clustering. By shifting focus to the *kinematic* marks (peculiar velocities) relative to the *luminosity* marks, we can test if the HOD model produces "dynamically hot" satellites in high-luminosity halos, which would indicate a violation of the assumed spatial-mark independence. 

**Methodology:** 
1. Calculate the local velocity dispersion $\sigma_v$ for satellites within each halo.
2. Compute the cross-marked correlation function $M_{v,L}(r) = \frac{\langle \sigma_v \cdot \Delta \ln L \rangle_r}{\langle \sigma_v \rangle \langle \Delta \ln L \rangle}$ to measure the coupling between satellite kinematics and luminosity.
3. Compare this against a "shuffled" control catalog where luminosity marks are randomly reassigned to satellites within the same mass-bin to isolate the physical correlation from the stochastic HOD noise.
4. If $M_{v,L}(r) \neq 1$, it proves that the HOD model contains implicit assembly bias, providing a new diagnostic metric that is independent of the previously identified radial sampling errors.