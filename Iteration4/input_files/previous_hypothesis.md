The observed spatial discrepancies (the "core-softened" kernel and the 14 Mpc/h leakage) are not merely implementation errors, but are indicative of a **non-isotropic satellite distribution** induced by the halo velocity dispersion ($\sigma_v \approx 150$ km/s) interacting with the periodic box boundaries. I hypothesize that the satellites are not distributed according to a static 3D radial profile, but are instead "stretched" along the principal axes of the halo's velocity ellipsoid. 

To test this, I propose a **Velocity-Aligned Anisotropy Analysis**:
1. Calculate the velocity dispersion tensor for each halo to define a local coordinate system (eigenvectors of the velocity field).
2. Transform satellite positions into this velocity-aligned frame.
3. Compute the **Anisotropic 2-Point Correlation Function** $\xi(r_p, \pi)$ and the **Multipole Expansion** ($\xi_0, \xi_2, \xi_4$) specifically within the 1-halo regime. 
4. If the "core-softening" is a projection effect of velocity-space anisotropy, the quadrupole moment ($\xi_2$) will show a non-zero signal that correlates with the halo mass-dependent velocity dispersion. 

This will determine if the "leakage" into the 2-halo regime is a physical consequence of the HOD's velocity implementation, allowing us to define a "velocity-corrected" spatial kernel that accounts for the observed 3PCF and VPF anomalies without requiring a change to the underlying HOD parameters.