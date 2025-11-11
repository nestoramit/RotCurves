# RotCurves
RotCurves is a python-based, light-weight astronomical tool designed for modelling galactic rotation curves
and to fit observational data. 
The code is uses a forward-modelling approach to model axisymmetric disk galaxies that are rotationally supported ($V/\sigma_0 \gtrsim 2$) and have a well-defined kinematic axis, with no strong deviations from circular motions. 
This forward-modelling approach allows to directly recover the physical parameters of the galaxy from the 1D rotation curve, without any additional steps. 

The code constructs a mass model of an idealized, axisymmetric galaxy with mass components following defined analytical density distributions and a constant, isotropic velocity dispersion. 
The model consists of a disk, a bulge and a dark matter halo, each having various available profile shapes the user can choose from.
The equilibrium rotation velocity is mock-observed given the sky-orientation and the instrumental point-spread-function (PSF). 
We simulate the projected beam on the plane of the galaxy (i.e., "beam projection"), reconstructing the beam-smeared rotation curve and fitting it directly with the observational data.
We perform the fit with a Bayesian inference approach, using a Monte-Carlo Markov-Chain (MCMC, with $\texttt{emcee}$) algorithm to sample the posterior parameters distribution.
The fitting procedure peforms best for resolved systems, with a beam size that is of the same size or smaller than the effective radius of the disk $PSF \lesssim R_{\rm eff,disk}$.

For Additional information see Nestor Shachar et al. 2025. 

### Installation

### Usage

### Dependencies
- A
- B

### Citations
If used in a scientific paper, please make sure to cite: Nestor Shachar et al. 2025.
### Contact
Please feel free to reach us at RotCurves@gmail.com.

### License
