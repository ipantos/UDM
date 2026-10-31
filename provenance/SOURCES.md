# Sources

Core analysis source: https://github.com/ipantos/UDM , commit `cff8bbef4eedca4b8f1b420a47fd07919f289762`.
The supplementary and paper-figure scripts are those of the final manuscript. Only the production numerical sources and the figure builder of the paper are packaged; earlier versions of the manuscript are not inputs of the run.

Union3: https://github.com/rubind/union3_release , commit `f5c387349b68f3535fdabf0fbc4f45cd407f9084`, file `mu_mat_union3_cosmo=2_mu.fits`. The supplied `union3_official.fits` is this source file; the numerical comparison is in `union3_verified.json` and is repeated by `verify_package.py`.

Growth correction: Qin et al., arXiv:2512.03231, power-spectrum measurement at z=0.07. The input comment records the original asymmetric uncertainties and the distinct upper cuts of the density/cross and momentum spectra.

`unchanged_inputs.json` stores the previous input sections and all growth rows except the corrected Qin row, exclusively for input-integrity checking. It is not loaded by the scientific likelihood.

`verification_results.json` and `boundary_regression.json` document targeted package tests, not new full scientific results. `SHA256SUMS.txt` lists the packaged files.

WiggleZ wavenumber range: the comment in `udm_data.txt` for the three WiggleZ rows (z = 0.44, 0.60, 0.73) attributes the range k < 0.2 h/Mpc to Blake et al. 2012, arXiv:1204.3674, Sec. 3.2, and states that k_min is not quoted. Both statements are incorrect. Blake et al. 2012 states no wavenumber range; its Sec. 2.3 says it uses the 2D power-spectrum analysis of Blake et al. 2011c (MNRAS 418, 1725; arXiv:1108.2637). That analysis fits the range 0 < k < k_max = 0.2 h/Mpc (Sec. 3.4 and the caption of Fig. 3; Sec. 3.3 writes k < 0.2 h/Mpc), so both ends of the range are quoted and k_min = 0 is not an assumption. The values in `udm_data.txt` (k_min = 0, k_max = 0.2 h/Mpc) are correct and unchanged; the source and the lower end are corrected here rather than in `udm_data.txt`, so that file and the cached results that depend on its exact contents remain unchanged.
