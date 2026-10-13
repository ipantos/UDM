# Sources

Core analysis source: https://github.com/ipantos/UDM , commit `cff8bbef4eedca4b8f1b420a47fd07919f289762`.
Supplementary and paper-figure scripts cross-referenced against the user's supplied `UDM_complete_manuscript(5).zip`, submission-support directory. Only the production numerical sources and the final paper's figure builder are packaged; old manuscript attempts are not input to the new run.

Union3: https://github.com/rubind/union3_release , commit `f5c387349b68f3535fdabf0fbc4f45cd407f9084`, file `mu_mat_union3_cosmo=2_mu.fits`. The supplied `union3_official.fits` is this source file; the numerical comparison is in `union3_verified.json` and is repeated by `verify_package.py`.

Growth correction: Qin et al., arXiv:2512.03231, power-spectrum measurement at z=0.07. The input comment records the original asymmetric uncertainties and the distinct upper cuts of the density/cross and momentum spectra.

`unchanged_inputs.json` stores the previous input sections and all growth rows except the corrected Qin row, exclusively for input-integrity checking. It is not loaded by the scientific likelihood.

`verification_results.json` and `boundary_regression.json` document targeted package tests, not new full scientific results. `SHA256SUMS.txt` lists the packaged files.
