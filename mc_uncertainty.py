"""Monte Carlo (sampling) uncertainty of the 95% posterior upper limit on c_s^2.

Method: batch means. The retained part of the main chain (raw steps after the
burn-in NBURN, stored every THIN steps for all walkers) is cut into B consecutive
blocks in time. The 95% quantile of c_s^2 is computed in every block; the standard
error of the full-chain limit is std(block limits, ddof=1)/sqrt(B). B = 5, 10, 20.
Also prints the limit in the two halves and the four quarters of the retained chain.
It is the sampling error of the computed limit, not an observational error on c_s^2.
Usage: python mc_uncertainty.py   (run in the package folder)
"""
import numpy as np
import fit_joint as fj

arrays = fj.load_result('n1_mcmc', 'mcmc', 'sampling', 1)
post = fj.post_burn_chain(arrays['thin_chain'], fj.NBURN, fj.THIN)   # (rows, walkers, params)
column = 8                                                     # log10 c_s^2 in the sampling coordinates
cs2 = 10.0**post[:, :, column]
upper = lambda a: np.quantile(a, 0.95)
limit = upper(cs2)
print(f"raw steps done: {int(arrays['done'])}; retained rows x walkers: {cs2.shape}; samples: {cs2.size}")
print(f"95% upper limit: {limit:.4e}")
print('halves:  ', ['%.4e' % upper(h) for h in np.array_split(cs2, 2, axis=0)])
print('quarters:', ['%.4e' % upper(h) for h in np.array_split(cs2, 4, axis=0)])
for blocks in (5, 10, 20):
    values = np.array([upper(h) for h in np.array_split(cs2, blocks, axis=0)])
    error = values.std(ddof=1)/np.sqrt(blocks)
    print(f"batch means, {blocks:2d} blocks: standard error {error:.3e} ({100*error/limit:.1f}%)")
