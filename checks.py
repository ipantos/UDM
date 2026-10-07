"""Sensitivity of the sound-speed limit to likelihood assumptions, by importance
reweighting of the stored posterior.

Keep this file next to fit_joint.py, figures.py and udm_data.txt, after the
posterior has been produced (start, nodes, mcmc 40000, summary).  Run

    python3 checks.py

For every stored sample the growth chi^2 is recomputed under an alternative
assumption, and the sample is reweighted by exp[-(chi^2_alt - chi^2_base)/2].
The 95% upper limit on c_s^2 is then the weighted 95th percentile.  The
background, supernova and BAO terms cancel in the weight.  No new chain is run;
the effective sample size (ESS) says how well the reweighted posterior is
resolved.  Variants:

  init_0.95, init_1.05   initial velocity rate V/D at z = 2.33 set to 0.95 or
                         1.05 instead of 1
  init_local_mode        V/D at z = 2.33 set to the local growing-mode value of
                         the system with the background frozen at its z = 2.33
                         values (w_N = 0 there): the growing root n of
                         n^2 + (1-q-3w) n - 3w(1-q) - (1+q) + Q = 0 gives
                         V/D = n - 3w
  z0.07_window_0.025_0.1 the z = 0.07 measurement assigned the window
                         0.025 < k < 0.1 h/Mpc of the power-spectrum fit
  z0.07_removed          the z = 0.07 measurement removed from the likelihood

The response of each growth measurement to the sound speed is also recorded:
at the best stored sample, the prediction is computed with c_s^2 = 0 and with
c_s^2 = the 95% upper limit, and the decrease is given as a fraction of the
prediction and in units of the measurement error (Sec. V.D of the paper).

Within-chain stability (Appendix E.3 of the paper) is assessed
from the sampler checkpoint: the 95% upper limit and the sigma_8 median are
computed separately from the two halves of the retained steps (20,001-30,000
and 30,001-40,000), and from the twenty walkers whose initial u = log10 c_s^2
was lowest and the twenty whose initial u was highest.  The initial walker
positions are regenerated with the seed and the procedure of fit_joint.py
(the checkpoint stores the chain from step 1 on, not the initial positions).

These diagnostics do not by themselves establish independence from the
initialization.

Output: checks.json next to this file.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import figures as G  # noqa: E402
import fit_joint as fj  # noqa: E402


def growth(parameters, k_values, z_out, init, steps=1600):
    """figures.growth with a selectable initial velocity rate V/D at z = 2.33."""
    t_out = -np.log1p(np.asarray(z_out))
    times = np.unique(np.r_[np.linspace(-G.XG[-1], 0, steps + 1), t_out])
    half = .5*(times[1:] + times[:-1])
    x = -np.r_[times, half]
    idx, w = G.interpolation_weights(G.XG, x)
    basis = G.SMAT[idx]*(1 - w[:, None]) + G.SMAT[idx + 1]*w[:, None]
    q = -1 - parameters[:, :7] @ basis.T
    h = G.hubble_at(parameters, x, grid=G.XG)
    pbase = parameters[:, 8, None]*(G.C_OVER_H0*h*np.exp(x))**2
    kval = np.square(k_values)[None, :]
    opw = 2*(1 + q)/3
    d = np.ones((len(parameters), len(k_values)))*opw[:, 0, None]
    if init == 'local_mode':
        q0 = q[:, 0, None]
        w0 = (2*(1 + q0)/3) - 1
        Q0 = pbase[:, 0, None]*kval
        b = 1 - q0 - 3*w0
        c = -3*w0*(1 - q0) - (1 + q0) + Q0
        n_plus = (-b + np.sqrt(b*b - 4*c))/2
        v = (n_plus - 3*w0)*np.ones_like(d)
    else:
        v = np.full_like(d, float(init))
    result_d = np.empty((len(parameters), len(z_out), len(k_values)))
    result_v = np.empty_like(result_d)
    destinations = {int(np.searchsorted(times, t)): j for j, t in enumerate(t_out)}
    offset = len(times)
    for i, dt in enumerate(np.diff(times)):
        d, v = G.rk4_step(d, v, dt, q[:, i, None], q[:, offset + i, None], q[:, i + 1, None],
                          pbase[:, i, None]*kval, pbase[:, offset + i, None]*kval,
                          pbase[:, i + 1, None]*kval)
        if i + 1 in destinations:
            j = destinations[i + 1]
            result_d[:, j], result_v[:, j] = d/opw[:, i + 1, None], v
    return result_d, result_v


def predictions(parameters, z, k, init):
    k_unique, kmap = np.unique(np.r_[k, 0.], return_inverse=True)
    j_norm = int(np.flatnonzero(np.isclose(z, fj.Z_NORM, rtol=0, atol=1e-12))[0])
    d, v = growth(parameters, k_unique, z, init)
    n = len(z)
    return parameters[:, 9, None]*v[:, np.arange(n), kmap[:n]]/d[:, j_norm, kmap[-1], None]


def chi2(prediction, y, covariance):
    residual = y - prediction
    return np.einsum('bi,ij,bj->b', residual, np.linalg.inv(covariance), residual)


def weighted_percentile(values, weights, percent):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(percent/100, cumulative, values[order]))


def initial_walker_positions(number_of_nodes):
    """The initial positions of the walkers, regenerated as in fit_joint.command_mcmc."""
    z_nodes = fj.node_positions(number_of_nodes)
    dimension = fj.NN + 3 + number_of_nodes
    central = fj.load_result(f'best_{number_of_nodes}', 'best_start', 'physical',
                             number_of_nodes)['parameters']
    central = fj.to_sampling(central, number_of_nodes)
    scale = np.abs(central)*1.0e-3 + 1.0e-9
    coords = np.empty((fj.NWALKERS, dimension))
    rng = np.random.default_rng(fj.SEED)
    for walker in range(fj.NWALKERS):
        for _ in range(100000):
            candidate = central + scale*rng.standard_normal(dimension)
            candidate[fj.cs_slot(number_of_nodes)] = rng.uniform(*fj.PRIOR_U, number_of_nodes)
            if np.isfinite(fj.log_prob(candidate, z_nodes)):
                coords[walker] = candidate
                break
        else:
            raise RuntimeError(f'Could not initialise walker {walker}')
    return coords


def within_chain_stability(number_of_nodes=1):
    """Upper limit and sigma_8 median from halves of the retained chain and from
    walker groups split by their initial u = log10 c_s^2."""
    arrays = fj.load_sampler_checkpoint(number_of_nodes)
    post = fj.post_burn_chain(arrays['thin_chain'], fj.NBURN, fj.THIN)
    dimension = fj.NN + 3 + number_of_nodes
    i_cs2, i_s8 = fj.NN + 1, fj.NN + 1 + number_of_nodes

    def summary(block):
        physical = fj.to_physical(block.reshape(-1, dimension), number_of_nodes)
        return {'rows': int(len(physical)),
                'cs2_upper95': float(np.percentile(physical[:, i_cs2], 95)),
                'sigma8_median': float(np.median(physical[:, i_s8]))}

    half = len(post)//2
    first_step = fj.NBURN + 1
    out = {'thinned_rows_per_walker': int(len(post)),
           'halves': {f'steps_{first_step}-{fj.NBURN + half*fj.THIN}': summary(post[:half]),
                      f'steps_{fj.NBURN + half*fj.THIN + 1}-{fj.NBURN + len(post)*fj.THIN}':
                          summary(post[half:])}}
    u_start = initial_walker_positions(number_of_nodes)[:, i_cs2]
    order = np.argsort(u_start)
    groups = {'lowest_initial_u': order[:fj.NWALKERS//2],
              'highest_initial_u': order[fj.NWALKERS//2:]}
    out['walker_groups'] = {}
    for name, index in groups.items():
        entry = summary(post[:, np.sort(index)])
        entry['walkers'] = int(len(index))
        entry['initial_u_range'] = [float(u_start[index].min()), float(u_start[index].max())]
        out['walker_groups'][name] = entry
    return out


def main():
    chain = fj.posterior_chain()
    obs = G.load_observations()
    z, k, y, cov = obs['growth_z'], obs['growth_k'], obs['growth_y'], obs['growth_cov']
    i07 = int(np.argmin(np.abs(z - 0.07)))
    k_alt = k.copy()
    k_alt[i07] = np.round(fj.k_effective(0.025, 0.1), 6)
    variants = {'baseline': (1.0, k), 'init_0.95': (0.95, k), 'init_1.05': (1.05, k),
                'init_local_mode': ('local_mode', k), 'z0.07_window_0.025_0.1': (1.0, k_alt)}
    keep = np.arange(len(z)) != i07
    chi = {name: [] for name in variants}
    chi['z0.07_removed'] = []
    block = 2000
    for first in range(0, len(chain), block):
        part = chain[first:first + block]
        for name, (init, kk) in variants.items():
            pred = predictions(part, z, kk, init)
            chi[name].append(chi2(pred, y, cov))
            if name == 'baseline':
                chi['z0.07_removed'].append(chi2(pred[:, keep], y[keep], cov[np.ix_(keep, keep)]))
        print(f'{min(first + block, len(chain))}/{len(chain)} samples', flush=True)
    chi = {name: np.concatenate(values) for name, values in chi.items()}
    cs2 = chain[:, 8]
    base_limit = float(np.percentile(cs2, 95))
    # Consistency of this solver with the fit: the baseline chi^2 of the best
    # stored sample is compared with the fit's own value.
    stored = fj.load_result('n1_posterior', 'posterior', 'physical', 1)
    best = int(np.argmin(stored['chi2']))
    inv_h, _, _, q_grid = fj.background(chain[best, :7], chain[best, 7])
    shape = fj.growth_shape(fj.cs2_on_grid(chain[best, 8:9], fj.node_positions(1)), q_grid, inv_h, chain[best, 7])
    fit_growth_chi2 = float(chi2((chain[best, 9]*shape)[None, :], y, cov)[0])
    out = {'baseline_upper95': base_limit,
           'solver_consistency': {'sample': best, 'growth_chi2_fit_solver': fit_growth_chi2,
                                  'growth_chi2_this_solver': float(chi['baseline'][best])},
           'variants': {}}
    for name, values in chi.items():
        if name == 'baseline':
            continue
        log_weight = -.5*(values - chi['baseline'])
        log_weight -= log_weight.max()
        weight = np.exp(log_weight)
        limit = weighted_percentile(cs2, weight, 95)
        out['variants'][name] = {'upper95': limit, 'change_percent': 100*(limit/base_limit - 1),
                                 'effective_sample_size': float(weight.sum()**2/np.sum(weight**2))}
    # Response of each measurement to the sound speed at the best stored sample:
    # prediction with c_s^2 = 0 against prediction with c_s^2 = the 95% upper limit.
    def best_prediction(cs2_value):
        grid = fj.cs2_on_grid(np.array([cs2_value]), fj.node_positions(1))
        return chain[best, 9]*fj.growth_shape(grid, q_grid, inv_h, chain[best, 7])
    sigma = np.sqrt(np.diag(cov))
    without, with_limit = best_prediction(0.0), best_prediction(base_limit)
    out['response_to_cs2_at_best_sample'] = {
        'cs2_upper95': base_limit,
        'z': [float(v) for v in z], 'k_eff': [float(v) for v in k],
        'decrease_fraction': [float(v) for v in 1 - with_limit/without],
        'decrease_over_sigma': [float(v) for v in (without - with_limit)/sigma]}
    out['within_chain_stability'] = within_chain_stability()
    (ROOT / 'checks.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
