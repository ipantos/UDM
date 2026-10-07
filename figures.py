"""Figures of the UDM manuscript, computed from the posterior of fit_joint.py.

Keep this file next to fit_joint.py and udm_data.txt.  First produce the
posterior with fit_joint.py (start, nodes, mcmc 40000, summary), then run

    python3 figures.py

Stage 1 recomputes the plotted quantities from the posterior sample with an
independent NumPy implementation of the background spline and of the growth
equations used in the likelihood (the exact y-V system of Appendix C,
fixed-step fourth-order integration), and compares the growth predictions
with the fit's own integration on 32 samples.
Stage 2 writes PGFPlots sources; stage 3 compiles them with pdflatex.
Outputs: figure_data/ (tables and numerical_validation.json),
figure_tex/ (plot sources) and figures/ (fig01-fig04 PDFs).
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fit_joint as fj  # noqa: E402  (posterior sample, data, and the fit's own growth solver)

posterior_indices = fj.posterior_indices
OUT = ROOT / 'figure_data'
TEX = ROOT / 'figure_tex'
FIGURES = ROOT / 'figures'
NGRID, SEED = 3201, 20260809
C_OVER_H0 = 2997.92458  # Hubble distance c/H0 in h^-1 Mpc; wavenumbers are in h Mpc^-1
Z_NODES = np.array([.295, .510, .706, .934, 1.321, 1.484, 2.330])
X_NODES = np.log1p(Z_NODES)
XG = np.linspace(0., X_NODES[-1], NGRID)
ZG = np.expm1(XG)
ANISO = np.array(fj.ANISO)   # BAO data, read by fit_joint.py from udm_data.txt
BGS = fj.BGS
PERCENTILES = [2.5, 16., 50., 84., 97.5]


class NotAKnotSpline:
    """Piecewise cubic interpolation from a direct second-derivative solve.

    Equal third derivatives at the first and last interior knots give the
    not-a-knot boundary conditions. End polynomials are used for extrapolation.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.scalar = y.ndim == 1
        if self.scalar:
            y = y[:, None]
        h = np.diff(self.x)
        if len(x) < 4 or np.any(h <= 0) or y.shape[0] != len(x):
            raise ValueError('At least four increasing knots are required')
        n = len(x)
        matrix = np.zeros((n, n)); rhs = np.zeros_like(y)
        matrix[0, :3] = [-h[1], h[0] + h[1], -h[0]]
        matrix[-1, -3:] = [-h[-1], h[-2] + h[-1], -h[-2]]
        slope = np.diff(y, axis=0) / h[:, None]
        for i in range(1, n - 1):
            matrix[i, i-1:i+2] = [h[i-1], 2*(h[i-1] + h[i]), h[i]]
            rhs[i] = 6*(slope[i] - slope[i-1])
        second = np.linalg.solve(matrix, rhs)
        self.coeff = np.stack([y[:-1], slope - h[:, None]*(2*second[:-1] + second[1:])/6,
                               second[:-1]/2, np.diff(second, axis=0)/(6*h[:, None])])

    def __call__(self, x, derivative=0):
        x = np.asarray(x, dtype=float)
        index = np.clip(np.searchsorted(self.x, x, side='right') - 1, 0, len(self.x)-2)
        d = (x - self.x[index])[..., None]
        a, b, c, e = self.coeff[:, index]
        if derivative == 0:
            value = a + d*(b + d*(c + d*e))
        elif derivative == 1:
            value = b + d*(2*c + 3*d*e)
        elif derivative == 2:
            value = 2*c + 6*d*e
        elif derivative == 3:
            value = 6*e
        else:
            raise ValueError('Unsupported derivative')
        return value[..., 0] if self.scalar else value


BASIS = NotAKnotSpline(X_NODES, np.eye(7))
SMAT = BASIS(XG)
IMAT = np.vstack([np.zeros(7), np.cumsum(.5*(SMAT[1:] + SMAT[:-1])*np.diff(XG)[:, None], axis=0)])


def interpolation_weights(grid, points):
    points = np.asarray(points)
    index = np.clip(np.searchsorted(grid, points, side='right') - 1, 0, len(grid)-2)
    weight = (points - grid[index]) / (grid[index+1] - grid[index])
    return index, weight


def interpolate_rows(values, points, grid=ZG):
    i, w = interpolation_weights(grid, points)
    return values[:, i]*(1-w) + values[:, i+1]*w


def hubble_at(parameters, points, grid=ZG):
    """Interpolate exp(IMAT @ eps), exactly in the order used by the source."""
    i, w = interpolation_weights(grid, points)
    low = np.exp(parameters[:, :7] @ IMAT[i].T)
    high = np.exp(parameters[:, :7] @ IMAT[i+1].T)
    return low*(1-w) + high*w


def background(parameters):
    inverse = np.exp(parameters[:, :7] @ IMAT.T)
    dh = parameters[:, 7, None]*inverse
    dm = parameters[:, 7, None]*np.column_stack([
        np.zeros(len(parameters)), np.cumsum(.5*(inverse[:, 1:]+inverse[:, :-1])*np.diff(ZG), axis=1)])
    return inverse, dm, dh


def load_observations():
    # Raw data as read by fit_joint.py from udm_data.txt; the selection, effective
    # wavenumbers and covariance assembly below are computed independently here.
    rows = [row for row in csv.DictReader(fj.DATA['growth']) if row['space'] == 'fourier' and row['k_max']]
    z = np.array([float(row['z_eff']) for row in rows])
    order = np.argsort(z)
    y = np.array([float(row['fsigma8']) for row in rows])[order]
    sigma = np.array([float(row['sigma']) for row in rows])[order]
    lower = np.array([float(row['k_min']) if row['k_min'] else 0 for row in rows])[order]
    upper = np.array([float(row['k_max']) for row in rows])[order]
    z = z[order]
    keff = np.sqrt(.6*(upper**5-lower**5)/(upper**3-lower**3))
    covariance = np.diag(sigma**2)
    idx = {round(float(value), 3): i for i, value in enumerate(z)}
    wiggle = fj.WIGGLEZ_COV
    wi = [idx[value] for value in (.44, .60, .73)]
    covariance[np.ix_(wi, wi)] = wiggle
    eboss = fj.EBOSS_CORR
    ei = [idx[value] for value in (.978, 1.230, 1.526, 1.944)]
    covariance[np.ix_(ei, ei)] = eboss*sigma[ei, None]*sigma[ei][None, :]
    sn = np.array([[float(line.split()[1]), float(line.split()[4])] for line in fj.DATA['sn_union3']])
    sncov = np.array([[float(v) for v in line.split()] for line in fj.DATA['sn_union3_covariance']])
    assert len(rows) == 17 and len(sn) == 22
    assert np.min(np.linalg.eigvalsh(covariance)) > 0 and np.min(np.linalg.eigvalsh(sncov)) > 0
    return {'growth_z': z, 'growth_y': y, 'growth_cov': covariance,
            'growth_k': np.round(keff, 6), 'growth_kmax': upper,
            'sn_z': sn[:, 0], 'sn_y': sn[:, 1], 'sn_cov': sncov}


def rk4_step(y, v, dt, q0, qm, q1, p0, pm, p1):
    """One RK4 step of the exact system (Appendix C), in N = ln a:
        y_N = 3 w y + (1+w) V,   V_N = -(1-q) V + [3/2 - P/(1+w)] y,
    with y = (1+w) D, V = D_N + beta D, 1+w = 2(1+q)/3, P = c_s^2 k^2/(aH)^2."""
    def rates(a, b, q, p):
        opw = 2*(1+q)/3
        return 3*(opw-1)*a + opw*b, -(1-q)*b + (1.5-p/opw)*a
    ky1, kv1 = rates(y, v, q0, p0)
    ky2, kv2 = rates(y+.5*dt*ky1, v+.5*dt*kv1, qm, pm)
    ky3, kv3 = rates(y+.5*dt*ky2, v+.5*dt*kv2, qm, pm)
    ky4, kv4 = rates(y+dt*ky3, v+dt*kv3, q1, p1)
    return y + dt*(ky1+2*ky2+2*ky3+ky4)/6, v + dt*(kv1+2*kv2+2*kv3+kv4)/6


def growth(parameters, k_values, z_out, steps=1600):
    """Batch RK4 with exact output times and the source's interpolated background.

    D=1 and V/D=1 at z=2.330 for every mode; no early-time continuation is added.
    Returns D = y/(1+w) and the velocity V at the output redshifts.
    """
    t_out = -np.log1p(np.asarray(z_out))
    times = np.unique(np.r_[np.linspace(-XG[-1], 0, steps+1), t_out])
    if times[0] < -XG[-1]-1e-12 or times[-1] > 1e-12:
        raise ValueError('Growth prediction requested outside the fitted interval')
    half = .5*(times[1:] + times[:-1])
    x = -np.r_[times, half]
    idx, w = interpolation_weights(XG, x)
    basis = SMAT[idx]*(1-w[:, None]) + SMAT[idx+1]*w[:, None]
    q = -1 - parameters[:, :7] @ basis.T
    h = hubble_at(parameters, x, grid=XG)
    pbase = parameters[:, 8, None]*(C_OVER_H0*h*np.exp(x))**2
    kval = np.square(k_values)[None, :]
    opw = 2*(1+q)/3                       # 1 + w on the integration points
    d = np.ones((len(parameters), len(k_values)))*opw[:, 0, None]; v = np.ones_like(d)
    result_d = np.empty((len(parameters), len(z_out), len(k_values)))
    result_v = np.empty_like(result_d)
    destinations = {int(np.searchsorted(times, t)): j for j, t in enumerate(t_out)}
    if 0 in destinations:
        result_d[:, destinations[0]], result_v[:, destinations[0]] = d/opw[:, 0, None], v
    offset = len(times)
    for i, dt in enumerate(np.diff(times)):
        d, v = rk4_step(d, v, dt, q[:, i, None], q[:, offset+i, None], q[:, i+1, None],
                       pbase[:, i, None]*kval, pbase[:, offset+i, None]*kval, pbase[:, i+1, None]*kval)
        if i+1 in destinations:
            j = destinations[i+1]
            result_d[:, j], result_v[:, j] = d/opw[:, i+1, None], v
    if not (np.isfinite(result_d).all() and np.isfinite(result_v).all()):
        raise FloatingPointError('Nonfinite growth output')
    return result_d, result_v


def growth_predictions(parameters, obs, steps=1600):
    """sigma_8 V(k,z)/D(k=0, z=Z_NORM), with the normalisation redshift of the fit."""
    k, kmap = np.unique(np.r_[obs['growth_k'], 0.], return_inverse=True)
    z = obs['growth_z']
    # Z_NORM is one of the growth redshifts; using it directly avoids a duplicate output time.
    j_norm = int(np.flatnonzero(np.isclose(z, fj.Z_NORM, rtol=0, atol=1e-12))[0])
    d, v = growth(parameters, k, z, steps)
    return parameters[:, 9, None]*v[:, np.arange(17), kmap[:17]]/d[:, j_norm, kmap[-1], None]


def table_range(name, columns):
    """Minimum and maximum over the listed columns of a figure_data table."""
    with open(OUT / (name + '.csv')) as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row[c]) for row in rows for c in columns]
    return min(values), max(values)


def write_table(name, columns):
    lengths = {len(np.atleast_1d(value)) for value in columns.values()}
    assert len(lengths) == 1, name
    np.savetxt(OUT / (name + '.csv'), np.column_stack(list(columns.values())), delimiter=',',
               header=','.join(columns), comments='', fmt='%.12g')


def quantiles(array):
    return np.percentile(array, PERCENTILES, axis=0)


def band_table(name, x, samples, xname='z', extras=None):
    q = quantiles(samples)
    columns = {xname: x, 'lo95': q[0], 'lo68': q[1], 'median': q[2], 'hi68': q[3], 'hi95': q[4]}
    if extras:
        columns.update(extras)
    write_table(name, columns)
    return q


def basic_validation():
    points = np.linspace(0, XG[-1]+.05, 300)
    worst = 0
    for power in range(4):
        interpolator = NotAKnotSpline(X_NODES, X_NODES**power)
        worst = max(worst, float(np.max(np.abs(interpolator(points)-points**power))))
    cardinal_error = float(np.max(np.abs(BASIS(X_NODES)-np.eye(7))))
    # Independent closed-form growing mode for pressureless Einstein-de Sitter.
    d = np.ones((1, 1)); v = np.ones_like(d)
    interval = np.log(3.33)
    for _ in range(512):
        d, v = rk4_step(d, v, interval/512, .5, .5, .5, 0., 0., 0.)
    eds_error = float(abs(d[0, 0]/3.33-1))
    assert worst < 1e-12 and cardinal_error < 1e-12 and eds_error < 1e-10
    return {'cubic_polynomial_max_error': worst, 'spline_cardinality_max_error': cardinal_error,
            'pressureless_EdS_relative_error': eds_error}


def compute(draws=4096):
    """Stage 1: figure tables from the stored posterior that fit_joint.py loads."""
    OUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    chain = fj.posterior_chain()
    obs = load_observations()
    check = basic_validation()
    indices = posterior_indices(len(chain), draws, SEED)
    sample = chain[indices]
    write_table('posterior_indices', {'chain_row': indices})

    # Full-chain normalized reconstruction; no subsampling and no median-history approximation.
    zz = np.unique(np.r_[np.linspace(.295, 1.484, 201), Z_NODES[:6], .51])
    inverse = hubble_at(chain, zz)
    q = -1 - chain[:, :7] @ BASIS(np.log1p(zz)).T
    cs2 = chain[:, 8, None]
    h2 = (1 / (C_OVER_H0*inverse))**2
    prefactor = 2*(1+q)*np.sqrt(cs2)/(1-cs2)*h2 / (1+zz)**3
    potential = (3-2*(1+q)/(1-cs2))*h2
    anchor = int(np.flatnonzero(zz == .51)[0])
    prefactor /= prefactor[:, anchor, None]
    potential /= potential[:, anchor, None]
    fq = band_table('reconstructed_coupling', zz, prefactor)
    vq = band_table('reconstructed_potential', zz, potential)
    assert np.max(np.abs(prefactor[:, anchor]-1)) == 0 and np.max(np.abs(potential[:, anchor]-1)) == 0
    node_idx = np.searchsorted(zz, Z_NODES[:6])
    write_table('reconstruction_nodes', {'z': Z_NODES[:6], 'coupling': fq[2, node_idx], 'potential': vq[2, node_idx]})
    check['reconstruction_node_medians'] = {'coupling': fq[2, node_idx].tolist(), 'potential': vq[2, node_idx].tolist()}
    del prefactor, potential, inverse, q, h2
    print('Full-chain reconstructed functions tabulated.', flush=True)

    hist, edges = np.histogram(np.log10(chain[:, 8]), bins=np.linspace(-12, -5.8, 94), density=True)
    # PGFPlots ybar interval uses left edges plus one terminal edge.
    write_table('sound_speed_density', {'u': edges, 'density': np.r_[hist, 0.]})
    floors = np.array([1e-12, 1e-10, 1e-9, 1e-8])
    limits = np.array([np.percentile(chain[chain[:, 8] >= floor, 8], 95) for floor in floors])
    write_table('prior_sensitivity', {'log_floor': np.log10(floors), 'upper95': limits, 'upper95_scaled': limits/1e-7})
    check['cs2_upper95'] = float(limits[0])
    check['prior_floor_upper95'] = dict(zip([f'{v:.0e}' for v in floors], limits.tolist()))

    # Jeans estimate of Eq. (32) [Eq. (C12)]: (aH/k_max)^2 (5+2q)/4 at every growth redshift,
    # with the k_max of that measurement's window, over the full chain; aH/k = E/(d_H (1+z) k).
    zg, kmax = obs['growth_z'], obs['growth_kmax']
    inverse = hubble_at(chain, zg)
    q = -1 - chain[:, :7] @ BASIS(np.log1p(zg)).T
    jeans = (1/(C_OVER_H0*inverse*(1+zg)*kmax))**2*(5+2*q)/4
    lo68, med, hi68 = np.percentile(jeans, [16, 50, 84], axis=0)
    i = int(np.argmin(med))
    write_table('jeans_estimate', {'z': zg, 'kmax': kmax, 'lo68': lo68, 'median': med, 'hi68': hi68})
    check['jeans_estimate_minimum'] = {'z': float(zg[i]), 'kmax': float(kmax[i]), 'median': float(med[i]),
                                       'lo68': float(lo68[i]), 'hi68': float(hi68[i]),
                                       'median_over_cs2_upper95': float(med[i]/limits[0])}
    del inverse, q, jeans

    # Matched illustrative initial values; bounded analytic bare solution, no clipping.
    dn = np.linspace(0, 3., 401)
    initial, ceiling = 1e-10, 1e-7
    bare = initial*np.exp(6*dn)/(1-initial+initial*np.exp(6*dn))
    coupled = ceiling/(1+(ceiling/initial-1)*np.exp(-3*dn))
    write_table('mechanism', {'dn': dn, 'bare': bare, 'coupled': coupled,
                              'ceiling': np.full_like(dn, ceiling)})
    assert np.isclose(bare[0], coupled[0], rtol=1e-14, atol=0)
    check['mechanism'] = {'initial_cs2': initial, 'ceiling_cs2': ceiling, 'A': ceiling/initial-1}

    # Integrator convergence at high-sound-speed tail samples and random draws.
    validation_rows = np.unique(np.r_[indices[:24], np.argsort(chain[:, 8])[-8:]])
    p = chain[validation_rows]
    pred800 = growth_predictions(p, obs, 800)
    pred1600 = growth_predictions(p, obs, 1600)
    pred3200 = growth_predictions(p, obs, 3200)
    sigma_g = np.sqrt(np.diag(obs['growth_cov']))
    error = float(np.max(abs(pred1600-pred3200)/sigma_g))
    check['growth_step_refinement'] = {'samples': len(p), 'max_800_1600_sigma': float(np.max(abs(pred800-pred1600)/sigma_g)),
                                      'max_1600_3200_sigma': error}
    # Same samples through the fit's own growth integration (fit_joint.chi2_growth path).
    fit_pred = np.empty_like(pred1600)
    for n, par in enumerate(p):
        eps, amplitude, cs2_nodes, sigma8, _ = fj.unpack(par, 1)
        inv_h, _, _, q_grid = fj.background(eps, amplitude)
        solution, modes = fj.growth(fj.cs2_on_grid(cs2_nodes, fj.node_positions(1)), q_grid, inv_h, amplitude)
        norm = int(np.argmin(np.abs(solution.t + np.log1p(fj.Z_NORM))))
        q_norm = float(np.interp(np.log1p(fj.Z_NORM), fj.XG, q_grid))
        d_reference = solution.y[modes - 1, norm]/(2*(1+q_norm)/3)     # D = y/(1+w) at z=Z_NORM
        for i in range(fj.NG):
            t_index = int(np.argmin(np.abs(solution.t + np.log1p(fj.ZG_D[i]))))
            fit_pred[n, i] = sigma8*solution.y[modes + fj.K_IDX[i], t_index]/d_reference
    check['fit_solver_comparison_max_sigma'] = float(np.max(abs(pred1600-fit_pred)/sigma_g))
    assert check['fit_solver_comparison_max_sigma'] < 1e-2, check['fit_solver_comparison_max_sigma']
    assert error < 1e-3, check['growth_step_refinement']
    print(f'Growth step refinement: maximum error {error:.3g} observational sigma.', flush=True)

    # All three fitted datasets. Output curves use one common random joint-posterior subset.
    zcurve = np.unique(np.r_[np.linspace(.05, 2.33, 151), obs['sn_z'], ANISO[:, 0], BGS[0]])
    dm_store, dh_store, sn_store, gp_store = [], [], [], []
    for first in range(0, len(sample), 256):
        pars = sample[first:first+256]
        inverse, dm, dh = background(pars)
        dm_z = interpolate_rows(dm, zcurve)
        dm_store.append(dm_z)
        dh_store.append(interpolate_rows(dh, zcurve))
        sn_store.append(5*np.log10((1+zcurve)*dm_z)+pars[:, -1, None])
        gp_store.append(growth_predictions(pars, obs))
        print(f'Observable predictions: {min(first+256, len(sample))}/{len(sample)} samples.', flush=True)
    dm, dh, mu, gp = [np.concatenate(items) for items in (dm_store, dh_store, sn_store, gp_store)]
    dv = np.cbrt(zcurve[None, :]*dm**2*dh)
    q_dm = band_table('bao_dm_curve', zcurve, dm)
    q_dh = band_table('bao_dh_curve', zcurve, dh)
    q_mu = band_table('supernova_curve', zcurve, mu)
    ibao = np.searchsorted(zcurve, ANISO[:, 0]); iv = int(np.searchsorted(zcurve, BGS[0]))
    isn = np.searchsorted(zcurve, obs['sn_z'])
    for name, values, sigma, obs_y in [('bao_dm', dm[:, ibao], ANISO[:, 2], ANISO[:, 1]),
                                     ('bao_dh', dh[:, ibao], ANISO[:, 4], ANISO[:, 3]),
                                     ('supernova', mu[:, isn], np.sqrt(np.diag(obs['sn_cov'])), obs['sn_y']),
                                     ('growth', gp, sigma_g, obs['growth_y'])]:
        z = ANISO[:, 0] if name.startswith('bao') else (obs['sn_z'] if name == 'supernova' else obs['growth_z'])
        median = np.median(values, axis=0)
        extras = {'observed': obs_y, 'sigma': sigma, 'residual': (obs_y-median)/sigma}
        if name == 'growth':
            extras['keff'] = obs['growth_k']
        band_table(name+'_points', z, values, extras=extras)
    qdv = quantiles(dv[:, [iv]])
    band_table('bao_dv_points', np.array([BGS[0]]), dv[:, [iv]],
               extras={'observed': np.array([BGS[1]]), 'sigma': np.array([BGS[2]]),
                       'residual': np.array([(BGS[1]-qdv[2, 0])/BGS[2]])})

    # Covariance-aware chi2 values are audited, not used to claim an optimized minimum.
    chi_bao = ((dv[:, iv]-BGS[1])/BGS[2])**2
    for j, row in enumerate(ANISO):
        covariance = np.array([[row[2]**2, row[5]*row[2]*row[4]], [row[5]*row[2]*row[4], row[4]**2]])
        residual = np.column_stack([row[1]-dm[:, ibao[j]], row[3]-dh[:, ibao[j]]])
        chi_bao += np.einsum('bi,ij,bj->b', residual, np.linalg.inv(covariance), residual)
    rsn, rg = obs['sn_y']-mu[:, isn], obs['growth_y']-gp
    chi_sn = np.einsum('bi,ij,bj->b', rsn, np.linalg.inv(obs['sn_cov']), rsn)
    chi_g = np.einsum('bi,ij,bj->b', rg, np.linalg.inv(obs['growth_cov']), rg)
    total = chi_bao+chi_sn+chi_g
    best = int(np.argmin(total))
    check['best_in_plotted_random_subset'] = {'chain_row': int(indices[best]), 'samples_searched': len(sample),
                                             'chi2': float(total[best]), 'bao': float(chi_bao[best]),
                                             'supernova': float(chi_sn[best]), 'growth': float(chi_g[best]),
                                             'optimized': False}
    write_table('subset_chi2', {'chain_row': indices, 'bao': chi_bao, 'supernova': chi_sn, 'growth': chi_g, 'total': total})

    check['posterior_sha256'] = hashlib.sha256(np.ascontiguousarray(chain, dtype='<f8').tobytes()).hexdigest()
    check['observational_data_sha256'] = fj.data_hashes()
    check['samples_reconstruction_and_prior'] = len(chain)
    check['samples_observables'] = len(sample)
    check['numerics'] = {'backend': 'independent NumPy not-a-knot spline + RK4 of the exact y-V system', 'background_grid': NGRID,
                         'RK4_base_steps': 1600}
    check['figures_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    check['numpy_version'] = np.__version__
    check['elapsed_seconds'] = time.perf_counter()-start
    (OUT / 'numerical_validation.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check, indent=2), flush=True)
    return check



counter = 0


def draw(CHECK):
    """Stage 2: PGFPlots sources for the figures, labels taken from CHECK."""
    global counter
    TEX.mkdir(exist_ok=True)
    STYLE = r'''\documentclass[tikz,border=4pt]{standalone}
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage{amsmath}
    \usepackage{pgfplots}
    \usepgfplotslibrary{groupplots,fillbetween}
    \usetikzlibrary{arrows.meta}
    \pgfplotsset{compat=1.18}
    \definecolor{udmblue}{HTML}{21618C}
    \definecolor{udmorange}{HTML}{B75519}
    \definecolor{udmpurple}{HTML}{765589}
    \definecolor{udmgray}{HTML}{545B63}
    \pgfplotsset{every axis/.append style={
      font=\fontsize{9.5}{11}\selectfont,
      label style={font=\fontsize{10}{12}\selectfont},
      tick label style={font=\fontsize{9}{10}\selectfont},
      title style={font=\fontsize{10}{12}\selectfont,align=left},
      axis line style={black!65},axis x line*=bottom,axis y line*=left,
      tick align=outside,tick style={black!60},
      major tick length=2.5pt,minor tick length=1.2pt,
      grid=none,scaled ticks=false,
      legend style={draw=none,fill=white,fill opacity=.92,text opacity=1,
        font=\fontsize{9}{10.5}\selectfont,cells={anchor=west},inner sep=2pt},
      legend cell align=left,
      every axis plot/.append style={line join=round},
    }}
    \begin{document}
    '''
    (TEX / 'plot_style.tex').write_text(STYLE, encoding='utf-8')


    counter = 0


    def bands(name, color='udmblue', x='z', median=True):
        global counter
        counter += 1
        key = f'b{counter}'
        result = ''
        for level, opacity in [('95', 12), ('68', 30)]:
            for side in ('lo', 'hi'):
                result += rf'\addplot[draw=none,forget plot,name path={key}{side}{level}] table[col sep=comma,x={x},y={side}{level}] {{../figure_data/{name}.csv}};'+'\n'
            result += rf'\addplot[draw=none,fill={color}!{opacity},forget plot] fill between[of={key}lo{level} and {key}hi{level}];'+'\n'
        if median:
            result += rf'\addplot[{color},line width=1.2pt,forget plot] table[col sep=comma,x={x},y=median] {{../figure_data/{name}.csv}};'+'\n'
        return result


    def data_points(name, color='black', mark='*', size='1.8pt'):
        return rf'''\addplot[only marks,{color},mark={mark},mark size={size},
          error bars/.cd,y dir=both,y explicit,error bar style={{line width=.55pt}},error mark options={{mark size=1.5pt}}]
          table[col sep=comma,x=z,y=observed,y error=sigma] {{../figure_data/{name}.csv}};
    '''


    def prediction_points(name, color='udmblue'):
        return rf'''\addplot[only marks,{color}!45,mark=none,line width=2pt,
          error bars/.cd,y dir=both,y explicit,error bar style={{line width=2pt}},error mark=none]
          table[col sep=comma,x=z,y=median,y error plus expr=\thisrow{{hi95}}-\thisrow{{median}},
          y error minus expr=\thisrow{{median}}-\thisrow{{lo95}}] {{../figure_data/{name}.csv}};
    \addplot[only marks,{color},mark=square*,mark size=1.6pt,
          error bars/.cd,y dir=both,y explicit,error bar style={{line width=1.6pt}},error mark=none]
          table[col sep=comma,x=z,y=median,y error plus expr=\thisrow{{hi68}}-\thisrow{{median}},
          y error minus expr=\thisrow{{median}}-\thisrow{{lo68}}] {{../figure_data/{name}.csv}};
    '''


    def residual(name, color='black', mark='*'):
        return rf'''\addplot[only marks,{color},mark={mark},mark size=1.8pt,
          error bars/.cd,y dir=both,y explicit,error bar style={{line width=.5pt}},error mark options={{mark size=1.4pt}}]
          table[col sep=comma,x=z,y=residual,y error expr=1] {{../figure_data/{name}.csv}};
    '''


    def write_figure(name, body):
        (TEX / (name+'.tex')).write_text('\\input{plot_style.tex}\n'+body+'\n\\end{document}\n', encoding='utf-8')


    upper = CHECK['cs2_upper95']

    body = r'''\begin{tikzpicture}
    \begin{axis}[width=9.2cm,height=7.8cm,
      xmin=0,xmax=3,ymode=log,ymin=1e-11,ymax=2e-2,
      xtick={0,1,2,3},ytick={1e-10,1e-8,1e-6,1e-4,1e-2},
      xlabel={Expansion interval $\Delta N=\ln(a/a_i)$},ylabel={Sound speed squared $c_s^2$},
      legend pos=north west,legend style={font=\fontsize{8.8}{10}\selectfont}]
    \addplot[udmgray,dashed,line width=1.3pt] table[col sep=comma,x=dn,y=bare] {../figure_data/mechanism.csv};
    \addlegendentry{Constant prefactor, potential term neglected}
    \addplot[udmblue,line width=1.5pt] table[col sep=comma,x=dn,y=coupled] {../figure_data/mechanism.csv};
    \addlegendentry{Field-dependent prefactor, prescribed history}
    \addplot[udmblue,densely dotted,line width=.6pt,forget plot] coordinates {(0,1e-7) (3,1e-7)};
    \node[anchor=east,text=udmblue,font=\fontsize{9}{10}\selectfont] at (axis cs:2.98,2.1e-7) {$c_\infty^2=10^{-7}$};
    \addplot[only marks,black,mark=*,mark size=2pt,forget plot] coordinates {(0,1e-10)};
    \node[anchor=west,font=\fontsize{9}{10}\selectfont] at (axis cs:.08,2e-11) {Same initial $c_s^2=10^{-10}$};
    '''
    body += '\\end{axis}\n\\end{tikzpicture}'
    write_figure('fig01_mechanism', body)

    body = r'''\begin{tikzpicture}
    \begin{groupplot}[group style={group size=3 by 2,horizontal sep=38pt,vertical sep=13pt},
      width=5.95cm,xmin=0,xmax=2.43,xtick={0,1,2},
      tick label style={font=\fontsize{9}{10}\selectfont},
      legend style={font=\fontsize{8.5}{9.5}\selectfont}]
    \nextgroupplot[height=5.8cm,ymin=0,ymax='''+f'{1.15*max(table_range("bao_dm_curve",["hi95"])[1], table_range("bao_dm_points",["observed"])[1]):.1f}'+r''',ytick={0,10,20,30,40},xticklabels=\empty,
      title={(a) DESI DR2 BAO},ylabel={Distance / $r_d$},legend pos=north west]
    '''
    body += bands('bao_dm_curve') + bands('bao_dh_curve','udmorange')
    body += data_points('bao_dm_points','udmblue') + '\\addlegendentry{$D_M/r_d$}\n'
    body += data_points('bao_dh_points','udmorange','triangle*','2.2pt') + '\\addlegendentry{$D_H/r_d$}\n'
    body += data_points('bao_dv_points','udmpurple','diamond*','2.4pt') + '\\addlegendentry{$D_V/r_d$}\n'
    mu_lo, mu_hi = table_range('supernova_points', ['observed'])
    body += r'''\nextgroupplot[height=5.8cm,ymin='''+f'{mu_lo-1.2:.1f},ymax={mu_hi+1.0:.1f}'+r''',ytick={36,39,42,45},xticklabels=\empty,
      title={(b) Union3 supernovae},ylabel={Binned distance modulus}]
    '''+ bands('supernova_curve') + data_points('supernova_points','black','*','1.5pt')
    g_lo = min(table_range('growth_points', ['lo95'])[0], table_range('growth_points', ['observed'])[0] - .11)
    g_hi = max(table_range('growth_points', ['hi95'])[1], table_range('growth_points', ['observed'])[1] + .18)
    body += r'''\nextgroupplot[height=5.8cm,ymin='''+f'{g_lo-.05:.2f},ymax={g_hi+.08:.2f}'+r''',ytick={.2,.4,.6},xticklabels=\empty,
      title={(c) Growth compilation},ylabel={$f_g\sigma_8$},legend pos=north east]
    '''+prediction_points('growth_points')
    body += r'\addlegendentry{95\% model interval}'+'\n'+r'\addlegendentry{68\% model interval}'+'\n'
    body += data_points('growth_points','black','o','2.2pt') + '\\addlegendentry{Measurements}\n'
    for names, yl in [([('bao_dm_points','udmblue','*'),('bao_dh_points','udmorange','triangle*'),('bao_dv_points','udmpurple','diamond*')],True),
                       ([('supernova_points','black','*')],False),([('growth_points','black','o')],False)]:
        body += r'''\nextgroupplot[height=3.45cm,ymin=-3,ymax=3,ytick={-2,0,2},xlabel={Redshift $z$}'''
        body += ',ylabel={$\\Delta/\\sigma$}' if yl else ''
        body += r''']
    \path[fill=black!5] (axis cs:0,-1) rectangle (axis cs:2.43,1);
    \addplot[black!50,densely dashed,forget plot] coordinates {(0,0) (2.43,0)};
    '''
        body += ''.join(residual(*item) for item in names)
    body += '\\end{groupplot}\n\\end{tikzpicture}'
    write_figure('fig02_observations', body)

    body = r'''\begin{tikzpicture}
    \begin{groupplot}[group style={group size=2 by 1,horizontal sep=50pt},width=8.5cm,height=7cm]
    \nextgroupplot[xmin=-12,xmax=-5.55,ymin=0,ymax='''+f'{1.12*table_range("sound_speed_density",["density"])[1]:.3f}'+r''',
      xtick={-12,-10,-8,-6},yticklabel style={/pgf/number format/fixed},
      xlabel={$u=\log_{10}c_s^2$},ylabel={Posterior density $p(u\mid\mathrm{data})$},
      title={(a) An upper limit on sound speed},legend pos=north west]
    '''
    body += r'''\addplot[ybar interval,fill=udmblue!35,draw=udmblue!60,line width=.3pt,forget plot]
      table[col sep=comma,x=u,y=density] {../figure_data/sound_speed_density.csv};
    '''
    body += rf'''\addplot[udmblue,line width=1.3pt] coordinates {{({math.log10(upper)},0) ({math.log10(upper)},{1.12*table_range("sound_speed_density",["density"])[1]:.3f})}};
    \addlegendentry{{95\% upper limit: ${upper/1e-7:.2f}\times10^{{-7}}$}}
    '''
    body += r'''\node[anchor=south west,font=\fontsize{9}{11}\selectfont,align=left,fill=white,inner sep=2pt]
      at (axis cs:-11.8,.012) {Logarithmic prior:\\$10^{-12}\leq c_s^2\leq10^{-2}$};
    \nextgroupplot[xmin=-12.45,xmax=-7.55,ymin='''+f'{min(CHECK["prior_floor_upper95"].values())/1e-7-.45:.2f},ymax={max(CHECK["prior_floor_upper95"].values())/1e-7+.45:.2f}'+r''',
      xtick={-12,-10,-9,-8},xticklabels={$10^{-12}$,$10^{-10}$,$10^{-9}$,$10^{-8}$},
      xlabel={Lower cutoff of the logarithmic prior},
      ylabel={95\% upper limit on $c_s^2$ ($10^{-7}$)},title={(b) Dependence on the prior cutoff}]
    \addplot[udmblue,dashed,line width=1pt,mark=*,mark size=2.5pt]
      table[col sep=comma,x=log_floor,y=upper95_scaled] {../figure_data/prior_sensitivity.csv};
    '''
    for floor, value in CHECK['prior_floor_upper95'].items():
        body += rf'\node[anchor=south,font=\fontsize{{9}}{{11}}\selectfont] at (axis cs:{math.log10(float(floor))},{value/1e-7+.07}) {{{value/1e-7:.2f}}};'+'\n'
    body += r'''\node[anchor=south east,font=\fontsize{8.8}{10.5}\selectfont,align=right]
      at (rel axis cs:0.97,0.04) {Same stored posterior,\\conditioned on each cutoff};
    \end{groupplot}
    \end{tikzpicture}'''
    write_figure('fig03_sound_speed', body)

    body = r'''\begin{tikzpicture}
    \begin{groupplot}[group style={group size=2 by 1,horizontal sep=50pt},width=8.5cm,height=7cm,
      xmin=.27,xmax=1.51,xtick={.3,.6,.9,1.2,1.48},xlabel={Redshift $z$},
      legend pos=north west]
    \nextgroupplot[ymin='''+f'{table_range("reconstructed_coupling",["lo95"])[0]-.08:.2f},ymax={table_range("reconstructed_coupling",["hi95"])[1]+.12:.2f}'+r''',ylabel={Normalized $f(z)a^3(z)$},
      title={(a) Reconstructed kinetic prefactor}]
    '''+bands('reconstructed_coupling')
    body += r'''\addplot[udmorange,dashed,line width=1pt] coordinates {(.295,1) (1.484,1)};
    \addlegendentry{Reference: $f\propto a^{-3}$}
    \addlegendimage{area legend,fill=udmblue!30,draw=none}\addlegendentry{68\% pointwise interval}
    \addlegendimage{area legend,fill=udmblue!12,draw=none}\addlegendentry{95\% pointwise interval}
    \addplot[udmblue,only marks,mark=*,mark size=1.8pt,forget plot]
     table[col sep=comma,x=z,y=coupling] {../figure_data/reconstruction_nodes.csv};
    \draw[black!40,densely dotted] ({axis cs:.51,1}|-{rel axis cs:0,0}) -- (axis cs:.51,1.15);
    \addplot[black,only marks,mark=*,mark size=2.2pt,forget plot] coordinates {(.51,1)};
    \node[anchor=west,font=\fontsize{9}{10}\selectfont] at (axis cs:.53,1.1) {$z_\star=0.51$};
    \nextgroupplot[ymin='''+f'{min(0., table_range("reconstructed_potential",["lo95"])[0]-.1):.2f},ymax={table_range("reconstructed_potential",["hi95"])[1]+.15:.2f}'+r''',ylabel={Normalized $V(z)$},
      title={(b) Reconstructed potential},legend pos=north west]
    '''+bands('reconstructed_potential')
    body += r'''\addplot[udmorange,dashed,line width=1pt] coordinates {(.295,1) (1.484,1)};
    \addlegendentry{Reference: constant $V$}
    \addplot[udmblue,only marks,mark=*,mark size=1.8pt,forget plot]
     table[col sep=comma,x=z,y=potential] {../figure_data/reconstruction_nodes.csv};
    \draw[black!40,densely dotted] ({axis cs:.51,1}|-{rel axis cs:0,0}) -- (axis cs:.51,1.4);
    \addplot[black,only marks,mark=*,mark size=2.2pt,forget plot] coordinates {(.51,1)};
    \node[anchor=west,font=\fontsize{9}{10}\selectfont] at (axis cs:.53,1.32) {$z_\star=0.51$};
    \end{groupplot}
    \end{tikzpicture}'''
    write_figure('fig04_reconstructed_functions', body)



def render():
    """Stage 3: compile each figure source with pdflatex into FIGURES."""
    FIGURES.mkdir(exist_ok=True)
    for source in sorted(TEX.glob('fig[0-9][0-9]_*.tex')):
        print(f'Compiling {source.name}', flush=True)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                        '-output-directory', str(FIGURES), source.name],
                       cwd=TEX, check=True, stdout=subprocess.DEVNULL)
    for extra in FIGURES.glob('*'):
        if extra.suffix in ('.aux', '.log'):
            extra.unlink()


if __name__ == '__main__':
    check = compute()
    draw(check)
    render()
    print(f'Figures written to {FIGURES}')
