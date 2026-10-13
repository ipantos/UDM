"""Two assumptions of the growth comparison, tested on the stored results.

Keep this file next to extensions.py (which needs fit_joint.py, figures.py and
udm_data.txt beside it).  It imports extensions.py and reads extensions.json and changes
neither.  Its numbers go to assumptions.json and its tables to figure_data/asm_*.csv.
Run `extensions.py lcdm`, `bestfit`, `profile` and `grids` first.  Times are for one core of
a machine that was running other jobs.
The grids of `tracer` and `scales` are kept in assumptions_grids/ and read from there when
a stage is repeated; delete that folder to compute them again.

    python3 assumptions.py tracer    whose velocity the growth data measure             5 min
    python3 assumptions.py tracer_checks   the same with pressure acting before z = 2.33,
                                     with the other wavenumber at z = 0.07, and with
                                     a larger pressureless share                       29 min
    python3 assumptions.py scales    other wavenumbers for the growth measurements,
                                     subsets of them, and another value at z = 0.07    13 min
    python3 assumptions.py window    profile likelihood of the 11-parameter model with
                                     another wavenumber for the measurement at z = 0.07 10 min
    python3 assumptions.py tracer_general   profile likelihood of the 11-parameter model
                                     with the velocity of a pressureless component     53 min
    python3 assumptions.py figure    tables for the growth-response figure               1 s

TRACER.  The fit compares the growth data with the velocity of the scalar fluid, which
its own pressure gradient decelerates.  Pressureless matter responds to the gravitational
potential only, and galaxies may move as such matter does.  This stage evolves the scalar
together with a pressureless component, coupled by gravity, in the sub-horizon limit
(N = ln a, ' = d/dN):

    y'   = 3 w_u y + (1 + w_u) v                       scalar, y = delta rho_u/rho_u
    v'   = -(1 - q) v + S - c_s^2 k^2 y/[(1 + w_u) a^2 H^2]
    d_b' = v_b                                         pressureless component
    v_b' = -(1 - q) v_b + S
    S    = (3/2) (Omega_u y + Omega_b d_b)

q is the deceleration parameter of the total, Omega_b = rho_b/3H^2, Omega_u = 1 - Omega_b
and w_u = p/rho_u.  With Omega_b = 0 the first two equations are the system of the fit
(Eq. (30) of the paper; the system above is its Eq. (A10)) and v_b is the velocity of a
test particle in the potential of the scalar.  Both components start at z = 2.33 with the
same amplitude and rate, and the prediction is sigma_8 x velocity(k_i, z_i)/D_u(k=0, z_ref),
as in the fit.  The models are the two of `extensions.py grids`, which have a
LambdaCDM-like expansion:

    'constant'  LambdaCDM expansion, constant c_s^2 = 10^u, rate v/D = 1 at z = 2.33
    'minimal'   constant f and V, c_s^2(z=0) = 10^u, growing-mode rate at z = 2.33

Omega is the matter-like density today in units of the critical density (the pressureless
component plus the diluting part of the scalar); a fraction FB of it is pressureless.

TRACER_CHECKS.  Changes to the `tracer` stage, one at a time.  (1) In the fit no pressure
has acted before z = 2.33: all wavenumbers start there with the same amplitude.  Here the
constant sound speed acts from an earlier redshift, z = 10 or 100, where both components
start in the LambdaCDM growing mode; the minimal model is started at z = 10 in the same
way.  (2) The measurement at z = 0.07 is given the interval 0.025 < k < 0.1 h/Mpc, as in
`scales`, together with the pressureless velocity.  (3) The pressureless share of the
matter-like density is 30%, 50% or 70% in place of FB.  For each case the stage records
the lowest rise of chi^2 at sound speeds above the limit: where it is below 2.71 (as for
a share of 70%) large sound speeds are allowed again and there is no upper limit.  The
stage also stores, at the limits obtained with the pressureless velocity, the velocities
and the source S of the potential at the growth measurements divided by their values for
zero sound speed.

SCALES. The same two models with alternative wavenumbers or selected measurements
left out. The baseline already uses Qin's power-spectrum value 0.440 +/- 0.088.
The historical `power spectrum alone` result is an alias of the baseline, not a
second measurement. The six DESI amplitudes remain unchanged; their historical
leave-out diagnostic is not run by default (optional --legacy-desi-diagnostic).

WINDOW.  The profile likelihood of `extensions.py profile` (seven nodes of the expansion
history, constant c_s^2, V >= 0) when the measurement at z = 0.07 is given the interval
0.025 < k < 0.1 h/Mpc of its power-spectrum analysis in place of 0.025 < k < 0.3.

TRACER_GENERAL.  The same profile likelihood when the growth data are compared with the
velocity of a pressureless component, as in `tracer`, in place of that of the scalar.
Three cases.  (1) A test particle: the scalar is the only component, as in the fit, and
the expansion history and its restrictions are unchanged.  (2) A pressureless component
with density OMEGA_B = FB x Omega_m today, Omega_m being that of the LambdaCDM fit: the
spline is the deceleration parameter of the total, and the equation of state of the
scalar is w_u = w_total/(1 - Omega_b).  (3) As (2), with the interval 0.025 < k < 0.1
h/Mpc for the measurement at z = 0.07, as in `window`.  The restrictions on the nodes are
those of the fit, -1 < q <= (1 - 3 c_s^2)/2 on the grid.  For the scalar of cases (2) and
(3), V >= 0 reads q <= [1 - 3 c_s^2 (1 - Omega_b)]/2, which the restriction imposed
implies, and rho_u + p_u > 0 reads 1 + q > (3/2) Omega_b.  The latter is not imposed: the
smallest value of (1 + q) - (3/2) Omega_b on the grid is stored for every point of the
profile (column `scalar_margin`), and the sound speeds at which it is not positive are
listed; there the minimum found lies outside the model, and with the condition imposed
chi^2 could only be larger.  The chi^2 is evaluated with the RK4 solver only.  The
profile is followed to c_s^2 = 0.01, the upper edge of the prior of the fit, and the
lowest rise of chi^2 above the limit is recorded as in `tracer_checks`.

Limits are defined as in extensions.py: profile limits where the smallest chi^2 has risen
by 2.71 (one-sided 95% for a Gaussian likelihood) or 3.84 above its minimum; posterior
limits are 95th percentiles for a prior uniform in u from -12 and in Omega.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import extensions as ext  # noqa: E402
import figures as G  # noqa: E402
import fit_joint as fj  # noqa: E402

OUT_JSON = ROOT / 'assumptions.json'
GRIDS = ROOT / 'assumptions_grids'
ZG = ext.ZG
GZ, GK = ext.GZ, ext.GK
FB = 0.02237/0.1430          # Planck 2018 (TT,TE,EE+lowE+lensing): Omega_b h^2 / Omega_m h^2
U_WIDE = np.r_[-12., -10., -9., np.arange(-8.5, -0.999, .05)]
OMEGA_GRID = ext.OMEGA_GRID
GROWING_MODE = {'constant': False, 'minimal': True}       # the conventions of the paper's table
LEVEL95 = '95% one-sided (Delta chi2 = 2.71)'


def store(key, value):
    data = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    data[key] = value
    OUT_JSON.write_text(json.dumps(data, indent=2))


def table(name, columns):
    ext.OUT.mkdir(exist_ok=True)
    G.write_table('asm_' + name, columns)


# ----------------------------------------------------------------------------
# backgrounds with a pressureless fraction, and the two-fluid growth system
# ----------------------------------------------------------------------------
def background(model, solver, omega, c0, fb):
    """H0/H on the grid ZG; q, H0/H, c_s^2 and Omega_b on the points of the solver.
    omega: matter-like density today / critical; c0: c_s^2 (constant, or today); fb: the
    pressureless fraction of omega."""
    om, c0 = np.atleast_1d(omega)[:, None], np.atleast_1d(c0)[:, None]
    ob0, ok = fb*om, (1 - fb)*om
    if model == 'constant':
        def at(z):
            rho = om*(1 + z)**3 + 1 - om
            return rho, -(1 - om) + 0*z, c0 + 0*z
    else:
        of, oc = ok*np.sqrt(c0), ok*np.sqrt(1 - c0)
        ol = 1 - ob0 - ok

        def at(z):
            kin = np.sqrt(of**2 + oc**2*(1 + z)**6)
            return ob0*(1 + z)**3 + kin + ol, -of**2/kin - ol, of**2/kin**2
    rho, p, cs2 = at(solver.z)
    return (1/np.sqrt(at(ZG)[0]), (1 + 3*p/rho)/2, 1/np.sqrt(rho), cs2, ob0*(1 + solver.z)**3/rho)


def two_fluid(solver, q_pts, inv_pts, cs2_pts, ob_pts, f_ini=1.0):
    """The system of the module docstring, fixed-step RK4 on the points of an
    extensions.Solver.  Returns D_u = y/(1+w_u), v, d_b, v_b, 1+w_u and Omega_b at the
    output redshifts for every wavenumber: arrays (n, nz+1, nk); the last output row is
    the normalisation redshift."""
    n, nt, nk = len(q_pts), len(solver.times), len(solver.k_all)
    pk = (np.broadcast_to(cs2_pts, q_pts.shape)*(G.C_OVER_H0*inv_pts*np.exp(solver.x))**2)
    kval = np.square(solver.k_all)[None, :]
    opw = 1 + ((2*q_pts - 1)/3)/(1 - ob_pts)

    def rates(state, i):
        y, v, d, vb = state
        q, ob, w1 = q_pts[:, i, None], ob_pts[:, i, None], opw[:, i, None]
        source = 1.5*((1 - ob)*y + ob*d)
        return (3*(w1 - 1)*y + w1*v, -(1 - q)*v + source - pk[:, i, None]*kval*y/w1,
                vb, -(1 - q)*vb + source)

    def shifted(state, slope, h):
        return tuple(s + h*k for s, k in zip(state, slope))
    rate = np.ones((n, nk))*np.reshape(f_ini, (-1, 1))
    state = (np.ones((n, nk))*opw[:, 0, None], rate, np.ones((n, nk)), rate.copy())
    out = np.empty((6, n, len(solver.z_out) + 1, nk))

    def record(i):
        for j in solver.dest.get(i, []):
            out[0, :, j], out[1, :, j] = state[0]/opw[:, i, None], state[1]
            out[2, :, j], out[3, :, j] = state[2], state[3]
            out[4, :, j], out[5, :, j] = opw[:, i, None], ob_pts[:, i, None]
    record(0)
    for i, dt in enumerate(np.diff(solver.times)):
        k1 = rates(state, i)
        k2 = rates(shifted(state, k1, .5*dt), nt + i)
        k3 = rates(shifted(state, k2, .5*dt), nt + i)
        k4 = rates(shifted(state, k3, dt), i + 1)
        state = tuple(s + dt*(a + 2*b + 2*c + d)/6 for s, a, b, c, d in zip(state, k1, k2, k3, k4))
        record(i + 1)
    return out


def shapes(solver, fields):
    """Growth prediction for sigma_8 = 1 from the output of two_fluid, for the velocity of
    the scalar, of the pressureless component, and for their momentum-weighted mean; and
    the source S of the gravitational potential in the same normalisation."""
    d_u, v, d_b, vb, opw, ob = fields
    m = len(solver.z_out)
    pick = lambda a: a[:, np.arange(m), solver.kmap[:m]]
    norm = d_u[:, -1, solver.kmap[-1], None]
    weight_u = pick(opw)*(1 - pick(ob))
    return {'scalar': pick(v)/norm, 'pressureless': pick(vb)/norm,
            'mass_weighted': (weight_u*pick(v) + pick(ob)*pick(vb))/(weight_u + pick(ob))/norm,
            'potential': 1.5*(weight_u*pick(d_u) + pick(ob)*pick(d_b))/norm}


# ----------------------------------------------------------------------------
# grids in (u, Omega) and their analysis
# ----------------------------------------------------------------------------
def grid_chi2(model, solver, fb, rows=None, wanted=('scalar',), u_grid=U_WIDE, cache=None, start=None):
    """Profile chi^2 over (A, sigma_8, M) on the grid (u, Omega), for each velocity in
    `wanted`.  With fb = 0 and only the scalar wanted, the one-fluid solver of
    extensions.py is used; the two agree (checked in `tracer`).  cache: name of a file in
    assumptions_grids/ that keeps the result; delete that folder to compute again.
    start: redshift at which the solver starts, if not 2.33; the rate there is that of the
    LambdaCDM growing mode."""
    # Cache identity includes the actual data, source code and solver settings.
    import hashlib
    identity = hashlib.sha256()
    for source in ('udm_data.txt', 'fit_joint.py', 'figures.py', 'extensions.py', 'assumptions.py'):
        identity.update((ROOT/source).read_bytes())
    identity.update(repr((model, fb, rows, wanted, start)).encode())
    for array in (u_grid, OMEGA_GRID, ext.GY, ext.GCOV, solver.k_all, solver.kmap, solver.times, solver.z_out):
        identity.update(np.ascontiguousarray(array).tobytes())
    path = None if cache is None else GRIDS / (cache + '_' + identity.hexdigest()[:16] + '.npz')
    if path is not None and path.exists():
        saved = np.load(path)
        return {name: {key[len(name) + 1:]: saved[key] for key in saved.files if key.startswith(name + '/')}
                for name in wanted}
    uu, oo = [a.ravel() for a in np.meshgrid(u_grid, OMEGA_GRID, indexing='ij')]
    growing = GROWING_MODE[model] or start is not None
    rate_table = ext.lcdm_growing_mode(OMEGA_GRID, **({} if start is None else {'z': start})) if growing else None
    parts = {name: [] for name in wanted}
    for first in range(0, len(uu), 1000):
        om, c0 = oo[first:first + 1000], 10.0**uu[first:first + 1000]
        inv_grid, q_pts, inv_pts, cs2_pts, ob_pts = background(model, solver, om, c0, fb)
        rate = np.interp(om, OMEGA_GRID, rate_table) if growing else 1.0
        if fb == 0 and tuple(wanted) == ('scalar',) and start is None:
            found = {'scalar': solver.shape(q_pts, inv_pts, cs2_pts, rate)}
        else:
            found = shapes(solver, two_fluid(solver, q_pts, inv_pts, cs2_pts, ob_pts, rate))
        for name in wanted:
            parts[name].append(ext.linear_profile(inv_grid, found[name], rows))
    size = (len(u_grid), len(OMEGA_GRID))
    result = {name: {key: np.concatenate([p[key] for p in part]).reshape(size) for key in part[0]}
              for name, part in parts.items()}
    if path is not None:
        GRIDS.mkdir(exist_ok=True)
        np.savez(path, **{f'{name}/{key}': value for name, grid in result.items() for key, value in grid.items()})
    return result


def analyse(grid, model, fb=0., u_grid=U_WIDE):
    """As analyse_grid of extensions.py, for any grid of u: profile over Omega (parabola
    through the three lowest grid values), marginal likelihood for a flat prior on Omega
    (and on A, sigma_8, M), and the upper limits."""
    chi2 = grid['chi2']
    rows = np.arange(len(u_grid))
    j = np.clip(np.argmin(chi2, axis=1), 1, chi2.shape[1] - 2)
    y0, y1, y2 = chi2[rows, j - 1], chi2[rows, j], chi2[rows, j + 1]
    curvature = y0 - 2*y1 + y2
    profile = y1 - .125*(y0 - y2)**2/curvature
    step = OMEGA_GRID[1] - OMEGA_GRID[0]
    omega_best = OMEGA_GRID[j] + .5*(y0 - y2)/curvature*step
    lowest = int(np.argmin(profile))
    u_lowest, reference = ext.parabola_minimum(u_grid, profile)
    delta = profile - reference
    weight = np.exp(-.5*(chi2 - chi2.min()) + grid['logvol'] - grid['logvol'][0, j[0]])
    marginal = weight.sum(axis=1)*step
    marginal = marginal/marginal[0]
    out = {'chi2_minimum': float(reference), 'c_s2_at_minimum': float(10.0**u_lowest),
           'omega_at_minimum': float(omega_best[lowest]),
           'sigma8_ref_at_minimum': float(grid['sigma8'][lowest, j[lowest]]),
           'chi2_at_zero_sound_speed': float(profile[0]),
           'delta_chi2_at_zero_sound_speed': float(delta[0]),
           'delta_chi2_at_grid_end': float(delta[-1]), 'profile_upper_limits': {}}
    for name, level in ext.LEVELS.items():
        cross = ext.crossing(u_grid[lowest:], delta[lowest:], level)
        out['profile_upper_limits'][name] = None if cross is None else 10.0**cross
    out['bayes_95_upper_limit'] = 10.0**ext.upper_limit(u_grid, marginal, -12.)
    limit = out['profile_upper_limits'][LEVEL95]
    if limit is not None:
        out['omega_at_95_limit'] = float(np.interp(np.log10(limit), u_grid, omega_best))
        out['sigma8_ref_at_95_limit'] = float(np.interp(np.log10(limit), u_grid, grid['sigma8'][rows, j]))
    if model == 'minimal':
        # profile likelihood of f/3H0^2 = c_s(0) x (kinetic density today/critical)
        log_f = np.arange(-4.6, -1.2999, .01)
        columns = np.full((len(log_f), len(OMEGA_GRID)), np.inf)
        for col, omega in enumerate(OMEGA_GRID):
            u_here = 2*(log_f - np.log10((1 - fb)*omega))
            inside = (u_here >= u_grid[0]) & (u_here <= u_grid[-1])
            columns[inside, col] = PchipInterpolator(u_grid, chi2[:, col])(u_here[inside])
        jf = np.clip(np.argmin(columns, axis=1), 1, len(OMEGA_GRID) - 2)
        f0, f1, f2 = (columns[np.arange(len(log_f)), jf + shift] for shift in (-1, 0, 1))
        with np.errstate(all='ignore'):
            profile_f = np.where(np.isfinite(f0 + f1 + f2), f1 - .125*(f0 - f2)**2/(f0 - 2*f1 + f2), np.inf)
        finite = np.isfinite(profile_f)
        log_f, profile_f = log_f[finite], profile_f[finite]
        start = int(np.argmin(profile_f))
        cross = ext.crossing(log_f[start:], profile_f[start:] - reference, ext.LEVELS[LEVEL95])
        out['profile_95_upper_limit_omega_f'] = None if cross is None else 10.0**cross
        ratio = (1 + fj.Z_NORM)**6
        out['profile_95_upper_limit_c_s2_at_z_ref'] = (None if limit is None else
                                                       float(limit/(limit + (1 - limit)*ratio)))
    return out, {'u': u_grid, 'value': 10.0**u_grid, 'delta_chi2': delta, 'chi2': profile,
                 'omega': omega_best, 'sigma8_ref': grid['sigma8'][rows, j]}


def spot_values(grid, summary, u_grid=U_WIDE):
    """chi^2 at single grid points near the minimum and the limit, for the cross-check."""
    out = []
    limit = summary['profile_upper_limits'][LEVEL95]
    for target in (1e-9, summary['c_s2_at_minimum'], limit):
        if target is None:
            continue
        iu = int(np.argmin(np.abs(u_grid - np.log10(max(target, 1e-12)))))
        jo = int(np.argmin(grid['chi2'][iu]))
        out.append({'u': float(u_grid[iu]), 'omega': float(OMEGA_GRID[jo]),
                    **{key: float(grid[key][iu, jo]) for key in ('chi2', 'bao', 'sn', 'growth', 'sigma8')}})
    return out


# ----------------------------------------------------------------------------
# stage: whose velocity
# ----------------------------------------------------------------------------
TRACER_CASES = {  # name: (pressureless fraction, velocity compared with the data)
    'scalar, one fluid (the fit)': (0., 'scalar'),
    'scalar, with a pressureless fraction': (FB, 'scalar'),
    'momentum-weighted mean': (FB, 'mass_weighted'),
    'pressureless fraction': (FB, 'pressureless'),
    'test particle in the potential of the scalar': (0., 'pressureless'),
}


def command_tracer():
    reference = json.loads(ext.OUT_JSON.read_text())['grids']
    result = {'pressureless_fraction': FB, 'u_grid': [float(U_WIDE[0]), float(U_WIDE[-1])]}
    for model in ('constant', 'minimal'):
        result[model] = {}
        label = 'growing_mode_initial_rate' if GROWING_MODE[model] else 'same_initial_rate'
        base = reference[model][label]['all_data']
        grids = {}
        for fb in (0., FB):
            wanted = sorted({velocity for f, velocity in TRACER_CASES.values() if f == fb})
            grids[fb] = grid_chi2(model, ext.FINE, fb, wanted=wanted, cache=f'tracer_{model}_{"two" if fb else "one"}')
            print(f'  {model}: grid for pressureless fraction {fb:.4f} done', flush=True)
        for name, (fb, velocity) in TRACER_CASES.items():
            summary, columns = analyse(grids[fb][velocity], model, fb)
            summary['spot_values'] = spot_values(grids[fb][velocity], summary)
            result[model][name] = summary
            tag = name.split(',')[0].replace(' ', '_') + ('_fb' if fb else '')
            table(f'tracer_{model}_{tag}', columns)
            print(model, '|', name, json.dumps({k: v for k, v in summary.items() if k != 'spot_values'}, indent=1),
                  flush=True)
        # the two-fluid solver with no pressureless fraction against the stored one-fluid grids
        mine = result[model]['scalar, one fluid (the fit)']
        result[model]['check_against_extensions'] = {
            'chi2_minimum': [mine['chi2_minimum'], base['chi2_minimum']],
            'limit_95': [mine['profile_upper_limits'][LEVEL95], base['profile_upper_limits'][LEVEL95]],
            'bayes_95': [mine['bayes_95_upper_limit'], base['bayes_95_upper_limit_by_lower_prior_edge']['1e-12']]}
        print(model, 'check', json.dumps(result[model]['check_against_extensions']), flush=True)
        # how much each velocity is lowered at the lowest redshift, at fixed amplitude
        omega = base['omega_at_minimum']
        limits = {'one_fluid_limit': mine['profile_upper_limits'][LEVEL95],
                  'pressureless_limit': result[model]['pressureless fraction']['profile_upper_limits'][LEVEL95]}
        c0 = np.r_[1e-30, [v for v in limits.values() if v is not None]]
        rate = float(ext.lcdm_growing_mode(omega)[0]) if GROWING_MODE[model] else 1.0
        _, q_pts, inv_pts, cs2_pts, ob_pts = background(model, ext.FINE, np.full(len(c0), omega), c0, FB)
        found = shapes(ext.FINE, two_fluid(ext.FINE, q_pts, inv_pts, cs2_pts, ob_pts, rate))
        result[model]['velocity_relative_to_zero_sound_speed'] = {
            'omega': omega, 'c_s2': dict(zip(limits, [float(v) for v in c0[1:]])),
            'z': GZ.tolist(), 'k_eff': GK.tolist(),
            **{f'{velocity}_at_{key}': (found[velocity][n + 1]/found[velocity][0]).tolist()
               for n, key in enumerate(k for k, v in limits.items() if v is not None)
               for velocity in ('scalar', 'pressureless', 'potential')}}
        low = result[model]['velocity_relative_to_zero_sound_speed']
        print(model, 'velocity at z = %.2f, k = %.3f relative to c_s = 0:' % (GZ[0], GK[0]),
              {k: round(v[0], 4) for k, v in low.items() if k.startswith(('scalar_at', 'pressureless_at'))}, flush=True)
    store('tracer', result)


# ----------------------------------------------------------------------------
# stage: the pressureless velocity with an earlier start, and with the other wavenumber
# ----------------------------------------------------------------------------
class EarlyStart:
    """The integration points of an extensions.Solver for a start at z_start > 2.33, with
    the same number of steps per e-fold.  For the analytic backgrounds of this file only."""

    def __init__(self, z_start, base=None):
        base = ext.FINE if base is None else base
        self.z_out, self.k_all, self.kmap = base.z_out, base.k_all, base.kmap
        t_out = -np.log1p(np.r_[self.z_out, fj.Z_NORM])
        steps = int(np.ceil(1600*np.log1p(z_start)/ext.XG[-1]))
        self.times = np.unique(np.r_[np.linspace(-np.log1p(z_start), 0, steps + 1), t_out])
        self.x = -np.r_[self.times, .5*(self.times[1:] + self.times[:-1])]
        self.z = np.expm1(self.x)
        self.dest = {}
        for j, t in enumerate(t_out):
            self.dest.setdefault(int(np.searchsorted(self.times, t)), []).append(j)


def response(model, solver, omega, c_values, fb, rate):
    """Velocity of the scalar, velocity of the pressureless component and source S of the
    potential at the growth measurements, each divided by its value for zero sound speed;
    one row per entry of c_values, at fixed Omega, rate and amplitude."""
    c0 = np.r_[1e-30, c_values]
    _, q_pts, inv_pts, cs2_pts, ob_pts = background(model, solver, np.full(len(c0), omega), c0, fb)
    found = shapes(solver, two_fluid(solver, q_pts, inv_pts, cs2_pts, ob_pts, rate))
    return {name: (found[name][1:]/found[name][0]).tolist() for name in ('scalar', 'pressureless', 'potential')}


def command_tracer_checks():
    tracer = json.loads(OUT_JSON.read_text())['tracer']
    result = {'pressureless_fraction': FB, 'earlier_start': {}, 'other_wavenumber_at_z_0.07': {}}

    def report(label, grid, model, fb=0.):
        summary, columns = analyse(grid, model, fb)
        summary['spot_values'] = spot_values(grid, summary)
        # an upper limit holds only if the profile stays above the level at larger sound speeds
        limit = summary['profile_upper_limits'][LEVEL95]
        beyond = columns['delta_chi2'][columns['u'] > np.log10(limit)] if limit is not None else []
        summary['lowest_delta_chi2_above_the_limit'] = float(np.min(beyond)) if len(beyond) else None
        print(label, json.dumps({k: v for k, v in summary.items() if k != 'spot_values'}, indent=1), flush=True)
        return summary
    # (1) constant sound speed acting since z_start, LambdaCDM expansion
    for z_start in (10., 100.):
        solver = EarlyStart(z_start)
        one = grid_chi2('constant', solver, 0., wanted=('scalar', 'pressureless'), start=z_start,
                        cache=f'checks_start{z_start:.0f}_one')
        two = grid_chi2('constant', solver, FB, wanted=('scalar', 'pressureless'), start=z_start,
                        cache=f'checks_start{z_start:.0f}_two')
        cases = {'scalar, one fluid': one['scalar'], 'pressureless fraction': two['pressureless'],
                 'test particle in the potential of the scalar': one['pressureless']}
        result['earlier_start'][f'{z_start:.0f}'] = {
            name: report(f'start at z = {z_start:.0f} | {name}', grid, 'constant') for name, grid in cases.items()}
    # the same code started at z = 2.33 with v/D = 1 reproduces the `tracer` stage
    check = analyse(grid_chi2('constant', EarlyStart(fj.Z_INI), FB, wanted=('pressureless',),
                              cache='checks_start2p33_two')['pressureless'], 'constant')[0]
    base = tracer['constant']['pressureless fraction']
    result['check_against_tracer'] = {'limit_95': [check['profile_upper_limits'][LEVEL95], base['profile_upper_limits'][LEVEL95]],
                                      'chi2_minimum': [check['chi2_minimum'], base['chi2_minimum']]}
    print('check', json.dumps(result['check_against_tracer']), flush=True)
    # (2) the other wavenumber at z = 0.07, with the pressureless velocity
    k_values = GK.copy()
    k_values[0] = K_LOW_ALT
    solver = ext.Solver(1600, k_out=k_values)
    for model in ('constant', 'minimal'):
        grid = grid_chi2(model, solver, FB, wanted=('pressureless',), cache=f'checks_window_{model}')['pressureless']
        result['other_wavenumber_at_z_0.07'][model] = report(f'other wavenumber | {model}', grid, model, FB)
    # (1b) the minimal model started at z = 10: its sound speed is negligible before z = 2.33
    solver = EarlyStart(10.)
    one = grid_chi2('minimal', solver, 0., start=10., wanted=('scalar',), cache='checks_start10_minimal_one')
    two = grid_chi2('minimal', solver, FB, start=10., wanted=('pressureless',), cache='checks_start10_minimal_two')
    result['earlier_start_minimal'] = {'10': {
        'scalar, one fluid': report('minimal model, start at z = 10 | scalar, one fluid', one['scalar'], 'minimal'),
        'pressureless fraction': report('minimal model, start at z = 10 | pressureless fraction',
                                        two['pressureless'], 'minimal', FB)}}
    # (3) a larger pressureless share of the matter-like density
    result['other_pressureless_share'] = {}
    for share in (.30, .50, .70):
        result['other_pressureless_share'][f'{share:.2f}'] = {}
        for model in ('constant', 'minimal'):
            grid = grid_chi2(model, ext.FINE, share, wanted=('pressureless',),
                             cache=f'checks_share{100*share:.0f}_{model}')['pressureless']
            result['other_pressureless_share'][f'{share:.2f}'][model] = report(
                f'pressureless share {share:.2f} | {model}', grid, model, share)
    # (4) what the scalar and the potential do at the limits for pressureless galaxies
    result['response_at_pressureless_limit'] = {'z': GZ.tolist(), 'k_eff': GK.tolist(), 'constant': {}, 'minimal': {}}
    omega = tracer['constant']['velocity_relative_to_zero_sound_speed']['omega']
    for label, solver, start, limit in (
            ('2.33', ext.FINE, None, base['profile_upper_limits'][LEVEL95]),
            ('10', EarlyStart(10.), 10., result['earlier_start']['10']['pressureless fraction']['profile_upper_limits'][LEVEL95]),
            ('100', EarlyStart(100.), 100., result['earlier_start']['100']['pressureless fraction']['profile_upper_limits'][LEVEL95])):
        rate = 1.0 if start is None else float(ext.lcdm_growing_mode(omega, z=start)[0])
        found = response('constant', solver, omega, [limit], FB, rate)
        result['response_at_pressureless_limit']['constant'][label] = {
            'omega': omega, 'c_s2': limit, **{name: values[0] for name, values in found.items()}}
    omega = tracer['minimal']['velocity_relative_to_zero_sound_speed']['omega']
    limit = tracer['minimal']['pressureless fraction']['profile_upper_limits'][LEVEL95]
    found = response('minimal', ext.FINE, omega, [limit], FB, float(ext.lcdm_growing_mode(omega)[0]))
    result['response_at_pressureless_limit']['minimal']['2.33'] = {
        'omega': omega, 'c_s2': limit, **{name: values[0] for name, values in found.items()}}
    stored = tracer['constant']['velocity_relative_to_zero_sound_speed']
    mine = result['response_at_pressureless_limit']['constant']['2.33']
    assert np.allclose(mine['potential'], stored['potential_at_pressureless_limit'], rtol=1e-12, atol=0)
    low = {start: {name: round(case[name][0], 3) for name in ('scalar', 'pressureless', 'potential')}
           for start, case in result['response_at_pressureless_limit']['constant'].items()}
    print('response at z = %.2f, k = %.3f, constant sound speed:' % (GZ[0], GK[0]), json.dumps(low), flush=True)
    store('tracer_checks', result)


# ----------------------------------------------------------------------------
# stage: other wavenumbers, and subsets of the growth data
# ----------------------------------------------------------------------------
def growth_windows():
    """k_min and k_max of the 17 measurements, read as figures.py reads them."""
    rows = [row for row in csv.DictReader(fj.DATA['growth']) if row['space'] == 'fourier' and row['k_max']]
    order = np.argsort([float(row['z_eff']) for row in rows])
    lower = np.array([float(row['k_min']) if row['k_min'] else 0. for row in rows])[order]
    upper = np.array([float(row['k_max']) for row in rows])[order]
    assert np.allclose(np.round(np.sqrt(.6*(upper**5 - lower**5)/(upper**3 - lower**3)), 6), GK)
    return lower, upper


K_LOW_ALT = float(np.round(np.sqrt(.6*(.1**5 - .025**5)/(.1**3 - .025**3)), 6))   # 0.078034
FULL_SHAPE_Z = (.295, .510, .706, .919, 1.317, 1.491)             # the six DESI DR1 full-shape amplitudes
POWER_SPECTRUM_ALONE = (.440, .5*(.080 + .096))                   # value and error at z = 0.07, Qin et al., Sec. 6.2
OTHER_DATUM = 'z = 0.07 from the power spectrum alone'


class other_first_datum:
    """Within the block, the growth measurement at z = 0.07 has another value and error."""

    def __init__(self, value, sigma):
        self.value, self.sigma = value, sigma

    def __enter__(self):
        assert GZ[0] == .07 and np.count_nonzero(ext.GCOV[0]) == 1      # the first measurement, uncorrelated
        self.saved = ext.GY, ext.GCOV, ext.GCINV
        gy, cov = ext.GY.copy(), ext.GCOV.copy()
        gy[0], cov[0, 0] = self.value, self.sigma**2
        ext.GY, ext.GCOV, ext.GCINV = gy, cov, np.linalg.inv(cov)

    def __exit__(self, *exc):
        ext.GY, ext.GCOV, ext.GCINV = self.saved


def scale_cases():
    lower, upper = growth_windows()
    every = np.arange(len(GZ))
    alt = GK.copy()
    alt[0] = K_LOW_ALT
    assert GZ[0] == .07
    full_shape = np.array([int(np.argmin(np.abs(GZ - z))) for z in FULL_SHAPE_Z])
    assert np.allclose(GZ[full_shape], FULL_SHAPE_Z)
    low_high_k = np.flatnonzero((GZ < .4) & (GK > .2))
    cases = {  # name: (wavenumbers, rows kept)
        'as in the fit': (GK, None),
        'z = 0.07 with 0.025 < k < 0.1': (alt, None),
        'midpoint of each interval': (np.round(.5*(lower + upper), 6), None),
        'upper end of each interval': (upper, None),
        'k_eff capped at 0.155': (np.minimum(GK, .155), None),
        'without z = 0.07': (GK, np.setdiff1d(every, [0])),
        'without the six DESI full-shape amplitudes': (GK, np.setdiff1d(every, full_shape)),
        'without the three measurements with k_eff = 0.23 at z < 0.4': (GK, np.setdiff1d(every, low_high_k)),
        OTHER_DATUM: (GK, None),
    }
    if '--legacy-desi-diagnostic' not in sys.argv:
        cases.pop('without the six DESI full-shape amplitudes')
    return cases


def command_scales():
    result = {'k_eff_z_0.07_alternative': K_LOW_ALT,
              'power_spectrum_alone_at_z_0.07': {'value': POWER_SPECTRUM_ALONE[0], 'sigma': POWER_SPECTRUM_ALONE[1]}}
    for model in ('constant', 'minimal'):
        result[model] = {}
        for number, (name, (k_values, rows)) in enumerate(scale_cases().items()):
            solver = ext.FINE if k_values is GK else ext.Solver(1600, k_out=k_values)
            if name == OTHER_DATUM:
                result[model][name] = dict(result[model]['as in the fit'])
                result[model][name]['alias_of'] = 'as in the fit'
                continue
            else:
                grid = grid_chi2(model, solver, 0., rows=rows, cache=f'scales_{model}_{number}')['scalar']
            summary, columns = analyse(grid, model)
            summary['measurements'] = int(fj.NDATA) - (0 if rows is None else len(GZ) - len(rows))
            summary['spot_values'] = spot_values(grid, summary)
            result[model][name] = summary
            print(model, '|', name, json.dumps({k: v for k, v in summary.items() if k != 'spot_values'}, indent=1),
                  flush=True)
    store('scales', result)


# ----------------------------------------------------------------------------
# stage: the 11-parameter model with another wavenumber for the lowest redshift
# ----------------------------------------------------------------------------
U_WINDOW = np.r_[-12., -9., -8.5, -8., -7.75, -7.5, np.arange(-7.4, -5.499, .05)]


def command_window():
    """As `extensions.py leaveout`, with all 17 measurements kept and the wavenumber of the
    first one changed.  The chi^2 is evaluated with the RK4 solver only."""
    k_values = GK.copy()
    k_values[0] = K_LOW_ALT
    kept = np.arange(len(GZ))
    saved = ext.FAST, ext.FINE
    ext.FAST, ext.FINE = ext.Solver(400, k_out=k_values), ext.Solver(1600, k_out=k_values)
    try:
        best_all = json.loads(ext.OUT_JSON.read_text())['bestfit']['with_V_nonnegative']
        anchors = ext.starting_points(4, 0.) + [np.array(best_all['eps_nodes'])]
        found = []
        for start in anchors:                              # minimum with the sound speed free
            for first in (best_all['c_s2'], 5e-7, 1e-9):
                eps, cs2, value = ext.minimise_nodes(ext.moved_inside(start, first), first, rows=kept, free_cs2=True)
                if ext.feasible(eps, cs2):
                    found.append((value, eps, cs2))
        found.sort(key=lambda item: item[0])
        best = ext.describe(found[0][1], found[0][2], rows=kept)
        rows, previous = [], found[0][1]
        for u in U_WINDOW:
            cs2 = 10.0**u
            trials = []
            for start in [previous] + anchors:
                eps, _, value = ext.minimise_nodes(ext.moved_inside(start, cs2), cs2, rows=kept)
                if ext.feasible(eps, cs2):
                    trials.append((value, eps))
            trials.sort(key=lambda item: item[0])
            previous = trials[0][1]
            info = ext.describe(previous, cs2, rows=kept)
            info['u'], info['spread_of_starts'] = float(u), float(trials[-1][0] - trials[0][0])
            rows.append(info)
            print(f'  u = {u:6.2f}: chi2 = {info["chi2_rk4_solver"]:.4f} sigma8 = {info["sigma8_ref"]:.4f} '
                  f'starts {len(trials)} spread {info["spread_of_starts"]:.1e}', flush=True)
    finally:
        ext.FAST, ext.FINE = saved
    u = np.array([r['u'] for r in rows])
    chi2 = np.array([r['chi2_rk4_solver'] for r in rows])
    minimum = min(float(chi2.min()), best['chi2_rk4_solver'])
    delta = chi2 - minimum
    table('profile_other_window', {'u': u, 'cs2': 10.0**u, 'chi2': chi2, 'delta_chi2': delta,
                                   'sigma8_ref': [r['sigma8_ref'] for r in rows],
                                   **{f'eps{i}': [r['eps_nodes'][i] for r in rows] for i in range(7)}})
    lowest = int(np.argmin(chi2))
    limits = {}
    for name, level in ext.LEVELS.items():
        cross = ext.crossing(u[lowest:], delta[lowest:], level)
        limits[name] = None if cross is None else 10.0**cross
    result = {'k_eff_z_0.07': K_LOW_ALT, 'measurements': int(fj.NDATA), 'chi2_minimum': minimum,
              'c_s2_at_minimum': best['c_s2'], 'sigma8_ref_at_minimum': best['sigma8_ref'],
              'chi2_bao_sn_growth_at_minimum': [best['chi2_bao'], best['chi2_sn'], best['chi2_growth']],
              'eps_nodes_at_minimum': best['eps_nodes'],
              'starts_for_the_minimum': len(found),
              'starts_within_0.001_of_minimum': int(sum(item[0] - found[0][0] < 1e-3 for item in found)),
              'delta_chi2_at_zero_sound_speed': float(delta[0]), 'upper_limits_c_s2': limits,
              'largest_spread_between_starts_where_delta_chi2_below_10': float(max(
                  r['spread_of_starts'] for r, dlt in zip(rows, delta) if dlt < 10))}
    print(json.dumps(result, indent=1))
    store('window', result)


# ----------------------------------------------------------------------------
# stage: the 11-parameter model with the velocity of a pressureless component
# ----------------------------------------------------------------------------
class TracerSolver(ext.Solver):
    """An extensions.Solver whose growth prediction is the velocity of the pressureless
    component of the two-fluid system.  omega_b: density of that component today in units
    of the critical density (0: a test particle in the potential of the scalar)."""

    def __init__(self, steps, omega_b=0., k_out=None):
        super().__init__(steps, k_out=k_out)
        self.omega_b = omega_b

    def shape(self, q_pts, inv_pts, cs2_pts, f_ini=1.0):
        ob_pts = self.omega_b*(1 + self.z)**3*inv_pts**2
        return shapes(self, two_fluid(self, q_pts, inv_pts, cs2_pts, ob_pts, f_ini))['pressureless']


U_GENERAL = np.r_[-12., -9., -8., -7.5, -7., -6.75, -6.5, -6.25, np.arange(-6., -4.999, .05),
                  -4.75, -4.5, -4.25, -4., -3.5, -3., -2.5, -2.]
U_GENERAL_HIGH = np.r_[-12., -9., -8., -7.5, -7., -6.75, -6.5, -6.25, -6., -5.75, np.arange(-5.5, -4.499, .05),
                       -4.25, -4., -3.5, -3., -2.5, -2.]            # for case (3), whose limit is higher


def command_tracer_general():
    lcdm = json.loads(ext.OUT_JSON.read_text())['lcdm']['same_initial_rate']
    best_all = json.loads(ext.OUT_JSON.read_text())['bestfit']['with_V_nonnegative']
    omega_b = FB*lcdm['omega_m']
    kept = np.arange(len(GZ))
    k_alt = GK.copy()
    k_alt[0] = K_LOW_ALT
    result = {'pressureless_fraction': FB, 'omega_b': omega_b, 'k_eff_z_0.07_alternative': K_LOW_ALT,
              'u_grid': [float(U_GENERAL[0]), float(U_GENERAL[-1])]}
    assert U_GENERAL_HIGH[0] == U_GENERAL[0] and U_GENERAL_HIGH[-1] == U_GENERAL[-1]
    cases = (('test particle in the potential of the scalar', 0., None, 'test_particle', U_GENERAL),
             ('pressureless component', omega_b, None, 'pressureless', U_GENERAL),
             ('pressureless component, other wavenumber at z = 0.07', omega_b, k_alt, 'pressureless_other_window',
              U_GENERAL_HIGH))
    for label, ob, k_out, file_name, u_grid in cases:
        saved = ext.FAST, ext.FINE
        ext.FAST, ext.FINE = TracerSolver(400, ob, k_out), TracerSolver(1600, ob, k_out)

        def scalar_margin(eps):
            """Smallest value on the grid of (1 + q) - (3/2) Omega_b: positive where rho_u + p_u > 0."""
            grid = G.SMAT @ np.asarray(eps)
            return float(np.min(-grid - 1.5*ob*np.exp(3*ext.XG)*np.exp(2*(G.IMAT @ np.asarray(eps)))))
        try:
            anchors = ext.starting_points(4, 0.) + [np.array(best_all['eps_nodes'])]
            found = []
            for start in anchors:                              # minimum with the sound speed free
                for first in (best_all['c_s2'], 2e-6, 1e-9):
                    eps, cs2, value = ext.minimise_nodes(ext.moved_inside(start, first), first, rows=kept, free_cs2=True)
                    if ext.feasible(eps, cs2):
                        found.append((value, eps, cs2))
            found.sort(key=lambda item: item[0])
            if not found:
                raise RuntimeError(f'{label}: no admissible free-sound-speed minimum')
            best = ext.describe(found[0][1], found[0][2], rows=kept)
            rows, previous = [], found[0][1]
            for u in u_grid:
                cs2 = 10.0**u
                trials = []
                for start in [previous] + anchors:
                    # at large sound speeds the line searches try nodes far outside the
                    # restrictions (w < -1), where the integration overflows; such points are
                    # rejected by the search, and a minimum is kept only if it is inside
                    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                        eps, _, value = ext.minimise_nodes(ext.moved_inside(start, cs2), cs2, rows=kept)
                    if ext.feasible(eps, cs2):
                        trials.append((value, eps))
                if not trials:
                    raise SystemExit(f'{label}: no start ended inside the restrictions at u = {u}')
                trials.sort(key=lambda item: item[0])
                previous = trials[0][1]
                info = ext.describe(previous, cs2, rows=kept)
                info['u'], info['spread_of_starts'] = float(u), float(trials[-1][0] - trials[0][0])
                info['starts'], info['scalar_margin'] = len(trials), scalar_margin(previous)
                rows.append(info)
                # Preserve completed points even if a later profile point fails.
                checkpoint = ROOT / ('tracer_general_' + file_name + '_partial.json')
                temporary = checkpoint.with_suffix('.tmp')
                temporary.write_text(json.dumps({'complete': False, 'case': label,
                                                 'best': best, 'rows': rows}, indent=2))
                temporary.replace(checkpoint)
                print(f'  {label} u = {u:6.2f}: chi2 = {info["chi2_rk4_solver"]:.4f} sigma8 = {info["sigma8_ref"]:.4f} '
                      f'starts {len(trials)} spread {info["spread_of_starts"]:.1e}', flush=True)
        finally:
            ext.FAST, ext.FINE = saved
        u = np.array([r['u'] for r in rows])
        chi2 = np.array([r['chi2_rk4_solver'] for r in rows])
        minimum = min(float(chi2.min()), best['chi2_rk4_solver'])
        delta = chi2 - minimum
        table('profile_general_' + file_name,
              {'u': u, 'cs2': 10.0**u, 'chi2': chi2, 'delta_chi2': delta,
               'sigma8_ref': [r['sigma8_ref'] for r in rows], 'starts': [r['starts'] for r in rows],
               'scalar_margin': [r['scalar_margin'] for r in rows],
               **{f'eps{i}': [r['eps_nodes'][i] for r in rows] for i in range(7)}})
        lowest = int(np.argmin(chi2))
        limits = {}
        for level_name, level in ext.LEVELS.items():
            cross = ext.crossing(u[lowest:], delta[lowest:], level)
            limits[level_name] = None if cross is None else 10.0**cross
        limit = limits[LEVEL95]
        beyond = delta[u > np.log10(limit)] if limit is not None else []
        result[label] = {
            'omega_b': ob, 'measurements': int(fj.NDATA), 'chi2_minimum': minimum,
            'c_s2_at_minimum': best['c_s2'], 'sigma8_ref_at_minimum': best['sigma8_ref'],
            'chi2_bao_sn_growth_at_minimum': [best['chi2_bao'], best['chi2_sn'], best['chi2_growth']],
            'eps_nodes_at_minimum': best['eps_nodes'], 'starts_for_the_minimum': len(found),
            'starts_within_0.001_of_minimum': int(sum(item[0] - found[0][0] < 1e-3 for item in found)),
            'delta_chi2_at_zero_sound_speed': float(delta[0]), 'delta_chi2_at_grid_end': float(delta[-1]),
            'upper_limits_c_s2': limits,
            'lowest_delta_chi2_above_the_limit': float(np.min(beyond)) if len(beyond) else None,
            'largest_spread_between_starts_where_delta_chi2_below_10': float(max(
                r['spread_of_starts'] for r, dlt in zip(rows, delta) if dlt < 10)),
            'fewest_starts_inside_the_restrictions': int(min(r['starts'] for r in rows)),
            'smallest_scalar_margin_along_the_profile': float(min(r['scalar_margin'] for r in rows)),
            'smallest_scalar_margin_where_delta_chi2_below_10': float(min(
                r['scalar_margin'] for r, dlt in zip(rows, delta) if dlt < 10)),
            'c_s2_where_the_scalar_margin_is_not_positive': [r['c_s2'] for r in rows if r['scalar_margin'] <= 0],
            'delta_chi2_where_the_scalar_margin_is_not_positive': [
                float(dlt) for r, dlt in zip(rows, delta) if r['scalar_margin'] <= 0],
            'scalar_margin_at_minimum': scalar_margin(best['eps_nodes'])}
        checkpoint.write_text(json.dumps({'complete': True, 'case': label, 'best': best, 'rows': rows}, indent=2))
        print(label, json.dumps(result[label], indent=1), flush=True)
    store('tracer_general', result)


# ----------------------------------------------------------------------------
# stage: tables for the growth-response figure
# ----------------------------------------------------------------------------
def command_figure():
    """The response of a pressureless component to the sound speed, for the two panels of
    the growth-response figure (extensions.py figure): velocity at the profile limit of
    each model divided by the velocity for zero sound speed, at the largest group of
    wavenumbers.  Panel (a) uses the best-fitting expansion history of the spline model,
    with rho_b/3H0^2 = FB x Omega_m of the LambdaCDM fit; panel (b) the minimal model with
    the same total matter-like density as in extensions.py."""
    data = json.loads(ext.OUT_JSON.read_text())
    zc = np.linspace(2.33, 0., 234)
    k = .232413
    solver = ext.Solver(1600, z_out=zc, k_out=np.full(len(zc), k))

    def ratios(q_pts, inv_pts, cs2_pts, ob_pts, rate):
        found = shapes(solver, two_fluid(solver, q_pts, inv_pts, cs2_pts, ob_pts, rate))
        return {'z': zc, 'scalar': found['scalar'][1]/found['scalar'][0],
                'pressureless': found['pressureless'][1]/found['pressureless'][0]}
    # (a) constant sound speed on the best-fitting history of the spline model
    eps = np.array(data['bestfit']['with_V_nonnegative']['eps_nodes'])
    limit = data['profile']['with_V_nonnegative']['upper_limits_c_s2'][LEVEL95]
    omega_b = FB*data['lcdm']['same_initial_rate']['omega_m']
    _, q_pts, inv_pts = solver.spline(np.repeat(eps[None, :], 2, axis=0))
    table('response_constant', ratios(q_pts, inv_pts, np.array([[0.], [limit]]),
                                      omega_b*(1 + solver.z)**3*inv_pts**2, 1.0))
    # (b) constant f and V
    minimal = data['grids']['minimal']['growing_mode_initial_rate']['all_data']
    omega = minimal['omega_at_zero_sound_speed']
    limit_minimal = minimal['profile_upper_limits'][LEVEL95]
    _, q_pts, inv_pts, cs2_pts, ob_pts = background('minimal', solver, np.full(2, omega),
                                                    np.array([1e-30, limit_minimal]), FB)
    table('response_minimal', ratios(q_pts, inv_pts, cs2_pts, ob_pts, float(ext.lcdm_growing_mode(omega)[0])))
    store('figure', {'wavenumber': k, 'constant_sound_speed': limit, 'omega_b_spline': omega_b,
                     'minimal_model_value_today': limit_minimal, 'omega_minimal': omega,
                     'pressureless_fraction': FB})


if __name__ == '__main__':
    commands = {name[8:]: value for name, value in globals().items() if name.startswith('command_')}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit('usage: python3 assumptions.py {' + '|'.join(commands) + '}')
    start_time = time.time()
    commands[sys.argv[1]]()
    print(f'[{sys.argv[1]}] done in {time.time() - start_time:.1f} s')
