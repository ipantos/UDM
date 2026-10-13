"""Cross-check of the numbers of assumptions.py with an independent likelihood.

The independent code is udm_indep.py (own data reader, SciPy spline with analytic
antiderivative, Simpson distances) with, here, analytic backgrounds and a two-fluid growth
system integrated with DOP853.  It recomputes the chi^2 values that assumptions.py stores
at single grid points (`spot_values`, including the case with another value of the
measurement at z = 0.07) and along the profiles of the `window` and `tracer_general`
stages (three profiles), and the velocities and the source of the potential at the limits
for pressureless galaxies.
A configuration that two stages store is counted once.  Keep this file next to
udm_indep.py.

Usage: python3 crosscheck_assumptions.py <udm_data.txt> <assumptions.json> <figure_data dir> [<summary.json>]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import udm_indep as U

data_path, json_path, fig_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
d0 = U.load_data(data_path)
A = json.loads(Path(json_path).read_text())
FB = A['tracer']['pressureless_fraction']
X_INI, X_REF = np.log1p(U.Z_INI), np.log1p(U.Z_REF)


class Expansion:
    """Total expansion E^2(x), x = ln(1+z), with eps_H = dlnH/dlna, the share Omega_b(x) of a
    pressureless component and the sound speed c_s^2(x) of the scalar."""

    def __init__(self, model, omega, c0, fb):
        ob0, ok = fb*omega, (1 - fb)*omega
        if model == 'constant':
            self.e2 = lambda x: omega*np.exp(3*x) + 1 - omega
            self.eps = lambda x: -1.5*omega*np.exp(3*x)/self.e2(x)
            self.cs2 = lambda x: c0 + 0*x
        else:
            of, oc = ok*np.sqrt(c0), ok*np.sqrt(1 - c0)
            kin = lambda x: np.sqrt(of**2 + oc**2*np.exp(6*x))
            self.e2 = lambda x: ob0*np.exp(3*x) + kin(x) + 1 - ob0 - ok
            self.eps = lambda x: -1.5*(ob0*np.exp(3*x) + oc**2*np.exp(6*x)/kin(x))/self.e2(x)
            self.cs2 = lambda x: of**2/kin(x)**2
        self.ob = lambda x: ob0*np.exp(3*x)/self.e2(x)
        xs = np.linspace(0.0, X_INI, 20001)
        f = np.exp(xs)/np.sqrt(self.e2(xs))
        h = xs[1] - xs[0]
        cum = np.zeros(len(xs))
        cum[2::2] = np.cumsum(h/3.0*(f[0:-2:2] + 4*f[1:-1:2] + f[2::2]))
        self._chi = CubicSpline(xs[::2], cum[::2])

    def at(self, z, what):
        x = np.log1p(np.asarray(z, float))
        return {'invE': lambda: 1/np.sqrt(self.e2(x)), 'chi': lambda: self._chi(x)}[what]()


def velocities(bg, z_out, k_out, f_ini, x_start=X_INI):
    """Scalar, pressureless and momentum-weighted velocity and the source of the potential
    at (z_i, k_i), each divided by D_u(k=0, z_ref).  One integration per distinct
    wavenumber, state (y, v, d_b, v_b), from x_start = ln(1 + z_start)."""
    def rhs(n, s, k):
        x = -n
        e, ob = bg.eps(x), bg.ob(x)
        q, w_tot = -1 - e, -1 - 2*e/3
        opw = 1 + w_tot/(1 - ob)
        press = bg.cs2(x)*(k*U.C_OVER_H0*np.exp(x))**2/bg.e2(x)
        y, v, db, vb = s
        source = 1.5*((1 - ob)*y + ob*db)
        return [3*(opw - 1)*y + opw*v, -(1 - q)*v + source - press*y/opw, vb, -(1 - q)*vb + source]
    opw_at = lambda x: 1 + (-1 - 2*bg.eps(x)/3)/(1 - bg.ob(x))
    start = [opw_at(x_start), f_ini, 1.0, f_ini]
    ref = solve_ivp(rhs, [-x_start, -X_REF], start, args=(0.0,), method='DOP853', rtol=1e-10, atol=1e-13)
    norm = ref.y[0, -1]/opw_at(X_REF)
    out = {name: np.empty(len(z_out)) for name in ('scalar', 'pressureless', 'mass_weighted', 'potential')}
    for k in np.unique(k_out):
        rows = np.flatnonzero(k_out == k)
        times = -np.log1p(z_out[rows])
        order = np.argsort(times)
        sol = solve_ivp(rhs, [-x_start, 0.0], start, args=(k,), method='DOP853', rtol=1e-10, atol=1e-13,
                        t_eval=times[order])
        for j, row in enumerate(rows[order]):
            x = np.log1p(z_out[row])
            weight = opw_at(x)*(1 - bg.ob(x))
            out['scalar'][row] = sol.y[1, j]/norm
            out['pressureless'][row] = sol.y[3, j]/norm
            out['mass_weighted'][row] = (weight*sol.y[1, j] + bg.ob(x)*sol.y[3, j])/(weight + bg.ob(x))/norm
            out['potential'][row] = 1.5*((1 - bg.ob(x))*sol.y[0, j] + bg.ob(x)*sol.y[2, j])/norm
    return out


def profile(bg, shape, rows=None, growth=None):
    """Minimum of chi^2 over A, sigma_8 and M.  growth: (values, covariance) of the growth
    measurements, where they differ from those of the data file."""
    a = b = c = 0.0
    for z, dm, sm, dh, sh, r in d0['aniso']:
        tv = np.array([bg.at(z, 'chi'), bg.at(z, 'invE')])
        ci = np.linalg.inv(np.array([[sm**2, r*sm*sh], [r*sm*sh, sh**2]]))
        ob = np.array([dm, dh])
        a += tv @ ci @ tv; b += tv @ ci @ ob; c += ob @ ci @ ob
    z, dv, sv = d0['iso']
    tv = (z*bg.at(z, 'chi')**2*bg.at(z, 'invE'))**(1/3)
    a += tv**2/sv**2; b += tv*dv/sv**2; c += dv**2/sv**2
    y = d0['musn'] - 5*np.log10((1 + d0['zsn'])*bg.at(d0['zsn'], 'chi'))
    one = np.ones(len(y))
    m = (y @ d0['cinv_sn'] @ one)/(one @ d0['cinv_sn'] @ one)
    values, cov = (d0['fs8'], d0['cov_g']) if growth is None else growth
    if rows is None:
        ci, s, gy = np.linalg.inv(cov), shape, values
    else:
        ci, s, gy = np.linalg.inv(cov[np.ix_(rows, rows)]), shape[rows], values[rows]
    s8 = (s @ ci @ gy)/(s @ ci @ s)
    return {'chi2': float(c - b*b/a + (y - m) @ d0['cinv_sn'] @ (y - m) + (gy - s8*s) @ ci @ (gy - s8*s)),
            'sigma8': float(s8)}


def growing_mode(om, x_end=X_INI):
    rhs = lambda n, y: [y[1], -(2 - 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om))*y[1]
                        + 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om)*y[0]]
    s = solve_ivp(rhs, [np.log(1/1001.), -x_end], [1., 1.], method='DOP853', rtol=1e-11, atol=1e-13)
    return s.y[1, -1]/s.y[0, -1]


worst, count, seen = 0.0, 0, set()
def compare(label, mine, theirs):
    global worst, count
    repeated = round(theirs, 9) in seen
    seen.add(round(theirs, 9))
    worst, count = max(worst, abs(mine - theirs)), count + (not repeated)
    print(f'  {label:74s} independent {mine:11.5f}   assumptions.py {theirs:11.5f}   difference {mine - theirs:+.1e}'
          + ('   (stored twice)' if repeated else ''))


keff = np.round(d0['keff'], 6)
zg = d0['zg']
every = np.arange(len(zg))
alt = keff.copy()
alt[0] = A['scales']['k_eff_z_0.07_alternative']
full_shape = np.array([int(np.argmin(np.abs(zg - z))) for z in (.295, .510, .706, .919, 1.317, 1.491)])
SCALES = {
    'as in the fit': (keff, None),
    'z = 0.07 with 0.025 < k < 0.1': (alt, None),
    'midpoint of each interval': (np.round(.5*(d0['kmin'] + d0['kmax']), 6), None),
    'upper end of each interval': (d0['kmax'], None),
    'k_eff capped at 0.155': (np.minimum(keff, .155), None),
    'without z = 0.07': (keff, every[1:]),
    'without the six DESI full-shape amplitudes': (keff, np.setdiff1d(every, full_shape)),
    'without the three measurements with k_eff = 0.23 at z < 0.4': (keff, np.flatnonzero(~((zg < .4) & (keff > .2)))),
    'z = 0.07 from the power spectrum alone': (keff, None),
}
# the last case changes a datum: the value and error of the power-spectrum analysis at z = 0.07
alone = A['scales'].get('power_spectrum_alone_at_z_0.07')
if alone:
    assert zg[0] == .07 and np.count_nonzero(d0['cov_g'][0]) == 1
    values_alone, cov_alone = d0['fs8'].copy(), d0['cov_g'].copy()
    values_alone[0], cov_alone[0, 0] = alone['value'], alone['sigma']**2
TRACER = {
    'scalar, one fluid (the fit)': (0., 'scalar'),
    'scalar, with a pressureless fraction': (FB, 'scalar'),
    'momentum-weighted mean': (FB, 'mass_weighted'),
    'pressureless fraction': (FB, 'pressureless'),
    'test particle in the potential of the scalar': (0., 'pressureless'),
}

for stage, cases in (('tracer', TRACER), ('scales', SCALES)):
    print(f'{stage}: chi2 at single grid points')
    for model in ('constant', 'minimal'):
        for name, spec in cases.items():
            if name not in A[stage][model]:
                continue
            fb, velocity = spec if stage == 'tracer' else (0., 'scalar')
            k_values, rows = (keff, None) if stage == 'tracer' else spec
            growth = (values_alone, cov_alone) if name == 'z = 0.07 from the power spectrum alone' else None
            for spot in A[stage][model][name]['spot_values']:
                bg = Expansion(model, spot['omega'], 10.0**spot['u'], fb)
                rate = growing_mode(spot['omega']) if model == 'minimal' else 1.0
                shape = velocities(bg, zg, k_values, rate)[velocity]
                compare(f'{model} | {name} | u = {spot["u"]:.2f}', profile(bg, shape, rows, growth)['chi2'], spot['chi2'])

if 'tracer_checks' in A:
    print('tracer_checks: chi2 at single grid points')
    for z_start, cases in A['tracer_checks']['earlier_start'].items():
        x_start = np.log1p(float(z_start))
        for name, case in cases.items():
            fb, velocity = {'scalar, one fluid': (0., 'scalar'), 'pressureless fraction': (FB, 'pressureless'),
                            'test particle in the potential of the scalar': (0., 'pressureless')}[name]
            for spot in case['spot_values']:
                bg = Expansion('constant', spot['omega'], 10.0**spot['u'], fb)
                shape = velocities(bg, zg, keff, growing_mode(spot['omega'], x_start), x_start)[velocity]
                compare(f'start at z = {z_start} | {name} | u = {spot["u"]:.2f}', profile(bg, shape)['chi2'], spot['chi2'])
    for model, case in A['tracer_checks']['other_wavenumber_at_z_0.07'].items():
        for spot in case['spot_values']:
            bg = Expansion(model, spot['omega'], 10.0**spot['u'], FB)
            rate = growing_mode(spot['omega']) if model == 'minimal' else 1.0
            shape = velocities(bg, zg, alt, rate)['pressureless']
            compare(f'other wavenumber | {model} | pressureless | u = {spot["u"]:.2f}', profile(bg, shape)['chi2'], spot['chi2'])
    for z_start, cases in A['tracer_checks'].get('earlier_start_minimal', {}).items():
        x_start = np.log1p(float(z_start))
        for name, case in cases.items():
            fb, velocity = {'scalar, one fluid': (0., 'scalar'), 'pressureless fraction': (FB, 'pressureless')}[name]
            for spot in case['spot_values']:
                bg = Expansion('minimal', spot['omega'], 10.0**spot['u'], fb)
                shape = velocities(bg, zg, keff, growing_mode(spot['omega'], x_start), x_start)[velocity]
                compare(f'minimal, start at z = {z_start} | {name} | u = {spot["u"]:.2f}', profile(bg, shape)['chi2'], spot['chi2'])
    for share, cases in A['tracer_checks'].get('other_pressureless_share', {}).items():
        for model, case in cases.items():
            for spot in case['spot_values']:
                bg = Expansion(model, spot['omega'], 10.0**spot['u'], float(share))
                rate = growing_mode(spot['omega']) if model == 'minimal' else 1.0
                shape = velocities(bg, zg, keff, rate)['pressureless']
                compare(f'pressureless share {share} | {model} | u = {spot["u"]:.2f}', profile(bg, shape)['chi2'], spot['chi2'])
    response = A['tracer_checks'].get('response_at_pressureless_limit')
    worst_ratio, ratios = 0.0, 0
    if response:
        print('tracer_checks: velocities and source of the potential at the limits for pressureless galaxies, over their values for c_s = 0')
        for model in ('constant', 'minimal'):
            for start, case in response[model].items():
                x_start = X_INI if start == '2.33' else np.log1p(float(start))
                rate = 1.0 if (model == 'constant' and start == '2.33') else growing_mode(case['omega'], x_start)
                with_c = velocities(Expansion(model, case['omega'], case['c_s2'], FB), zg, keff, rate, x_start)
                zero = velocities(Expansion(model, case['omega'], 1e-30, FB), zg, keff, rate, x_start)
                for name in ('scalar', 'pressureless', 'potential'):
                    mine = with_c[name]/zero[name]
                    gap = float(np.max(np.abs(mine - np.array(case[name]))))
                    worst_ratio, ratios = max(worst_ratio, gap), ratios + len(mine)
                    print(f'  {model} | start at z = {start} | {name:12s} at z = 0.07: independent {mine[0]:+.5f}   '
                          f'assumptions.py {case[name][0]:+.5f}   largest difference over the 17 measurements {gap:.1e}')

print('window: profile of the 11-parameter model with the other wavenumber at z = 0.07 (chi2 at tabulated nodes)')
with open(fig_dir/'asm_profile_other_window.csv') as fh:
    T = list(csv.DictReader(fh))
d_alt = dict(d0, keff=alt)
for i in np.unique(np.r_[0, np.searchsorted([float(r['u']) for r in T], [-7.0, -6.4, -6.1, -6.0, -5.9])]):
    if i >= len(T):
        continue
    eps = np.array([float(T[i][f'eps{j}']) for j in range(7)])
    bg = U.Background(eps)
    shape = U.growth_shape(bg, d_alt, U.cs2_constant([10.0**float(T[i]['u'])]))[0]
    a = b = c = 0.0
    for z, dm, sm, dh, sh, r in d0['aniso']:
        tv = np.array([float(bg.at(z, 'chi')), float(bg.at(z, 'invE'))])
        ci = np.linalg.inv(np.array([[sm**2, r*sm*sh], [r*sm*sh, sh**2]]))
        ob = np.array([dm, dh])
        a += tv @ ci @ tv; b += tv @ ci @ ob; c += ob @ ci @ ob
    z, dv, sv = d0['iso']
    tv = float((z*bg.at(z, 'chi')**2*bg.at(z, 'invE'))**(1/3))
    a += tv**2/sv**2; b += tv*dv/sv**2; c += dv**2/sv**2
    y = d0['musn'] - 5*np.log10((1 + d0['zsn'])*bg.at(d0['zsn'], 'chi'))
    one = np.ones(len(y))
    m = (y @ d0['cinv_sn'] @ one)/(one @ d0['cinv_sn'] @ one)
    s8 = (shape @ d0['cinv_g'] @ d0['fs8'])/(shape @ d0['cinv_g'] @ shape)
    chi2 = c - b*b/a + (y - m) @ d0['cinv_sn'] @ (y - m) + (d0['fs8'] - s8*shape) @ d0['cinv_g'] @ (d0['fs8'] - s8*shape)
    compare(f'window | u = {float(T[i]["u"]):.2f}', float(chi2), float(T[i]['chi2']))


class SplineExpansion:
    """The expansion history of the 11-parameter model (spline of eps_H through seven nodes)
    with a pressureless component of density omega_b today, in the form `velocities` needs."""

    def __init__(self, eps_nodes, omega_b, cs2):
        self.bg = U.Background(eps_nodes)
        self.eps = lambda x: self.bg.sp(x)
        self.e2 = lambda x: np.exp(-2*(self.bg.anti(x) - self.bg.anti(0.0)))
        self.ob = lambda x: omega_b*np.exp(3*x)/self.e2(x)
        self.cs2 = lambda x: cs2 + 0*x

    def at(self, z, what):
        return self.bg.at(z, what)


if 'tracer_general' in A:
    print('tracer_general: profiles of the 11-parameter model with the velocity of a pressureless component '
          '(chi2 at tabulated nodes)')
    general = A['tracer_general']
    for label, name, k_values in (('test particle in the potential of the scalar', 'test_particle', keff),
                                  ('pressureless component', 'pressureless', keff),
                                  ('pressureless component, other wavenumber at z = 0.07', 'pressureless_other_window', alt)):
        if label not in general:
            continue
        with open(fig_dir/f'asm_profile_general_{name}.csv') as fh:
            T = list(csv.DictReader(fh))
        u_all = np.array([float(r['u']) for r in T])
        limit = np.log10(general[label]['upper_limits_c_s2']['95% one-sided (Delta chi2 = 2.71)'])
        wanted = np.unique(np.r_[0, np.searchsorted(u_all, [-7.0, -6.0, -5.5, limit - .05, limit, -5.0, -4.5, -3.0]),
                                 len(T) - 1])
        for i in wanted:
            if i >= len(T):
                continue
            eps = np.array([float(T[i][f'eps{j}']) for j in range(7)])
            bg = SplineExpansion(eps, general[label]['omega_b'], 10.0**u_all[i])
            shape = velocities(bg, zg, k_values, 1.0)['pressureless']
            compare(f'general model | {label} | u = {u_all[i]:.2f}', profile(bg, shape)['chi2'], float(T[i]['chi2']))
print(f'largest absolute difference in the {count} distinct values of chi2: {worst:.1e}')
if 'tracer_checks' in A and response:
    print(f'largest absolute difference in the {ratios} ratios: {worst_ratio:.1e}')
if len(sys.argv) > 4:
    summary = {'chi2_values_compared': count, 'largest_difference_in_chi2': worst}
    if 'tracer_checks' in A and response:
        summary.update({'ratios_compared': ratios, 'largest_difference_in_ratios': worst_ratio})
    Path(sys.argv[4]).write_text(json.dumps(summary, indent=1))
