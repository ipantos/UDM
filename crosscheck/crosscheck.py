"""Cross-check of the numbers of extensions.py with the independent likelihood udm_indep.py
(own data reader, SciPy spline with analytic antiderivative, Simpson distances, DOP853
growth).  Keep both files in one directory.

Usage: python3 crosscheck.py <udm_data.txt> <extensions.json> <figure_data dir> [<summary.json>]
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import udm_indep as U

data_path, json_path, fig_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
d = U.load_data(data_path)
J = json.loads(Path(json_path).read_text())


class Analytic:
    """Background given by E^2(x) and eps_H(x) as functions of x = ln(1+z)."""
    def __init__(self, e2, eps, n=20001):
        self.e2f, self.epsf = e2, eps
        self.x = np.linspace(0.0, np.log1p(U.Z_INI), n)
        self.eps = eps(self.x)
        self.invE = 1/np.sqrt(e2(self.x))
        f = np.exp(self.x)*self.invE
        h = self.x[1] - self.x[0]
        cum = np.zeros(n)
        cum[2::2] = np.cumsum(h/3.0*(f[0:-2:2] + 4*f[1:-1:2] + f[2::2]))
        self._chi = CubicSpline(self.x[::2], cum[::2])

    def at(self, z, what):
        x = np.log1p(np.asarray(z, float))
        return {'invE': lambda: 1/np.sqrt(self.e2f(x)), 'eps': lambda: self.epsf(x), 'chi': lambda: self._chi(x)}[what]()


def lcdm(om):
    e2 = lambda x: om*np.exp(3*x) + 1 - om
    return Analytic(e2, lambda x: -1.5*om*np.exp(3*x)/e2(x))


def minimal(om_kin, c0):
    of, oc = om_kin*np.sqrt(c0), om_kin*np.sqrt(1 - c0)
    kin = lambda x: np.sqrt(of**2 + oc**2*np.exp(6*x))
    e2 = lambda x: kin(x) + 1 - om_kin
    return Analytic(e2, lambda x: -1.5*oc**2*np.exp(6*x)/kin(x)/e2(x))


def profile(bg, shape, rows=None):
    """Minimum of chi^2 over A, sigma_8 and M; returns total and parts."""
    a = b = c = 0.0
    for z, dm, sm, dh, sh, r in d['aniso']:
        tv = np.array([bg.at(z, 'chi'), bg.at(z, 'invE')])
        ci = np.linalg.inv(np.array([[sm**2, r*sm*sh], [r*sm*sh, sh**2]]))
        ob = np.array([dm, dh])
        a += tv @ ci @ tv; b += tv @ ci @ ob; c += ob @ ci @ ob
    z, dv, sv = d['iso']
    tv = (z*bg.at(z, 'chi')**2*bg.at(z, 'invE'))**(1/3)
    a += tv**2/sv**2; b += tv*dv/sv**2; c += dv**2/sv**2
    A = b/a
    chi_bao = c - b*b/a
    y = d['musn'] - 5*np.log10((1 + d['zsn'])*bg.at(d['zsn'], 'chi'))
    one = np.ones(len(y))
    m = (y @ d['cinv_sn'] @ one)/(one @ d['cinv_sn'] @ one)
    chi_sn = (y - m) @ d['cinv_sn'] @ (y - m)
    if rows is None:
        ci, s, gy = d['cinv_g'], shape, d['fs8']
    else:
        ci, s, gy = np.linalg.inv(d['cov_g'][np.ix_(rows, rows)]), shape[rows], d['fs8'][rows]
    s8 = (s @ ci @ gy)/(s @ ci @ s)
    chi_g = (gy - s8*s) @ ci @ (gy - s8*s)
    return dict(chi2=float(chi_bao + chi_sn + chi_g), bao=float(chi_bao), sn=float(chi_sn), growth=float(chi_g),
                A=float(A), sigma8=float(s8), M=float(m - 5*np.log10(A)))


def spline_chi2(eps, cs2=None, omega_f=None, f_ini=1.0, rows=None):
    bg = U.Background(eps)
    if omega_f is None:
        fn = U.cs2_constant([cs2])
    else:
        def fn(x):
            r = 3*omega_f*np.interp(x, bg.x, bg.invE)**2/(-2*np.interp(x, bg.x, bg.eps))
            c = (np.sqrt(1 + 4*r*r) - 1)/(2*r) if r > 1e-8 else r
            return np.array([c*c])
    return profile(bg, U.growth_shape(bg, d, fn, f_ini=f_ini)[0], rows)


def read(name):
    with open(fig_dir/(name + '.csv')) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


worst, worst_chi2, count_chi2 = 0.0, 0.0, 0
def compare(label, mine, theirs):
    global worst, worst_chi2, count_chi2
    worst = max(worst, abs(mine - theirs))
    if 'chi2' in label:
        worst_chi2, count_chi2 = max(worst_chi2, abs(mine - theirs)), count_chi2 + 1
    print(f'  {label:58s} independent {mine:12.5f}   extensions.py {theirs:12.5f}   difference {mine - theirs:+.1e}')


print('1. LambdaCDM on the 52 points')
for key in ('same_initial_rate', 'growing_mode_initial_rate'):
    L = J['lcdm'][key]
    bg = lcdm(L['omega_m'])
    p = profile(bg, U.growth_shape(bg, d, U.cs2_constant([0.0]), f_ini=L['initial_rate'])[0])
    compare(key + ' chi2', p['chi2'], L['chi2'])
    compare(key + ' sigma8_ref', p['sigma8'], L['sigma8_ref'])
    compare(key + ' A', p['A'], L['A'])
    if key.startswith('growing'):
        # growing-mode rate of LambdaCDM at z = 2.33 by an independent integration from z = 1000
        om = L['omega_m']
        rhs = lambda n, y: [y[1], -(2 - 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om))*y[1]
                            + 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om)*y[0]]
        s = solve_ivp(rhs, [np.log(1/1001.), -np.log1p(2.33)], [1., 1.], method='DOP853', rtol=1e-11, atol=1e-13)
        compare('growing-mode rate at z = 2.33', s.y[1, -1]/s.y[0, -1], L['initial_rate'])

print('2. Best fit of the 11-parameter model')
for key in ('with_V_nonnegative', 'without_V_restriction'):
    B = J['bestfit'][key]
    p = spline_chi2(np.array(B['eps_nodes']), B['c_s2'])
    compare(key + ' chi2', p['chi2'], B['chi2'])
    compare(key + ' chi2 growth', p['growth'], B['chi2_growth'])
    compare(key + ' sigma8_ref', p['sigma8'], B['sigma8_ref'])

print('3. Profile likelihood of the constant sound speed (chi2 at the tabulated nodes)')
for key in ('with_V_nonnegative', 'without_V_restriction'):
    T = read('ext_profile_' + key)
    base = spline_chi2(np.array(J['bestfit'][key]['eps_nodes']), J['bestfit'][key]['c_s2'])['chi2']
    for i in np.unique(np.r_[0, np.searchsorted(T['u'], [-7.0, -6.5, -6.3, -6.2, -6.1, -6.0])]):
        if i >= len(T['u']):
            continue
        eps = np.array([T[f'eps{j}'][i] for j in range(7)])
        compare(f'{key} u = {T["u"][i]:.2f} delta chi2', spline_chi2(eps, 10.0**T['u'][i])['chi2'] - base,
                T['chi2'][i] - J['bestfit'][key]['chi2'])

print('4. Models with a LambdaCDM-like expansion')
for model in ('constant', 'minimal'):
    for key in ('same_initial_rate', 'growing_mode_initial_rate'):
        Gd = J['grids'][model][key]['all_data']
        T = read(f'ext_grid_{model}_{key}')
        rate = 1.0
        near = int(np.argmin(np.abs(T['u'] - np.log10(Gd['profile_upper_limits']['95% one-sided (Delta chi2 = 2.71)']))))
        for u in (-12.0, float(T['u'][near]), float(T['u'][near + 1])):      # tabulated values around the limit
            om = float(np.interp(u, T['u'], T['omega']))
            if key.startswith('growing'):
                rhs = lambda n, y: [y[1], -(2 - 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om))*y[1]
                                    + 1.5*om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om)*y[0]]
                s = solve_ivp(rhs, [np.log(1/1001.), -np.log1p(2.33)], [1., 1.], method='DOP853', rtol=1e-11, atol=1e-13)
                rate = s.y[1, -1]/s.y[0, -1]
            if model == 'constant':
                bg, fn = lcdm(om), U.cs2_constant([10.0**u])
            else:
                bg, fn = minimal(om, 10.0**u), U.cs2_a6([10.0**u])
            p = profile(bg, U.growth_shape(bg, d, fn, f_ini=rate)[0])
            compare(f'{model} {key} u = {u:.3f} chi2', p['chi2'], float(np.interp(u, T['u'], T['chi2'])))

print('5. Constant prefactor with a free potential (chi2 at the tabulated nodes)')
T = read('ext_profile_tied')
for i in np.unique(np.r_[0, np.searchsorted(T['u'], [-3.6, -3.3, -3.2, -3.1, -3.0])]):
    if i >= len(T['u']):
        continue
    eps = np.array([T[f'eps{j}'][i] for j in range(7)])
    compare(f'tied log10 Omega_f = {T["u"][i]:.2f} chi2', spline_chi2(eps, omega_f=10.0**T['u'][i])['chi2'], T['chi2'][i])
if (fig_dir/'ext_profile_without_lowest_z.csv').exists():
    print('6. Profile likelihood without the growth point at z = 0.07 (chi2 of 51 measurements at the tabulated nodes)')
    T = read('ext_profile_without_lowest_z')
    kept = np.flatnonzero(d['zg'] > 0.1)
    for i in np.unique(np.r_[0, np.searchsorted(T['u'], [-7.0, -6.5, -6.2, -6.0, -5.9])]):
        if i >= len(T['u']):
            continue
        eps = np.array([T[f'eps{j}'][i] for j in range(7)])
        compare(f'u = {T["u"][i]:.2f} chi2', spline_chi2(eps, 10.0**T['u'][i], rows=kept)['chi2'], T['chi2'][i])
print(f'largest absolute difference: {worst:.1e}; in the {count_chi2} values of chi2: {worst_chi2:.1e}')
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps({'chi2_values_compared': count_chi2, 'largest_difference_in_chi2': worst_chi2,
                                             'largest_difference': worst}, indent=1))
