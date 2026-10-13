"""Baselines and extensions of the joint fit of fit_joint.py, from its stored run.

Keep this file next to fit_joint.py, figures.py and udm_data.txt.  It reads the stored
posterior and the result of the `nodes` stage and changes neither.  Every stage adds its
numbers to extensions.json; tables go to figure_data/ext_*.csv and the chains of the
`chain` stage to extensions_chains/.  Run the stages in this order (times for one core):

    python3 extensions.py lcdm        LambdaCDM on the same 52 points                  15 s
    python3 extensions.py bestfit     minimum of the 11-parameter model                5 min
    python3 extensions.py profile     profile likelihood of the constant c_s^2        15 min
    python3 extensions.py leaveout    the same profile without the growth point at
                                      z = 0.07                                         8 min
    python3 extensions.py grids       constant c_s^2, and constant f and V, with a
                                      LambdaCDM-like expansion history                 2 min
    python3 extensions.py coverage    simulated data: how often the profile limits lie
                                      above the truth (LambdaCDM-like expansion)       1 min
    python3 extensions.py tied        constant f with a free potential: the sound
                                      speed follows from rho+p                        11 min
    python3 extensions.py posterior   stored posterior: q(z), the V = 0 boundary,
                                      f and V over 0 < z < 2.33, Jeans threshold       10 s
    python3 extensions.py figure      tables for the growth-response figure             2 s
    python3 extensions.py early       interaction scale of the constant-f, constant-V
                                      model at early times                              1 s
    python3 extensions.py chain constant 12000         posterior samples for the model
    python3 extensions.py chain constant_any_V 12000   and prior of fit_joint.py, for the
    python3 extensions.py chain tied 12000             same without V >= 0, and for
                                      constant f with a free potential         25 min each
    python3 extensions.py chains      summaries of these chains                        30 s

`bestfit` and `profile` are run with and without the restriction V >= 0.  Later stages
read earlier results from extensions.json: `profile`, `leaveout`, `tied` and `chain` need
`bestfit`; `coverage` needs `grids`; `figure` needs `profile`, `grids` and `posterior`.
`chain <name> <steps>` continues an existing chain of that name; delete
extensions_chains/<name>.npz to start again.

NUMERICS.  The model is linear in three parameters, A, sigma_8 and the supernova offset
M: chi^2 is quadratic in A (BAO), in sigma_8 (growth) and in M + 5 log10 A (supernovae;
with M free the supernova term does not depend on A).  They are eliminated exactly, and
all searches run over the expansion history and the sound speed only.  Growth is
integrated with the fixed-step RK4 of figures.py, vectorised over trial models.  Every
chi^2 stored for the spline model and for LambdaCDM is evaluated a second time with the
likelihood functions of fit_joint.py (key `chi2`; the RK4 value is `chi2_rk4_solver`).
A flat prior on (A, sigma_8, M) integrates to a Gaussian volume, returned as `logvol`;
the posterior limits of the `grids` and `chain` stages include it.

The prior ranges of fit_joint.py for A, sigma_8 and M are not imposed here: wherever the
likelihood is not negligible the best values lie far inside them.

LIMITS.  `profile` limits are the values at which the smallest chi^2 has risen by 1, 2.71
and 3.84 above its minimum.  For a Gaussian likelihood 2.71 gives the one-sided 95% limit
and 3.84 the upper end of the two-sided 95% interval (a one-sided 97.5% limit).  These
limits use no prior on the sound speed, but they do depend on the restrictions imposed
on the expansion history.  Posterior limits are 95th percentiles for the stated prior.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import LinearConstraint, brentq, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import figures as G  # noqa: E402
import fit_joint as fj  # noqa: E402

OUT_JSON = ROOT / 'extensions.json'
OUT = ROOT / 'figure_data'
XG, ZG = G.XG, G.ZG
OBS = G.load_observations()
GZ, GK, GY, GCOV = OBS['growth_z'], OBS['growth_k'], OBS['growth_y'], OBS['growth_cov']
GKMAX = OBS['growth_kmax']
GCINV = np.linalg.inv(GCOV)
GSIG = np.sqrt(np.diag(GCOV))
SNZ, SNY = OBS['sn_z'], OBS['sn_y']
SNCINV = np.linalg.inv(OBS['sn_cov'])
ANISO, BGS = G.ANISO, G.BGS
BAO_CINV = [np.linalg.inv(np.array([[r[2]**2, r[5]*r[2]*r[4]], [r[5]*r[2]*r[4], r[4]**2]])) for r in ANISO]
I_CS2, I_S8 = 8, 9
PRIOR_U = fj.PRIOR_U                           # (-12, -2): prior uniform in log10 c_s^2
LEVELS = {'68.3% two-sided (Delta chi2 = 1)': 1.0, '95% one-sided (Delta chi2 = 2.71)': 2.706,
          '95% two-sided (Delta chi2 = 3.84)': 3.841}


def store(key, value):
    data = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    data[key] = value
    OUT_JSON.write_text(json.dumps(data, indent=2))


def table(name, columns):
    OUT.mkdir(exist_ok=True)
    G.write_table('ext_' + name, columns)


# ----------------------------------------------------------------------------
# growth solver: the exact y-V system of Appendix C, batched fixed-step RK4 as in figures.py
# ----------------------------------------------------------------------------
def tied_cs2(q, inv_hubble, omega_f):
    """Constant prefactor f: c_s/(1-c_s^2) = f/(rho+p), with f/3H0^2 = omega_f and
    (rho+p)/H0^2 = 2(1+q)E^2."""
    r = 3*omega_f*inv_hubble**2/(2*(1 + q))
    c = 2*r/(1 + np.sqrt(1 + 4*r*r))               # the root of c/(1-c^2) = r with 0 <= c < 1
    return c*c


class Solver:
    """Integration points and backgrounds for a batch of models.

    steps: RK4 base steps over the fitted interval (figures.py uses 1600; with 400, used
    for searches and chains, the predictions change by 2e-6 of a measurement error).
    z_out, k_out: output redshifts and the wavenumber of each (default: the 17 data)."""

    def __init__(self, steps=1600, z_out=None, k_out=None, z_norm=fj.Z_NORM):
        self.z_out = GZ if z_out is None else np.asarray(z_out, float)
        k_out = GK if k_out is None else np.asarray(k_out, float)
        self.k_all, self.kmap = np.unique(np.r_[k_out, 0.], return_inverse=True)
        t_out = -np.log1p(np.r_[self.z_out, z_norm])
        self.times = np.unique(np.r_[np.linspace(-XG[-1], 0, steps + 1), t_out])
        self.x = -np.r_[self.times, .5*(self.times[1:] + self.times[:-1])]
        self.z = np.expm1(self.x)
        self.idx, self.wgt = G.interpolation_weights(XG, self.x)
        self.basis = G.SMAT[self.idx]*(1 - self.wgt[:, None]) + G.SMAT[self.idx + 1]*self.wgt[:, None]
        self.dest = {}                 # time index -> output rows (redshifts may repeat)
        for j, t in enumerate(t_out):
            self.dest.setdefault(int(np.searchsorted(self.times, t)), []).append(j)

    # backgrounds: H0/H on the grid ZG (for distances), and q and H0/H on the points
    def spline(self, eps):
        eps = np.atleast_2d(eps)
        low, high = np.exp(eps @ G.IMAT[self.idx].T), np.exp(eps @ G.IMAT[self.idx + 1].T)
        return np.exp(eps @ G.IMAT.T), -1 - eps @ self.basis.T, low*(1 - self.wgt) + high*self.wgt

    def lcdm(self, omega_m):
        """Single fluid with rho = rho_d a^-3 + Lambda and p = -Lambda."""
        om = np.atleast_1d(omega_m)[:, None]
        e2 = lambda z: om*(1 + z)**3 + 1 - om
        return 1/np.sqrt(e2(ZG)), 1.5*om*(1 + self.z)**3/e2(self.z) - 1, 1/np.sqrt(e2(self.z))

    def minimal(self, omega_c, omega_f):
        """Constant prefactor f and constant potential V (Chaplygin gas plus a constant):
        rho/3H0^2 = sqrt(Of^2 + Oc^2 (1+z)^6) + OL with OL = 1 - sqrt(Of^2 + Oc^2);
        c_s^2 = Of^2/(Of^2 + Oc^2 (1+z)^6);  p/3H0^2 = -Of c_s - OL.
        Returns also c_s^2 on the points."""
        oc, of = np.atleast_1d(omega_c)[:, None], np.atleast_1d(omega_f)[:, None]
        ol = 1 - np.sqrt(of**2 + oc**2)

        def at(z):
            kin = np.sqrt(of**2 + oc**2*(1 + z)**6)
            rho = kin + ol
            return 1/np.sqrt(rho), (1 + 3*(-of**2/kin - ol)/rho)/2, of**2/kin**2
        inv_pts, q_pts, cs2_pts = at(self.z)
        return at(ZG)[0], q_pts, inv_pts, cs2_pts

    def tied_cs2(self, q_pts, inv_pts, omega_f):
        return tied_cs2(q_pts, inv_pts, np.reshape(omega_f, (-1, 1)))

    def growth(self, q_pts, inv_pts, cs2_pts, f_ini=1.0):
        """D and V at the output redshifts for every wavenumber: arrays (n, nz+1, nk);
        the last output row is the normalisation redshift."""
        n, nt = len(q_pts), len(self.times)
        pbase = np.broadcast_to(cs2_pts, q_pts.shape)*(G.C_OVER_H0*inv_pts*np.exp(self.x))**2
        kval = np.square(self.k_all)[None, :]
        opw = 2*(1 + q_pts)/3
        d = np.ones((n, len(self.k_all)))*opw[:, 0, None]
        v = np.ones_like(d)*np.reshape(f_ini, (-1, 1))
        res_d = np.empty((n, len(self.z_out) + 1, len(self.k_all)))
        res_v = np.empty_like(res_d)
        for j in self.dest.get(0, []):
            res_d[:, j], res_v[:, j] = d/opw[:, 0, None], v
        for i, dt in enumerate(np.diff(self.times)):
            d, v = G.rk4_step(d, v, dt, q_pts[:, i, None], q_pts[:, nt + i, None], q_pts[:, i + 1, None],
                              pbase[:, i, None]*kval, pbase[:, nt + i, None]*kval, pbase[:, i + 1, None]*kval)
            for j in self.dest.get(i + 1, []):
                res_d[:, j], res_v[:, j] = d/opw[:, i + 1, None], v
        return res_d, res_v

    def shape(self, q_pts, inv_pts, cs2_pts, f_ini=1.0):
        """V(k_i, z_i)/D(k=0, z_ref): the growth prediction for sigma_8 = 1."""
        d, v = self.growth(q_pts, inv_pts, cs2_pts, f_ini)
        m = len(self.z_out)
        return v[:, np.arange(m), self.kmap[:m]]/d[:, -1, self.kmap[-1], None]


FINE, FAST = Solver(1600), Solver(400)


# ----------------------------------------------------------------------------
# the three linear parameters, eliminated exactly
# ----------------------------------------------------------------------------
def distances(inv_grid):
    return np.column_stack([np.zeros(len(inv_grid)),
                            np.cumsum(.5*(inv_grid[:, 1:] + inv_grid[:, :-1])*np.diff(ZG), axis=1)])


def linear_profile(inv_grid, shape, rows=None):
    """Minimum over (A, sigma_8, M) of chi^2 for a batch of models.  rows: indices of
    the growth measurements kept (default all).  Returns arrays: chi2 and its parts, the
    optimal A, sigma8, M, the 68% error of sigma8, and logvol = ln of the volume that a
    flat prior on (A, sigma_8, M) integrates to, up to a constant."""
    dm1 = distances(inv_grid)
    tm, th = G.interpolate_rows(dm1, ANISO[:, 0]), G.interpolate_rows(inv_grid, ANISO[:, 0])
    a = b = c = 0.
    for j, row in enumerate(ANISO):                        # BAO prediction = A t
        t = np.column_stack([tm[:, j], th[:, j]])
        obs = np.array([row[1], row[3]])
        a = a + np.einsum('bi,ij,bj->b', t, BAO_CINV[j], t)
        b = b + t @ BAO_CINV[j] @ obs
        c = c + obs @ BAO_CINV[j] @ obs
    tv = np.cbrt(BGS[0]*G.interpolate_rows(dm1, [BGS[0]])[:, 0]**2*G.interpolate_rows(inv_grid, [BGS[0]])[:, 0])
    a, b, c = a + tv**2/BGS[2]**2, b + tv*BGS[1]/BGS[2]**2, c + BGS[1]**2/BGS[2]**2
    y = SNY - 5*np.log10((1 + SNZ)*G.interpolate_rows(dm1, SNZ))      # mu = ... + (M + 5 log10 A)
    one = np.ones(len(SNZ))
    norm = one @ SNCINV @ one
    cal = (y @ SNCINV @ one)/norm
    chi_sn = np.einsum('bi,ij,bj->b', y, SNCINV, y) - (y @ SNCINV @ one)**2/norm
    if rows is None:
        cinv, s, gy = GCINV, shape, GY
    else:
        cinv, s, gy = np.linalg.inv(GCOV[np.ix_(rows, rows)]), shape[:, rows], GY[rows]
    ss, sy = np.einsum('bi,ij,bj->b', s, cinv, s), s @ cinv @ gy
    chi_g = gy @ cinv @ gy - sy**2/ss
    return {'chi2': c - b**2/a + chi_sn + chi_g, 'bao': c - b**2/a, 'sn': chi_sn, 'growth': chi_g,
            'A': b/a, 'sigma8': sy/ss, 'sigma8_error': 1/np.sqrt(ss), 'M': cal - 5*np.log10(b/a),
            'logvol': -.5*np.log(a) - .5*np.log(ss)}


def spline_profile(eps, cs2=0., omega_f=None, solver=FINE, f_ini=1.0, rows=None):
    """Profile chi^2 for a batch of node vectors (n, 7).  cs2: constant sound speed
    (scalar or (n,)); omega_f, if given, ties the sound speed to rho+p instead."""
    inv_grid, q_pts, inv_pts = solver.spline(eps)
    cs2_pts = (np.reshape(cs2, (-1, 1))*np.ones((len(q_pts), 1)) if omega_f is None
               else solver.tied_cs2(q_pts, inv_pts, omega_f))
    return linear_profile(inv_grid, solver.shape(q_pts, inv_pts, cs2_pts, f_ini), rows)


def fit_chi2(eps, amp, cs2, s8, cal, omega_f=None):
    """chi^2 of fit_joint.py for the spline model, and its (BAO, SN, growth) parts.
    omega_f, if given, ties the sound speed to rho+p on the grid of fit_joint.py."""
    inv_h, d_m, d_h, q_grid = fj.background(np.asarray(eps, float), amp)
    cs2_grid = (fj.cs2_on_grid(np.array([cs2]), fj.node_positions(1)) if omega_f is None
                else tied_cs2(q_grid, inv_h, omega_f))
    parts = [fj.chi2_bao(d_m, d_h), fj.chi2_sne(d_m, cal), fj.chi2_growth(cs2_grid, q_grid, inv_h, amp, s8)]
    return float(sum(parts)), [float(v) for v in parts]


def crossing(x, y, level):
    """First x at which y rises through `level`, else None; monotone cubic interpolation
    between the tabulated points."""
    above = np.flatnonzero((y[1:] >= level) & (y[:-1] < level))
    if not len(above):
        return None
    i = above[0]
    curve = PchipInterpolator(x, y)
    return float(brentq(lambda t: float(curve(t)) - level, x[i], x[i + 1]))


def upper_limit(u, weight, lower_edge, fraction=.95):
    """Percentile of u for a prior uniform in u from lower_edge to u[-1]; the likelihood
    weight is taken constant (= weight[0]) below u[0]."""
    keep = u >= lower_edge - 1e-12
    uu, ww = u[keep], weight[keep]
    if lower_edge < uu[0]:
        uu, ww = np.r_[lower_edge, uu], np.r_[ww[0], ww]
    cdf = np.r_[0., np.cumsum(.5*(ww[1:] + ww[:-1])*np.diff(uu))]
    return float(np.interp(fraction, cdf/cdf[-1], uu))


# ----------------------------------------------------------------------------
# stage: LambdaCDM baseline
# ----------------------------------------------------------------------------
def lcdm_fit_chi2(omega_m, amp, s8, cal, f_ini=1.0):
    """The LambdaCDM chi^2 through the functions of fit_joint.py, on its own grid."""
    e2 = omega_m*(1 + fj.ZGRID)**3 + 1 - omega_m
    inv_h = 1/np.sqrt(e2)
    q_grid = 1.5*omega_m*(1 + fj.ZGRID)**3/e2 - 1
    d_m = amp*np.concatenate([[0.], np.cumsum(.5*(inv_h[1:] + inv_h[:-1])*fj.DZ)])
    saved, fj.F_INI = fj.F_INI, f_ini
    try:
        parts = [fj.chi2_bao(d_m, amp*inv_h), fj.chi2_sne(d_m, cal),
                 fj.chi2_growth(np.zeros_like(fj.XG), q_grid, inv_h, amp, s8)]
    finally:
        fj.F_INI = saved
    return float(sum(parts)), [float(v) for v in parts]


def lcdm_growing_mode(omega_m, z=fj.Z_INI):
    """dlnD/dlna of the LambdaCDM growing mode at redshift z (array input allowed)."""
    out = []
    for om in np.atleast_1d(omega_m):
        def rhs(n, y, om=om):
            oma = om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om)
            return [y[1], -(2 - 1.5*oma)*y[1] + 1.5*oma*y[0]]
        sol = solve_ivp(rhs, [np.log(1e-3), -np.log1p(z)], [1e-3, 1e-3], rtol=1e-10, atol=1e-14)
        out.append(sol.y[1, -1]/sol.y[0, -1])
    return np.array(out)


def lcdm_profile(omega_m, cs2=0., growing_mode=False, solver=FINE, rows=None):
    inv_grid, q_pts, inv_pts = solver.lcdm(omega_m)
    f_ini = lcdm_growing_mode(omega_m) if growing_mode else 1.0
    return linear_profile(inv_grid, solver.shape(q_pts, inv_pts, np.reshape(cs2, (-1, 1)), f_ini), rows)


def command_lcdm():
    result = {}
    for label, mode in (('same_initial_rate', False), ('growing_mode_initial_rate', True)):
        best = minimize_scalar(lambda om: float(lcdm_profile(om, growing_mode=mode)['chi2'][0]),
                               bounds=(.15, .6), method='bounded', options={'xatol': 1e-9})
        om = float(best.x)
        prof = lcdm_profile(om, growing_mode=mode)
        rate = float(lcdm_growing_mode(om)[0]) if mode else 1.0
        chi2, parts = lcdm_fit_chi2(om, prof['A'][0], prof['sigma8'][0], prof['M'][0], rate)
        # 68% intervals from Delta chi2 = 1: Omega_m on a grid, sigma_8 profiled over Omega_m
        grid = np.linspace(om - .06, om + .06, 481)
        scan = lcdm_profile(grid, growing_mode=mode, solver=FAST)
        inside = grid[scan['chi2'] - best.fun <= 1]
        s8 = np.linspace(prof['sigma8'][0] - .15, prof['sigma8'][0] + .15, 1201)
        surface = scan['chi2'][:, None] + ((s8[None, :] - scan['sigma8'][:, None])/scan['sigma8_error'][:, None])**2
        inside8 = s8[surface.min(axis=0) - best.fun <= 1]
        result[label] = {
            'omega_m': om, 'omega_m_68': [float(inside[0]), float(inside[-1])],
            'A': float(prof['A'][0]), 'sigma8_ref': float(prof['sigma8'][0]),
            'sigma8_ref_68': [float(inside8[0]), float(inside8[-1])],
            'sigma8_ref_error_at_fixed_omega_m': float(prof['sigma8_error'][0]), 'M': float(prof['M'][0]),
            'initial_rate': rate, 'chi2': chi2, 'chi2_bao': parts[0], 'chi2_sn': parts[1],
            'chi2_growth': parts[2], 'chi2_rk4_solver': float(prof['chi2'][0]),
            'parameters': 4, 'measurements': int(fj.NDATA)}
        print(label, json.dumps(result[label], indent=1))
        inv_grid, q_pts, inv_pts = FINE.lcdm(om)
        shape = FINE.shape(q_pts, inv_pts, 0., rate)[0]
        zc = np.unique(np.r_[np.linspace(.05, 2.33, 151), SNZ, ANISO[:, 0], BGS[0]])
        dm = prof['A'][0]*np.interp(zc, ZG, distances(inv_grid)[0])
        dh = prof['A'][0]*np.interp(zc, ZG, inv_grid[0])
        curve = Solver(1600, z_out=zc, k_out=np.zeros(len(zc)))
        growth_curve = prof['sigma8'][0]*curve.shape(*curve.lcdm(om)[1:], 0., rate)[0]
        table('lcdm_curves_' + label, {'z': zc, 'dm': dm, 'dh': dh, 'dv': np.cbrt(zc*dm**2*dh),
                                       'mu': 5*np.log10((1 + zc)*dm) + prof['M'][0], 'growth': growth_curve})
        table('lcdm_growth_points_' + label, {'z': GZ, 'keff': GK, 'prediction': prof['sigma8'][0]*shape,
                                              'observed': GY, 'sigma': GSIG,
                                              'residual': (GY - prof['sigma8'][0]*shape)/GSIG})
    # sigma_8 at the reference redshift for Planck 2018 LambdaCDM (TT,TE,EE+lowE+lensing):
    # Omega_m = 0.3153, sigma_8 = 0.8111

    def growth_factor(om, z):
        def rhs(n, y):
            oma = om*np.exp(-3*n)/(om*np.exp(-3*n) + 1 - om)
            return [y[1], -(2 - 1.5*oma)*y[1] + 1.5*oma*y[0]]
        sol = solve_ivp(rhs, [np.log(1e-3), 0.], [1e-3, 1e-3], rtol=1e-10, atol=1e-14, dense_output=True)
        return float(sol.sol(-np.log1p(z))[0]/sol.sol(0.)[0])
    result['planck_sigma8_at_z_ref'] = {'omega_m': .3153, 'sigma8_today': .8111, 'z_ref': fj.Z_NORM,
                                        'sigma8_at_z_ref': .8111*growth_factor(.3153, fj.Z_NORM)}
    print(json.dumps(result['planck_sigma8_at_z_ref']))
    store('lcdm', result)


# ----------------------------------------------------------------------------
# constrained minimisation over the seven nodes (and optionally the sound speed)
# ----------------------------------------------------------------------------
def node_constraint(cs2, positive_potential, margin=1e-9):
    """eps_H < 0 (w > -1) and, if requested, eps_H >= -(3/2)(1 - c_s^2) (V >= 0), on the grid;
    c_s^2 may be one number or an array over the grid.  Without V >= 0 the lower bound is
    the prior edge of the nodes, which is never reached."""
    lower = -1.5*(1 - cs2) + margin if positive_potential else fj.PRIOR_EPS[0] + margin
    return LinearConstraint(G.SMAT, np.full(len(XG), lower), np.full(len(XG), -margin))


def feasible(eps, cs2=0., positive_potential=True, omega_f=None):
    """The restrictions of the fit on its grid: eps_H < 0 and, if requested, V >= 0."""
    grid = G.SMAT @ np.asarray(eps)
    if not np.all(grid < 0):
        return False
    if omega_f is not None:
        cs2 = tied_cs2(-1 - grid, np.exp(G.IMAT @ np.asarray(eps)), omega_f)
    return bool(not positive_potential or np.all(grid >= -1.5*(1 - cs2)))


def minimise_nodes(start, cs2=0., omega_f=None, positive_potential=True, rows=None, free_cs2=False):
    """SLSQP over the seven nodes, with value and central-difference gradient taken from
    one batched call.  free_cs2: the constant sound speed is an eighth variable
    (t = 1e7 c_s^2).  Returns (eps, cs2, chi2 of the fast solver)."""
    step = 1e-5
    nvar = 8 if free_cs2 else 7

    def value_and_gradient(x):
        trial = np.repeat(x[None, :], 2*nvar + 1, axis=0)
        trial[1 + np.arange(nvar), np.arange(nvar)] += step
        trial[1 + nvar + np.arange(nvar), np.arange(nvar)] -= step
        c = np.maximum(trial[:, 7], 0.)*1e-7 if free_cs2 else cs2
        values = spline_profile(trial[:, :7], c, omega_f, FAST, rows=rows)['chi2']
        return float(values[0]), (values[1:1 + nvar] - values[1 + nvar:])/(2*step)

    if free_cs2:
        matrix = np.zeros((2*len(XG), 8))
        matrix[:len(XG), :7] = G.SMAT
        matrix[len(XG):, :7] = G.SMAT
        matrix[len(XG):, 7] = -1.5e-7 if positive_potential else 0.
        low = -1.5 + 1e-9 if positive_potential else fj.PRIOR_EPS[0] + 1e-9
        constraint = LinearConstraint(matrix, np.r_[np.full(len(XG), -np.inf), np.full(len(XG), low)],
                                      np.r_[np.full(len(XG), -1e-9), np.full(len(XG), np.inf)])
        bounds = [(fj.PRIOR_EPS[0] + 1e-6, fj.PRIOR_EPS[1] - 1e-6)]*7 + [(1e-5, 1e5)]
        x0 = np.r_[start, cs2*1e7]
    else:
        constraint = node_constraint(cs2, positive_potential)
        bounds = [(fj.PRIOR_EPS[0] + 1e-6, fj.PRIOR_EPS[1] - 1e-6)]*7
        x0 = np.asarray(start, float)
    for _ in range(1 if omega_f is None else 3):
        if omega_f is not None:
            # tied model: V >= 0 is eps_H >= -(3/2)(1 - c_s^2) with c_s^2 depending weakly on
            # the nodes; it is evaluated at the current point and the search is repeated
            constraint = node_constraint(tied_cs2(-1 - G.SMAT @ x0, np.exp(G.IMAT @ x0), omega_f),
                                         positive_potential)
        best = minimize(value_and_gradient, x0, jac=True, method='SLSQP', bounds=bounds,
                        constraints=[constraint], options={'maxiter': 400, 'ftol': 1e-10})
        x0 = best.x
    eps = best.x[:7].copy()
    fitted_cs2 = float(best.x[7]*1e-7) if free_cs2 else cs2
    value = float(best.fun)
    # SLSQP can miss an active linear constraint by a few floating-point units.
    # Move such a point into the original domain and RECOMPUTE its likelihood.
    # Never accept an exterior point by weakening the physical feasibility test.
    if omega_f is None and not feasible(eps, fitted_cs2, positive_potential):
        grid = G.SMAT @ eps
        lower = -1.5*(1-fitted_cs2) if positive_potential else fj.PRIOR_EPS[0]
        violation = max(float(np.max(grid)), float(np.max(lower-grid)), 0.)
        if np.isfinite(value) and violation <= 1e-7:
            centre = .5*lower
            alpha = 0.
            # A convex combination with a constant interior spline preserves the
            # node basis and handles upper and lower active bounds simultaneously.
            above = grid > -2e-9
            below = grid < lower+2e-9
            if np.any(above):
                alpha = max(alpha, float(np.max((grid[above]+2e-9)/(grid[above]-centre))))
            if np.any(below):
                alpha = max(alpha, float(np.max((lower+2e-9-grid[below])/(centre-grid[below]))))
            eps = (1-alpha)*eps + alpha*centre
            value = float(spline_profile(eps[None, :], fitted_cs2, omega_f, FAST, rows=rows)['chi2'][0])
    return eps, fitted_cs2, value


def moved_inside(eps, cs2=0., omega_f=None):
    """The node vector raised by a constant, if needed, until V >= 0 holds on the grid;
    a constant added to the nodes is added to the whole spline."""
    eps = np.asarray(eps, float)
    for _ in range(2):
        grid = G.SMAT @ eps
        bound = cs2 if omega_f is None else tied_cs2(-1 - grid, np.exp(G.IMAT @ eps), omega_f)
        eps = eps + max(0., float(np.max(-1.5*(1 - bound) + 2e-9 - grid)))
    return eps


def describe(eps, cs2=0., omega_f=None, rows=None):
    """Full description of a profile minimum: parameters, chi^2 (fine solver) and, for a
    constant sound speed with all data, the chi^2 of fit_joint.py."""
    prof = spline_profile(np.asarray(eps)[None, :], cs2, omega_f, FINE, rows=rows)
    out = {'eps_nodes': [float(v) for v in eps], 'A': float(prof['A'][0]), 'sigma8_ref': float(prof['sigma8'][0]),
           'M': float(prof['M'][0]), 'chi2_rk4_solver': float(prof['chi2'][0]),
           'chi2_bao': float(prof['bao'][0]), 'chi2_sn': float(prof['sn'][0]), 'chi2_growth': float(prof['growth'][0]),
           'max_w': float(np.max(-1 - 2*(G.SMAT @ eps)/3)), 'min_w': float(np.min(-1 - 2*(G.SMAT @ eps)/3)),
           'q0': float(-1 - G.BASIS(0.) @ eps)}
    if rows is None:
        out['chi2'], parts = fit_chi2(eps, out['A'], cs2, out['sigma8_ref'], out['M'], omega_f)
        out['chi2_bao'], out['chi2_sn'], out['chi2_growth'] = parts
    if omega_f is None:
        out['c_s2'] = float(cs2)
    else:
        z = np.array([0., fj.Z_NORM])
        q = -1 - G.BASIS(np.log1p(z)) @ eps
        inv = G.hubble_at(np.r_[eps, 0.][None, :], np.log1p(z), grid=XG)[0]
        out['omega_f'] = float(omega_f)
        out['c_s2_today'], out['c_s2_at_z_ref'] = [float(v) for v in tied_cs2(q, inv, omega_f)]
    return out


def starting_points(count, cs2=0., positive_potential=True, random=0, perturbed=0, seed=1):
    """Node vectors to start from: the optimised vector of the `nodes` stage; the stored
    posterior samples of lowest chi^2 that are admissible for this sound speed; `random`
    further admissible samples drawn from the stored posterior; and `perturbed` copies of
    the first vector with Gaussian noise of standard deviation 0.25 on every node."""
    stored = fj.load_result('n1_posterior', 'posterior', 'physical', 1)
    points = [fj.load_result('best_1', 'best_start', 'physical', 1)['parameters'][:7]]
    for row in np.argsort(stored['chi2']):
        if len(points) >= count:
            break
        if feasible(stored['chain'][row, :7], cs2, positive_potential):
            points.append(stored['chain'][row, :7])
    rng = np.random.default_rng(seed)
    for row in rng.permutation(len(stored['chain'])):
        if len(points) >= count + random:
            break
        if feasible(stored['chain'][row, :7], cs2, positive_potential):
            points.append(stored['chain'][row, :7])
    while len(points) < count + random + perturbed:
        trial = points[0] + .25*rng.standard_normal(7)
        if feasible(trial, cs2, positive_potential):
            points.append(trial)
    return points


def command_bestfit():
    result = {}
    for label, positive in (('with_V_nonnegative', True), ('without_V_restriction', False)):
        found = []
        for n, start in enumerate(starting_points(12, 2e-7, positive, random=12, perturbed=8)):
            for first in (1.7e-7, 1e-9):
                eps, cs2, value = minimise_nodes(start, first, positive_potential=positive, free_cs2=True)
                if feasible(eps, cs2, positive):
                    found.append((value, eps, cs2))
                print(f'  {label} start {n:2d} (c_s^2 from {first:.0e}): chi2 = {value:.4f}  c_s^2 = {cs2:.3e}',
                      flush=True)
        found.sort(key=lambda item: item[0])
        value, eps, cs2 = found[0]
        result[label] = describe(eps, cs2)
        result[label]['starts'] = len(found)
        result[label]['starts_within_0.001_of_minimum'] = int(sum(item[0] - value < 1e-3 for item in found))
        result[label]['chi2_of_all_starts'] = [float(item[0]) for item in found]
        result[label]['c_s2_of_all_starts'] = [float(item[2]) for item in found]
        result[label]['parameters'], result[label]['measurements'] = 11, int(fj.NDATA)
        print(label, json.dumps({k: v for k, v in result[label].items() if not k.endswith('all_starts')}, indent=1))
    result['nodes_stage_of_fit_joint'] = float(fj.chi2_total(
        fj.load_result('best_1', 'best_start', 'physical', 1)['parameters'], fj.node_positions(1)))
    store('bestfit', result)


# ----------------------------------------------------------------------------
# stage: profile likelihood of the constant sound speed
# ----------------------------------------------------------------------------
U_PROFILE = np.r_[-12., -9., -8.5, -8., -7.75, -7.5, np.arange(-7.4, -5.799, .05)]


def command_profile():
    best_all = json.loads(OUT_JSON.read_text())['bestfit']
    result = {}
    for label, positive in (('with_V_nonnegative', True), ('without_V_restriction', False)):
        rows, previous = [], np.array(best_all[label]['eps_nodes'])
        anchors = starting_points(4, 0., positive) + [previous]
        for u in U_PROFILE:
            cs2 = 10.0**u
            starts = [previous] + anchors
            if positive:                   # a larger sound speed moves the boundary V = 0
                starts = [moved_inside(p, cs2) for p in starts]
            found = []
            for start in starts:
                if not feasible(start, cs2, positive):
                    continue
                eps, _, value = minimise_nodes(start, cs2, positive_potential=positive)
                if feasible(eps, cs2, positive):
                    found.append((value, eps))
            found.sort(key=lambda item: item[0])
            previous = found[0][1]
            info = describe(previous, cs2)
            info['u'], info['spread_of_starts'] = float(u), float(found[-1][0] - found[0][0])
            rows.append(info)
            print(f'  {label} u = {u:6.2f}: chi2 = {info["chi2"]:.4f} (BAO {info["chi2_bao"]:.3f}, SN '
                  f'{info["chi2_sn"]:.3f}, growth {info["chi2_growth"]:.3f}) sigma8 = {info["sigma8_ref"]:.4f}'
                  f' starts {len(found)} spread {info["spread_of_starts"]:.1e}', flush=True)
        u = np.array([r['u'] for r in rows])
        chi2 = np.array([r['chi2'] for r in rows])
        minimum = min(float(chi2.min()), best_all[label]['chi2'])
        delta = chi2 - minimum
        table('profile_' + label, {'u': u, 'cs2': 10.0**u, 'chi2': chi2, 'delta_chi2': delta,
                                   'chi2_rk4_solver': [r['chi2_rk4_solver'] for r in rows],
                                   'chi2_bao': [r['chi2_bao'] for r in rows], 'chi2_sn': [r['chi2_sn'] for r in rows],
                                   'chi2_growth': [r['chi2_growth'] for r in rows],
                                   'sigma8_ref': [r['sigma8_ref'] for r in rows], 'A': [r['A'] for r in rows],
                                   'q0': [r['q0'] for r in rows], 'max_w': [r['max_w'] for r in rows],
                                   **{f'eps{i}': [r['eps_nodes'][i] for r in rows] for i in range(7)}})
        start = int(np.argmin(chi2))
        limits = {}
        for name, level in LEVELS.items():
            cross = crossing(u[start:], delta[start:], level)
            limits[name] = None if cross is None else 10.0**cross
        result[label] = {'chi2_minimum': minimum, 'c_s2_at_minimum': best_all[label]['c_s2'],
                         'delta_chi2_at_zero_sound_speed': float(delta[0]), 'upper_limits_c_s2': limits,
                         'largest_spread_between_starts_where_delta_chi2_below_10': float(max(
                             r['spread_of_starts'] for r, dlt in zip(rows, delta) if dlt < 10))}
        print(label, json.dumps(result[label], indent=1))
    store('profile', result)


def command_leaveout():
    """The same profile without the growth measurement at z = 0.07, with V >= 0.  The
    chi^2 of 51 measurements is evaluated with the RK4 solver only."""
    kept = np.flatnonzero(GZ > .1)
    best_all = json.loads(OUT_JSON.read_text())['bestfit']['with_V_nonnegative']
    anchors = starting_points(4, 0.) + [np.array(best_all['eps_nodes'])]
    found = []
    for start in anchors:                              # minimum with the sound speed free
        for first in (best_all['c_s2'], 5e-7, 1e-9):
            eps, cs2, value = minimise_nodes(moved_inside(start, first), first, rows=kept, free_cs2=True)
            if feasible(eps, cs2):
                found.append((value, eps, cs2))
    found.sort(key=lambda item: item[0])
    best = describe(found[0][1], found[0][2], rows=kept)
    rows, previous = [], found[0][1]
    for u in U_PROFILE:
        cs2 = 10.0**u
        trials = []
        for start in [previous] + anchors:
            eps, _, value = minimise_nodes(moved_inside(start, cs2), cs2, rows=kept)
            if feasible(eps, cs2):
                trials.append((value, eps))
        trials.sort(key=lambda item: item[0])
        previous = trials[0][1]
        info = describe(previous, cs2, rows=kept)
        info['u'], info['spread_of_starts'] = float(u), float(trials[-1][0] - trials[0][0])
        rows.append(info)
        print(f'  u = {u:6.2f}: chi2 = {info["chi2_rk4_solver"]:.4f} sigma8 = {info["sigma8_ref"]:.4f} '
              f'starts {len(trials)} spread {info["spread_of_starts"]:.1e}', flush=True)
    u = np.array([r['u'] for r in rows])
    chi2 = np.array([r['chi2_rk4_solver'] for r in rows])
    minimum = min(float(chi2.min()), best['chi2_rk4_solver'])
    delta = chi2 - minimum
    table('profile_without_lowest_z', {'u': u, 'cs2': 10.0**u, 'chi2': chi2, 'delta_chi2': delta,
                                       'sigma8_ref': [r['sigma8_ref'] for r in rows],
                                       **{f'eps{i}': [r['eps_nodes'][i] for r in rows] for i in range(7)}})
    lowest = int(np.argmin(chi2))
    limits = {}
    for name, level in LEVELS.items():
        cross = crossing(u[lowest:], delta[lowest:], level)
        limits[name] = None if cross is None else 10.0**cross
    result = {'measurements': int(fj.NDATA) - 1, 'chi2_minimum': minimum, 'c_s2_at_minimum': best['c_s2'],
              'sigma8_ref_at_minimum': best['sigma8_ref'], 'starts_for_the_minimum': len(found),
              'starts_within_0.001_of_minimum': int(sum(item[0] - found[0][0] < 1e-3 for item in found)),
              'delta_chi2_at_zero_sound_speed': float(delta[0]), 'upper_limits_c_s2': limits,
              'largest_spread_between_starts_where_delta_chi2_below_10': float(max(
                  r['spread_of_starts'] for r, dlt in zip(rows, delta) if dlt < 10))}
    print(json.dumps(result, indent=1))
    store('leaveout', result)


# ----------------------------------------------------------------------------
# stage: models with a LambdaCDM-like expansion, on exact two-dimensional grids
# ----------------------------------------------------------------------------
U_GRID = np.r_[-12., -10., -9., np.arange(-8.5, -2.999, .05)]
OMEGA_GRID = np.arange(.2, .45001, .001)
NO_LOWEST = np.flatnonzero(GZ > .1)                 # growth data without the z = 0.07 point


def grid_models(model, growing_mode):
    """Profile chi^2 on the grid (u, Omega) for a model with a LambdaCDM-like expansion,
    for all data and without the lowest-redshift growth point.
    growing_mode: start with the rate of the pressureless LambdaCDM growing mode for this
    Omega instead of v/D = 1 (for constant f and V this is the model's own growing mode;
    for a constant sound speed it is only another convention).
    'constant': LambdaCDM expansion with matter fraction Omega and c_s^2 = 10^u.
    'minimal':  constant f and V; Omega is the kinetic density today in units of the
                critical density and 10^u = c_s^2(z=0), so that f/3H0^2 = 10^(u/2) Omega."""
    uu, oo = [a.ravel() for a in np.meshgrid(U_GRID, OMEGA_GRID, indexing='ij')]
    rate_table = lcdm_growing_mode(OMEGA_GRID) if growing_mode else None
    parts = ([], [])
    for first in range(0, len(uu), 1000):
        om, c0 = oo[first:first + 1000], 10.0**uu[first:first + 1000]
        if model == 'constant':
            inv_grid, q_pts, inv_pts = FINE.lcdm(om)
            cs2_pts = c0[:, None]
        else:
            inv_grid, q_pts, inv_pts, cs2_pts = FINE.minimal(om*np.sqrt(1 - c0), om*np.sqrt(c0))
        rate = np.interp(om, OMEGA_GRID, rate_table) if growing_mode else 1.0
        shape = FINE.shape(q_pts, inv_pts, cs2_pts, rate)
        parts[0].append(linear_profile(inv_grid, shape))
        parts[1].append(linear_profile(inv_grid, shape, NO_LOWEST))
    size = (len(U_GRID), len(OMEGA_GRID))
    return [{key: np.concatenate([p[key] for p in part]).reshape(size) for key in part[0]} for part in parts]


def weighted_quantile(values, weights, fraction=.95):
    order = np.argsort(values, axis=None)
    cdf = np.cumsum(weights.ravel()[order])
    return float(np.interp(fraction*cdf[-1], cdf, values.ravel()[order]))


def parabola_minimum(x, y):
    """Position and value of the minimum of the parabola through the lowest tabulated
    point and its two neighbours (the tabulated minimum itself at an end of the table)."""
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i]), float(y[i])
    a, b, c = np.polyfit(x[i - 1:i + 2] - x[i], y[i - 1:i + 2], 2)
    return (float(x[i] - b/(2*a)), float(c - b*b/(4*a))) if a > 0 else (float(x[i]), float(y[i]))


def analyse_grid(grid, model):
    """Profile over Omega (parabola through the three lowest grid values), marginal
    likelihood for a flat prior on Omega (and on A, sigma_8, M), and upper limits."""
    chi2 = grid['chi2']
    rows = np.arange(len(U_GRID))
    j = np.clip(np.argmin(chi2, axis=1), 1, chi2.shape[1] - 2)
    y0, y1, y2 = chi2[rows, j - 1], chi2[rows, j], chi2[rows, j + 1]
    curvature = y0 - 2*y1 + y2
    profile = y1 - .125*(y0 - y2)**2/curvature
    step = OMEGA_GRID[1] - OMEGA_GRID[0]
    omega_best = OMEGA_GRID[j] + .5*(y0 - y2)/curvature*step
    lowest = int(np.argmin(profile))
    u_lowest, reference = parabola_minimum(U_GRID, profile)
    delta = profile - reference
    weight = np.exp(-.5*(chi2 - chi2.min()) + grid['logvol'] - grid['logvol'][0, j[0]])
    marginal = weight.sum(axis=1)*step
    marginal = marginal/marginal[0]
    out = {'chi2_minimum': float(reference), 'u_at_minimum': float(u_lowest),
           'c_s2_at_minimum': float(10.0**u_lowest), 'omega_at_minimum': float(omega_best[lowest]),
           'sigma8_ref_at_minimum': float(grid['sigma8'][lowest, j[lowest]]),
           'chi2_at_zero_sound_speed': float(profile[0]), 'omega_at_zero_sound_speed': float(omega_best[0]),
           'sigma8_ref_at_zero_sound_speed': float(grid['sigma8'][0, j[0]]),
           'delta_chi2_at_zero_sound_speed': float(delta[0]),
           'marginal_likelihood_at_grid_end': float(marginal[-1]),
           'profile_upper_limits': {}, 'bayes_95_upper_limit_by_lower_prior_edge': {}}
    for name, level in LEVELS.items():
        cross = crossing(U_GRID[lowest:], delta[lowest:], level)
        out['profile_upper_limits'][name] = None if cross is None else 10.0**cross
    for edge in (-12., -10., -9., -8.):
        out['bayes_95_upper_limit_by_lower_prior_edge'][f'1e{edge:.0f}'] = 10.0**upper_limit(U_GRID, marginal, edge)
    # priors uniform in the sound speed or in its square instead of in the logarithm
    out['bayes_95_upper_limit_prior_uniform_in_c_s'] = 10.0**upper_limit(U_GRID, marginal*10.0**(.5*U_GRID), -12.)
    out['bayes_95_upper_limit_prior_uniform_in_c_s2'] = 10.0**upper_limit(U_GRID, marginal*10.0**U_GRID, -12.)
    if model == 'minimal':
        # f/3H0^2 = c_s(0) Omega: 95% quantile over the grid, prior uniform in u and Omega
        widths = np.gradient(U_GRID)
        widths[[0, -1]] *= .5
        measure = weight*widths[:, None]
        omega_f = 10.0**(.5*U_GRID[:, None])*OMEGA_GRID[None, :]
        out['bayes_95_upper_limit_omega_f'] = weighted_quantile(omega_f, measure)
        # profile likelihood of f/3H0^2 itself: at fixed f, 10^u = (f/3H0^2 / Omega)^2 on each column
        log_f = np.arange(-4.6, -2.5999, .01)
        columns_f = np.full((len(log_f), len(OMEGA_GRID)), np.inf)
        for col, omega in enumerate(OMEGA_GRID):
            u_here = 2*(log_f - np.log10(omega))
            inside = (u_here >= U_GRID[0]) & (u_here <= U_GRID[-1])
            columns_f[inside, col] = PchipInterpolator(U_GRID, chi2[:, col])(u_here[inside])
        jf = np.clip(np.argmin(columns_f, axis=1), 1, len(OMEGA_GRID) - 2)
        f0, f1, f2 = (columns_f[np.arange(len(log_f)), jf + shift] for shift in (-1, 0, 1))
        profile_f = f1 - .125*(f0 - f2)**2/(f0 - 2*f1 + f2)
        start_f = int(np.argmin(profile_f))
        out['profile_upper_limits_omega_f'] = {}
        for name, level in LEVELS.items():
            cross = crossing(log_f[start_f:], profile_f[start_f:] - reference, level)
            out['profile_upper_limits_omega_f'][name] = None if cross is None else 10.0**cross
        ratio = (1 + fj.Z_NORM)**6
        to_ref = lambda c0: None if c0 is None else float(c0/(c0 + (1 - c0)*ratio))
        out['profile_upper_limits_c_s2_at_z_ref'] = {k: to_ref(v) for k, v in out['profile_upper_limits'].items()}
        out['bayes_95_upper_limit_c_s2_at_z_ref'] = to_ref(out['bayes_95_upper_limit_by_lower_prior_edge']['1e-12'])
    columns = {'u': U_GRID, 'value': 10.0**U_GRID, 'delta_chi2': delta, 'chi2': profile, 'omega': omega_best,
               'sigma8_ref': grid['sigma8'][rows, j], 'marginal_likelihood': marginal}
    return out, columns


def command_grids():
    result = {}
    for model in ('constant', 'minimal'):
        result[model] = {}
        for label, mode in (('same_initial_rate', False), ('growing_mode_initial_rate', True)):
            full, reduced = grid_models(model, mode)
            summary, columns = analyse_grid(full, model)
            summary_reduced, columns_reduced = analyse_grid(reduced, model)
            columns.update({key + '_without_lowest_z': value for key, value in columns_reduced.items()
                            if key not in ('u', 'value')})
            table(f'grid_{model}_{label}', columns)
            result[model][label] = {'all_data': summary, 'without_growth_point_at_z_0.07': summary_reduced}
            print(model, label, json.dumps(result[model][label], indent=1), flush=True)
    result['grid'] = {'u': [float(U_GRID[0]), float(U_GRID[-1])], 'omega': [float(OMEGA_GRID[0]), float(OMEGA_GRID[-1])],
                      'omega_step': float(OMEGA_GRID[1] - OMEGA_GRID[0]), 'prior': 'uniform in u and in Omega'}
    store('grids', result)


# ----------------------------------------------------------------------------
# stage: coverage of the profile limits, by simulation
# ----------------------------------------------------------------------------
def command_coverage(simulations=4000):
    """How often the profile limit lies above the true sound speed, for the model with a
    LambdaCDM expansion and a constant sound speed (v/D = 1 at z = 2.33).  Data sets are
    drawn from the model at a chosen true sound speed, with the other parameters at their
    best values for the real data, and each is analysed on the grid like the real data."""
    rng = np.random.default_rng(fj.SEED + 1)
    # predictions of every grid model for unit A and sigma_8 and zero supernova offset
    inv_grid, q_pts, inv_pts = FINE.lcdm(OMEGA_GRID)
    dm1 = distances(inv_grid)
    t_bao = np.empty((len(OMEGA_GRID), 13))
    t_bao[:, 0:12:2] = G.interpolate_rows(dm1, ANISO[:, 0])
    t_bao[:, 1:12:2] = G.interpolate_rows(inv_grid, ANISO[:, 0])
    t_bao[:, 12] = np.cbrt(BGS[0]*G.interpolate_rows(dm1, [BGS[0]])[:, 0]**2*G.interpolate_rows(inv_grid, [BGS[0]])[:, 0])
    cov_bao = np.zeros((13, 13))
    for j, row in enumerate(ANISO):
        cov_bao[2*j:2*j + 2, 2*j:2*j + 2] = [[row[2]**2, row[5]*row[2]*row[4]], [row[5]*row[2]*row[4], row[4]**2]]
    cov_bao[12, 12] = BGS[2]**2
    obs_bao = np.r_[np.column_stack([ANISO[:, 1], ANISO[:, 3]]).ravel(), BGS[1]]
    cinv_bao = np.linalg.inv(cov_bao)
    g_bao = t_bao @ cinv_bao
    a_bao = np.einsum('oi,oi->o', g_bao, t_bao)
    m_sn = 5*np.log10((1 + SNZ)*G.interpolate_rows(dm1, SNZ))
    one = np.ones(len(SNZ))
    weight_sn = one @ SNCINV/(one @ SNCINV @ one)                    # offset = weight_sn @ (mu - m)
    project = SNCINV - np.outer(SNCINV @ one, weight_sn)
    shapes = np.stack([FINE.shape(q_pts, inv_pts, np.full((len(OMEGA_GRID), 1), 10.0**u)) for u in U_GRID])
    h = shapes @ GCINV
    ss = np.einsum('uoi,uoi->uo', h, shapes)

    def profile(d_bao, mu, y):
        """chi^2 minimised over A, sigma_8, M and Omega for each data set and each u."""
        chi_bao = np.einsum('ni,ij,nj->n', d_bao, cinv_bao, d_bao)[:, None] - (d_bao @ g_bao.T)**2/a_bao
        r = mu[:, None, :] - m_sn[None, :, :]
        chi_sn = np.einsum('noi,ij,noj->no', r, project, r)
        chi_g = np.einsum('ni,ij,nj->n', y, GCINV, y)[:, None, None] - np.einsum('ni,uoi->nuo', y, h)**2/ss
        return (chi_g + (chi_bao + chi_sn)[:, None, :]).min(axis=2)
    real = profile(obs_bao[None, :], SNY[None, :], GY[None, :])[0]
    limit = json.loads(OUT_JSON.read_text())['grids']['constant']['same_initial_rate']['all_data'][
        'profile_upper_limits']['95% one-sided (Delta chi2 = 2.71)']
    chol = [np.linalg.cholesky(c) for c in (cov_bao, OBS['sn_cov'], GCOV)]
    result = {'simulations': simulations, 'model': 'LambdaCDM expansion, constant sound speed, v/D = 1 at z = 2.33',
              'chi2_minimum_of_the_data_on_this_grid': float(real.min()), 'cases': []}
    for target in (limit, .5*limit, 1e-12):
        iu = int(np.argmin(np.abs(U_GRID - np.log10(target))))
        # true model: the best Omega of the real data at this sound speed, with its A, sigma_8, M
        chi_bao = obs_bao @ cinv_bao @ obs_bao - (g_bao @ obs_bao)**2/a_bao
        r = SNY[None, :] - m_sn
        chi_g = GY @ GCINV @ GY - (h[iu] @ GY)**2/ss[iu]
        jo = int(np.argmin(chi_bao + np.einsum('oi,ij,oj->o', r, project, r) + chi_g))
        amplitude, sigma8, offset = g_bao[jo] @ obs_bao/a_bao[jo], h[iu, jo] @ GY/ss[iu, jo], weight_sn @ r[jo]
        missed = np.zeros(2, int)
        at_zero = 0
        for first in range(0, simulations, 250):
            n = min(250, simulations - first)
            prof = profile(amplitude*t_bao[jo] + rng.standard_normal((n, 13)) @ chol[0].T,
                           m_sn[jo] + offset + rng.standard_normal((n, len(SNZ))) @ chol[1].T,
                           sigma8*shapes[iu, jo] + rng.standard_normal((n, len(GY))) @ chol[2].T)
            best = prof.argmin(axis=1)
            rise = prof[:, iu] - prof.min(axis=1)
            missed += [int(np.sum((best < iu) & (rise > level))) for level in (2.706, 3.841)]
            at_zero += int(np.sum(best <= 2))                      # no preference for a sound speed above 1e-9
        result['cases'].append({
            'true_c_s2': float(10.0**U_GRID[iu]), 'true_omega': float(OMEGA_GRID[jo]), 'true_sigma8_ref': float(sigma8),
            'fraction_of_limits_above_truth_at_2.71': float(1 - missed[0]/simulations),
            'fraction_of_limits_above_truth_at_3.84': float(1 - missed[1]/simulations),
            'binomial_error_at_2.71': float(np.sqrt(max(missed[0], 1)*(1 - missed[0]/simulations))/simulations),
            'fraction_of_data_sets_with_best_c_s2_below_1e-9': float(at_zero/simulations)})
        print(json.dumps(result['cases'][-1]), flush=True)
    store('coverage', result)


# ----------------------------------------------------------------------------
# stage: constant prefactor with a free potential (sound speed tied to rho+p)
# ----------------------------------------------------------------------------
UF_PROFILE = np.r_[-8., -6., -5., -4.5, -4.25, np.arange(-4., -2.699, .05)]


def tied_minimum(u, starts):
    """Lowest chi^2 over the nodes at fixed f/3H0^2 = 10^u, from several starts."""
    omega_f = 10.0**u
    found = []
    for start in starts:
        start = moved_inside(start, omega_f=omega_f)
        if not feasible(start, omega_f=omega_f):
            continue
        eps, _, value = minimise_nodes(start, omega_f=omega_f)
        if feasible(eps, omega_f=omega_f):
            found.append((value, eps))
    found.sort(key=lambda item: item[0])
    return found


def command_tied():
    best_constant = np.array(json.loads(OUT_JSON.read_text())['bestfit']['with_V_nonnegative']['eps_nodes'])
    anchors = starting_points(3, 0.) + [best_constant]
    rows, previous = [], best_constant
    for u in UF_PROFILE:
        found = tied_minimum(u, [previous] + anchors)
        previous = found[0][1]
        info = describe(previous, omega_f=10.0**u)
        info['u'], info['spread_of_starts'] = float(u), float(found[-1][0] - found[0][0])
        rows.append(info)
        print(f'  log10 Omega_f = {u:6.2f}: chi2 = {info["chi2"]:.4f} (growth {info["chi2_growth"]:.3f}) '
              f'c_s^2(0) = {info["c_s2_today"]:.2e} c_s^2(z_ref) = {info["c_s2_at_z_ref"]:.2e} '
              f'sigma8 = {info["sigma8_ref"]:.4f} starts {len(found)} spread {info["spread_of_starts"]:.1e}', flush=True)
    u = np.array([r['u'] for r in rows])
    chi2 = np.array([r['chi2'] for r in rows])
    lowest = int(np.argmin(chi2))
    best = rows[lowest]
    if 0 < lowest < len(u) - 1:               # refine the minimum between the neighbouring grid values
        refined = minimize_scalar(lambda t: tied_minimum(t, [np.array(best['eps_nodes'])])[0][0],
                                  bounds=(u[lowest - 1], u[lowest + 1]), method='bounded', options={'xatol': 1e-3})
        candidate = describe(tied_minimum(refined.x, [np.array(best['eps_nodes'])])[0][1], omega_f=10.0**refined.x)
        if candidate['chi2'] < best['chi2']:
            best = dict(candidate, u=float(refined.x))
    delta = chi2 - best['chi2']
    table('profile_tied', {'u': u, 'omega_f': 10.0**u, 'chi2': chi2, 'delta_chi2': delta,
                           'chi2_rk4_solver': [r['chi2_rk4_solver'] for r in rows],
                           'chi2_bao': [r['chi2_bao'] for r in rows], 'chi2_sn': [r['chi2_sn'] for r in rows],
                           'chi2_growth': [r['chi2_growth'] for r in rows],
                           'c_s2_today': [r['c_s2_today'] for r in rows],
                           'c_s2_at_z_ref': [r['c_s2_at_z_ref'] for r in rows],
                           'sigma8_ref': [r['sigma8_ref'] for r in rows], 'q0': [r['q0'] for r in rows],
                           **{f'eps{i}': [r['eps_nodes'][i] for r in rows] for i in range(7)}})
    limits = {}
    for name, level in LEVELS.items():
        cross = crossing(u[lowest:], delta[lowest:], level)
        limits[name] = None if cross is None else {
            'omega_f': 10.0**cross,
            'c_s2_today': float(10.0**np.interp(cross, u, np.log10([r['c_s2_today'] for r in rows]))),
            'c_s2_at_z_ref': float(10.0**np.interp(cross, u, np.log10([r['c_s2_at_z_ref'] for r in rows])))}
    result = {'best_fit': best, 'delta_chi2_at_zero_sound_speed': float(delta[0]), 'upper_limits': limits,
              'largest_spread_between_starts_where_delta_chi2_below_10': float(max(
                  r['spread_of_starts'] for r, dlt in zip(rows, delta) if dlt < 10)),
              'parameters': 11, 'measurements': int(fj.NDATA)}
    print(json.dumps(result, indent=1))
    store('tied', result)


# ----------------------------------------------------------------------------
# stage: posterior samples with the three linear parameters integrated analytically
# ----------------------------------------------------------------------------
CHAINS = {  # name: model, V >= 0 imposed, prior range of u, lower prior edges compared, largest initial u
    # the model and the prior of fit_joint.py
    'constant': ('constant', True, PRIOR_U, (-12., -10., -9., -8.), -6.5),
    # the same without the restriction V >= 0
    'constant_any_V': ('constant', False, PRIOR_U, (-12., -10., -9., -8.), -6.5),
    # constant prefactor, u = log10(f/3H0^2); the range corresponds to 1e-12 < c_s^2 < 1e-2
    # for c_s = f/rho_kin with rho_kin = 0.3 x 3H0^2
    'tied': ('tied', True, (-6.5, -1.5), (-6.5, -5.5, -5., -4.5), -3.5),
}
WALKERS, CHAIN_THIN = 128, 10


def chain_log_prob(points, model, positive, prior):
    """Log posterior density of (seven nodes, u) for an array of points (n, 8), with
    flat priors on A, sigma_8 and M integrated out."""
    points = np.atleast_2d(points)
    eps, u = points[:, :7], points[:, 7]
    grid = eps @ G.SMAT.T
    ok = (np.all((eps > fj.PRIOR_EPS[0]) & (eps < fj.PRIOR_EPS[1]), axis=1) & (u >= prior[0]) & (u <= prior[1])
          & np.all(grid < 0, axis=1))
    value = 10.0**u
    if positive:
        with np.errstate(all='ignore'):
            cs2 = (tied_cs2(-1 - grid, np.exp(eps @ G.IMAT.T), value[:, None]) if model == 'tied'
                   else value[:, None])
            ok &= np.all(grid >= -1.5*(1 - cs2), axis=1)
    out = np.full(len(points), -np.inf)
    if ok.any():
        prof = (spline_profile(eps[ok], omega_f=value[ok], solver=FAST) if model == 'tied'
                else spline_profile(eps[ok], value[ok], solver=FAST))
        logp = -.5*prof['chi2'] + prof['logvol']
        out[ok] = np.where(np.isfinite(logp), logp, -np.inf)
    return out


def chain_file(name):
    folder = ROOT / 'extensions_chains'
    folder.mkdir(exist_ok=True)
    return folder / f'{name}.npz'


def command_chain():
    """python3 extensions.py chain <name> <steps>: run the chain <name>, or extend it."""
    import emcee
    name, steps = sys.argv[2], int(sys.argv[3])
    model, positive, prior, _, initial_top = CHAINS[name]
    if steps <= 0:
        raise ValueError('Additional steps must be positive')
    from run_identity import source_identity
    identity = source_identity()
    path = chain_file(name)
    sampler = emcee.EnsembleSampler(WALKERS, 8, chain_log_prob, args=(model, positive, prior), vectorize=True)
    if path.exists():
        saved = np.load(path)
        if 'source_identity' not in saved or str(saved['source_identity']) != identity:
            raise RuntimeError('Chain input/source identity mismatch; use a fresh package directory.')
        if 'rng_keys' not in saved:
            raise RuntimeError('Legacy chain lacks RNG state; start a fresh chain in this package.')
        position, kept, log_kept, done = saved['position'], [saved['chain']], [saved['log_prob']], int(saved['steps'])
    else:
        # start around the constrained best fit, moved inside the admissible region
        best = np.array(json.loads(OUT_JSON.read_text())['bestfit']['with_V_nonnegative']['eps_nodes'])
        rng = np.random.default_rng(fj.SEED)
        position = np.empty((0, 8))
        while len(position) < WALKERS:
            trial = np.column_stack([best + .03 + .01*rng.standard_normal((WALKERS, 7)),
                                     rng.uniform(prior[0], initial_top, WALKERS)])
            position = np.vstack([position, trial[np.isfinite(chain_log_prob(trial, model, positive, prior))]])
        position, kept, log_kept, done = position[:WALKERS], [], [], 0
    sampler.random_state = (fj.unpack_random_state(saved) if path.exists()
                            else np.random.RandomState(fj.SEED).get_state())
    start = time.time()
    while steps > 0:
        block = min(steps, 500)
        state = sampler.run_mcmc(position, block, progress=False, skip_initial_state_check=True)
        position = state.coords
        retain = (np.arange(done+1, done+block+1) % CHAIN_THIN) == 0
        kept.append(sampler.get_chain()[retain])
        log_kept.append(sampler.get_log_prob()[retain])
        acceptance = float(np.mean(sampler.acceptance_fraction))
        sampler.reset()
        steps -= block
        done += block
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, chain=np.concatenate(kept), log_prob=np.concatenate(log_kept),
                     position=position, steps=done, source_identity=identity, **fj.pack_random_state(state.random_state))
        temporary.replace(path)
        print(f'  {name}: {done} steps, acceptance {acceptance:.3f}, {time.time() - start:.0f} s', flush=True)


def chain_summary(name):
    import emcee
    model, positive, prior, edges, _ = CHAINS[name]
    saved = np.load(chain_file(name))
    chain, steps = saved['chain'], int(saved['steps'])        # (kept steps, walkers, 8)
    burn = len(chain)//3
    kept = chain[burn:]
    tau = emcee.autocorr.integrated_time(kept, tol=0)*CHAIN_THIN
    flat = kept.reshape(-1, 8)
    u = flat[:, 7]
    out = {'steps': steps, 'walkers': WALKERS, 'burn_in_steps': burn*CHAIN_THIN,
           'samples': int(len(flat)), 'prior_u': list(prior), 'V_nonnegative_imposed': positive,
           'largest_autocorrelation_time_steps': float(np.max(tau)),
           'autocorrelation_times_retained_per_walker': float((len(kept)*CHAIN_THIN)/np.max(tau)),
           'upper95': float(10.0**np.percentile(u, 95)),
           'upper95_in_four_consecutive_quarters': [float(10.0**np.percentile(part[:, :, 7], 95))
                                                    for part in np.array_split(kept, 4)],
           'upper95_by_lower_prior_edge': {f'{edge:g}': float(10.0**np.percentile(u[u >= edge], 95))
                                           for edge in edges}}
    # sigma_8 from its conditional Gaussian, and q today, on a subset of the samples
    rng = np.random.default_rng(fj.SEED)
    pick = flat[rng.choice(len(flat), min(20000, len(flat)), replace=False)]
    sigma8 = np.empty(len(pick))
    for first in range(0, len(pick), 1000):
        part = pick[first:first + 1000]
        prof = (spline_profile(part[:, :7], omega_f=10.0**part[:, 7], solver=FAST) if model == 'tied'
                else spline_profile(part[:, :7], 10.0**part[:, 7], solver=FAST))
        sigma8[first:first + 1000] = prof['sigma8'] + prof['sigma8_error']*rng.standard_normal(len(part))
    out['sigma8_ref_16_50_84'] = [float(v) for v in np.percentile(sigma8, [16, 50, 84])]
    q0 = -1 - pick[:, :7] @ G.BASIS(0.)
    out['q0_16_50_84'] = [float(v) for v in np.percentile(q0, [16, 50, 84])]
    w_grid = -1 - 2*(pick[:, :7] @ G.SMAT.T)/3
    out['max_w_16_50_84'] = [float(v) for v in np.percentile(w_grid.max(axis=1), [16, 50, 84])]
    out['w_at_z_2.33_2.5_16_50_84_97.5'] = [float(v) for v in np.percentile(w_grid[:, -1], [2.5, 16, 50, 84, 97.5])]
    out['w_at_z_0_2.5_16_50_84_97.5'] = [float(v) for v in np.percentile(w_grid[:, 0], [2.5, 16, 50, 84, 97.5])]
    out['fraction_with_largest_w_above_z_1.484'] = float(np.mean(ZG[w_grid.argmax(axis=1)] > 1.484))
    if model == 'constant':
        admissible = np.all(w_grid <= -10.0**pick[:, 7, None], axis=1)
        out['fraction_with_V_nonnegative'] = float(np.mean(admissible))
        if 200 < admissible.sum() < len(pick):
            out['upper95_of_samples_with_V_nonnegative'] = float(10.0**np.percentile(pick[admissible, 7], 95))
    if model == 'tied':
        z = np.array([0., fj.Z_NORM])
        q = -1 - pick[:, :7] @ G.BASIS(np.log1p(z)).T
        inv = G.hubble_at(pick, np.log1p(z), grid=XG)
        cs2 = tied_cs2(q, inv, 10.0**pick[:, 7, None])
        out['upper95_c_s2_today'], out['upper95_c_s2_at_z_ref'] = [float(v) for v in np.percentile(cs2, 95, axis=0)]
    top = prior[0] + (4. if model == 'tied' else 6.2)
    hist, bins = np.histogram(u, bins=np.linspace(prior[0], top, 94), density=True)
    table(f'chain_{name}_density', {'u': bins, 'density': np.r_[hist, 0.]})
    return out


def command_chains():
    """Summaries of the chains that have been run."""
    result = {}
    for name in CHAINS:
        if chain_file(name).exists():
            result[name] = chain_summary(name)
            print(name, json.dumps(result[name], indent=1), flush=True)
    stored = fj.load_result('n1_posterior', 'posterior', 'physical', 1)['chain']
    result['stored_posterior_of_fit_joint'] = {
        'upper95': float(np.percentile(stored[:, I_CS2], 95)),
        'sigma8_ref_16_50_84': [float(v) for v in np.percentile(stored[:, I_S8], [16, 50, 84])]}
    store('chains', result)


# ----------------------------------------------------------------------------
# stage: diagnostics of the stored posterior of fit_joint.py
# ----------------------------------------------------------------------------
def percentiles(values, levels=(16, 50, 84)):
    return [float(v) for v in np.percentile(values, levels)]


def command_posterior():
    chain = fj.load_result('n1_posterior', 'posterior', 'physical', 1)['chain']
    eps, cs2 = chain[:, :7], chain[:, I_CS2]
    limit = float(np.percentile(cs2, 95))
    result = {'samples': len(chain), 'upper95': limit, 'sigma8_ref_16_50_84': percentiles(chain[:, I_S8]),
              'A_16_50_84': percentiles(chain[:, 7])}

    def conditional(mask):
        return {'fraction_of_samples': float(np.mean(mask)), 'upper95': float(np.percentile(cs2[mask], 95))}

    # other priors by reweighting: the samples have a prior uniform in log10 c_s^2
    result['other_priors'] = {}

    def weighted_limit(values, weight):
        ranked = np.argsort(values)
        cdf = np.cumsum(weight[ranked])
        return float(np.interp(.95*cdf[-1], cdf, values[ranked]))
    quarters = np.array_split(np.arange(len(cs2)), 4)          # the samples are stored in the order of the steps
    result['upper95_in_four_consecutive_quarters'] = [float(np.percentile(cs2[part], 95)) for part in quarters]
    for name, weight in (('uniform in c_s', np.sqrt(cs2)), ('uniform in c_s^2', cs2)):
        result['other_priors'][name] = {
            'upper95': weighted_limit(cs2, weight),
            'upper95_in_four_consecutive_quarters': [weighted_limit(cs2[part], weight[part]) for part in quarters],
            'kish_effective_samples': float(weight.sum()**2/(weight**2).sum())}
    result['upper95_by_lower_prior_edge'] = {f'1e{edge}': float(np.percentile(cs2[cs2 >= 10.0**edge], 95))
                                             for edge in (-12, -10, -9, -8)}

    # 1. deceleration parameter at low redshift (z < 0.295 is an extrapolation of the spline)
    z_low = np.array([0., .05, .07, .15, .295, .51])
    q_low = -1 - eps @ G.BASIS(np.log1p(z_low)).T
    result['deceleration'] = {
        'z': z_low.tolist(), 'q_16_50_84': [percentiles(col) for col in q_low.T],
        'fraction_decelerating': [float(np.mean(col > 0)) for col in q_low.T],
        'limit_if_accelerating_today': conditional(q_low[:, 0] < 0),
        'limit_if_decelerating_today': conditional(q_low[:, 0] > 0),
        'limit_by_thirds_of_q_today': [conditional((q_low[:, 0] >= lo) & (q_low[:, 0] <= hi)) for lo, hi in
                                       zip(np.percentile(q_low[:, 0], [0, 100/3, 200/3]),
                                           np.percentile(q_low[:, 0], [100/3, 200/3, 100]))],
        'correlation_of_q_today_with_log10_cs2': float(np.corrcoef(q_low[:, 0], np.log10(cs2))[0, 1])}
    zq = np.linspace(0, 2.33, 234)
    table('deceleration', dict(zip(['z', 'lo95', 'lo68', 'median', 'hi68', 'hi95'],
                                   [zq, *np.percentile(-1 - eps @ G.BASIS(np.log1p(zq)).T, G.PERCENTILES, axis=0)])))

    # 2. distance from the boundary V = 0, i.e. w = -c_s^2
    w_grid = -1 - 2*(eps @ G.SMAT.T)/3
    w_max, where = w_grid.max(axis=1), ZG[w_grid.argmax(axis=1)]
    result['boundary'] = {
        'max_w_16_50_84': percentiles(w_max), 'redshift_of_max_w_16_50_84': percentiles(where),
        'fraction_with_max_w_above': {str(t): float(np.mean(w_max > t)) for t in (-.1, -.05, -.02, -.01)},
        'limit_if_max_w_above_-0.02': conditional(w_max > -.02),
        'limit_if_max_w_below_-0.02': conditional(w_max <= -.02),
        'w_at_last_node_16_50_84': percentiles(-1 - 2*eps[:, 6]/3)}

    # 3. prefactor and potential over the whole fitted range, f a^3 and V normalised at z_ref
    zz = np.unique(np.r_[np.linspace(0, 2.33, 234), G.Z_NODES])
    energy = 1/G.hubble_at(chain, np.log1p(zz), grid=XG)**2          # E^2 = (H/H0)^2
    q = -1 - eps @ G.BASIS(np.log1p(zz)).T
    prefactor = 2*(1 + q)*np.sqrt(cs2[:, None])/(1 - cs2[:, None])*energy/(1 + zz)**3   # f a^3/H0^2
    potential = (3 - 2*(1 + q)/(1 - cs2[:, None]))*energy                            # V/H0^2
    anchor = int(np.flatnonzero(zz == fj.Z_NORM)[0])
    fq = G.band_table('ext_prefactor_full_range', zz, prefactor/prefactor[:, anchor, None])
    vq = G.band_table('ext_potential_full_range', zz, potential/potential[:, anchor, None])
    G.band_table('ext_potential_over_critical_density_today', zz, potential/3)
    G.band_table('ext_w_full_range', zz, (2*q - 1)/3)
    inside_f = (fq[0] <= 1) & (fq[4] >= 1)
    inside_v = (vq[0] <= 1) & (vq[4] >= 1)

    def intervals(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
        return [[float(zz[a]), float(zz[b - 1])] for a, b in zip(edges[::2], edges[1::2])]
    nodes = np.searchsorted(zz, G.Z_NODES)
    result['reconstruction'] = {
        'range': [0., 2.33], 'normalised_at': fj.Z_NORM,
        'redshifts_where_95_band_contains_f_a3_constant': intervals(inside_f),
        'redshifts_where_95_band_contains_V_constant': intervals(inside_v),
        'nodes': G.Z_NODES.tolist(),
        'prefactor_at_nodes_2.5_16_50_84_97.5': fq[:, nodes].T.tolist(),
        'potential_at_nodes_2.5_16_50_84_97.5': vq[:, nodes].T.tolist(),
        'prefactor_today_2.5_16_50_84_97.5': fq[:, 0].tolist(),
        'potential_today_2.5_16_50_84_97.5': vq[:, 0].tolist(),
        'V_over_3H0^2_at_z_ref_16_50_84': percentiles(potential[:, anchor]/3),
        'V_over_3H0^2_at_last_node_16_50_84': percentiles(potential[:, -1]/3)}

    # 4. Jeans threshold read off the growth system: the bracket of the velocity equation
    #    vanishes when c_s^2 k^2/(aH)^2 = (3/2)(1+w) = 1+q
    inverse = G.hubble_at(chain, np.log1p(GZ), grid=XG)
    qg = -1 - eps @ G.BASIS(np.log1p(GZ)).T
    result['jeans'] = {}
    for name, k in (('k_eff', GK), ('k_max', GKMAX)):
        base = (1/(G.C_OVER_H0*inverse*(1 + GZ)*k))**2
        for label, factor in (('1+q', 1 + qg), ('(5+2q)/4', (5 + 2*qg)/4)):
            lo, med, hi = np.percentile(base*factor, [16, 50, 84], axis=0)
            i = int(np.argmin(med))
            result['jeans'][f'{name}, factor {label}'] = {
                'z': float(GZ[i]), 'k': float(k[i]), 'lo68': float(lo[i]), 'median': float(med[i]),
                'hi68': float(hi[i]), 'median_over_upper95': float(med[i]/limit)}
            if name == 'k_eff' and label == '1+q':
                table('jeans_threshold', {'z': GZ, 'keff': GK, 'lo68': lo, 'median': med, 'hi68': hi})
    # comoving Jeans wavenumber k_J = sqrt(1+q) aH/c_s for a sound speed at the upper limit
    z_j = np.array([0., .07, .18, .38, .51, 1., 2.])
    k_j = (np.sqrt(1 + (-1 - eps @ G.BASIS(np.log1p(z_j)).T))
           / (G.C_OVER_H0*G.hubble_at(chain, np.log1p(z_j), grid=XG)*(1 + z_j)*np.sqrt(limit)))
    result['jeans_wavenumber_at_upper95'] = {'z': z_j.tolist(),
                                             'k_J_16_50_84_h_per_Mpc': [percentiles(col) for col in k_j.T]}
    print(json.dumps(result, indent=1))
    store('posterior', result)


# ----------------------------------------------------------------------------
# stage: tables for the growth-response figure
# ----------------------------------------------------------------------------
def command_figure():
    data = json.loads(OUT_JSON.read_text())
    zc = np.linspace(2.33, 0., 234)
    kc = np.array([.116325, .154996, .232413])              # the three groups of effective wavenumber
    solver = Solver(1600, z_out=zc, k_out=kc)
    kcol = {k: int(np.flatnonzero(np.isclose(solver.k_all, k))[0]) for k in np.r_[kc, 0.]}

    def curves(q_pts, inv_pts, cs2_pts, rate, sigma8, values, prefix):
        d, v = solver.growth(q_pts, inv_pts, cs2_pts, rate)
        columns = {'z': zc, 'reference': sigma8*v[0, :-1, kcol[0.]]/d[0, -1, kcol[0.]]}
        for n, value in enumerate(values):
            for k in kc:
                tag = f'{prefix}{value:.2e}_k{k:.3f}'.replace('.', 'p').replace('-', 'm')
                columns['ratio_' + tag] = v[n + 1, :-1, kcol[k]]/v[0, :-1, kcol[k]]
        return columns

    # (a) constant sound speed on the best-fitting expansion history of the spline model
    best = data['bestfit']['with_V_nonnegative']
    eps = np.array(best['eps_nodes'])
    values = [1e-7, data['posterior']['upper95'],
              data['profile']['with_V_nonnegative']['upper_limits_c_s2']['95% one-sided (Delta chi2 = 2.71)'], 1e-6]
    zero = spline_profile(eps[None, :], 0.)                # amplitude refitted at zero sound speed
    _, q_pts, inv_pts = solver.spline(np.repeat(eps[None, :], len(values) + 1, axis=0))
    table('response_constant', curves(q_pts, inv_pts, np.r_[0., values][:, None], 1.0, zero['sigma8'][0],
                                      values, 'cs2_'))
    shapes = FINE.shape(*[np.repeat(a, 3, axis=0) for a in FINE.spline(eps[None, :])[1:]], np.r_[0., values[1:3]][:, None])
    prediction = zero['sigma8'][0]*shapes[0]
    table('response_data', {'z': GZ, 'keff': GK, 'observed': GY, 'sigma': GSIG,
                            'prediction_zero_sound_speed': prediction,
                            'ratio': GY/prediction, 'ratio_error': GSIG/prediction,
                            'shift_in_sigma_at_upper95': zero['sigma8'][0]*(shapes[1] - shapes[0])/GSIG,
                            'shift_in_sigma_at_profile_limit': zero['sigma8'][0]*(shapes[2] - shapes[0])/GSIG})
    constant_values = [float(v) for v in values]
    # (b) constant f and V: the sound speed grows as a^3; values of c_s^2 today
    minimal = data['grids']['minimal']['growing_mode_initial_rate']['all_data']
    omega = minimal['omega_at_zero_sound_speed']
    values = [1e-7, 1e-6, minimal['profile_upper_limits']['95% one-sided (Delta chi2 = 2.71)'], 1e-5]
    c0 = np.r_[1e-30, values]
    _, q_pts, inv_pts, cs2_pts = solver.minimal(omega*np.sqrt(1 - c0), omega*np.sqrt(c0))
    table('response_minimal', curves(q_pts, inv_pts, cs2_pts, float(lcdm_growing_mode(omega)[0]),
                                     minimal['sigma8_ref_at_zero_sound_speed'], values, 'cs2today_'))
    store('figure', {'constant_sound_speed_values': constant_values,
                     'minimal_model_values_today': [float(v) for v in values], 'wavenumbers': kc.tolist(),
                     'omega_minimal': omega, 'redshifts': [float(zc[0]), float(zc[-1])]})


# ----------------------------------------------------------------------------
# stage: where the interaction-scale estimate falls below the expansion rate
# ----------------------------------------------------------------------------
def command_early():
    """Constant f and V.  The interaction-scale estimate of the appendix on the effective
    description, Lambda_E^4 = 8 pi^2 (rho+p) c_s^5/(1-c_s^2), with rho+p = rho_kin (1-c_s^2)
    and c_s = f/rho_kin, gives Lambda_E^4 = 8 pi^2 f^5/rho_kin^4, which falls as a^12
    toward the past while H rises.  Illustrative inputs as in that appendix: h = 0.7 and
    a kinetic density fraction 0.3 today; radiation with Omega_r h^2 = 4.18e-5 is added
    to the expansion rate."""
    h, omega_c, omega_r = .7, .3, 4.18e-5/.7**2
    hbar_h0 = 6.582119569e-16*h*100/3.0856775814913673e19      # eV: hbar H0, H0 = 100 h km/s/Mpc
    m_planck = 2.435323e27                                      # eV, reduced Planck mass
    critical = 3*m_planck**2*hbar_h0**2                         # eV^4

    def scale(a, c0):                 # Lambda_E in eV; c0 = c_s today
        omega_f = c0*omega_c/np.sqrt(1 - c0**2)
        kinetic = np.sqrt(omega_f**2 + omega_c**2/a**6)
        return (8*np.pi**2*critical*omega_f**5/kinetic**4)**.25

    def hubble(a):
        return hbar_h0*np.sqrt(omega_c/a**3 + omega_r/a**4 + 1 - omega_c - omega_r)
    result = {'inputs': {'h': h, 'kinetic_density_fraction_today': omega_c, 'omega_radiation': omega_r}, 'cases': []}
    for c0sq in (1e-7, 1e-6, 1e-5):
        c0 = np.sqrt(c0sq)
        a_cross = brentq(lambda t: np.log(scale(np.exp(t), c0)/hubble(np.exp(t))), np.log(1e-9), 0.)
        result['cases'].append({'c_s2_today': c0sq, 'Lambda_E_today_eV': float(scale(1., c0)),
                                'H_today_eV': float(hubble(1.)),
                                'redshift_where_Lambda_E_equals_H': float(np.exp(-a_cross) - 1),
                                'Lambda_E_over_H_at_z_1100': float(scale(1/1101., c0)/hubble(1/1101.))})
    z = [c['redshift_where_Lambda_E_equals_H'] for c in result['cases']]
    result['power_law_index_in_c_s_today'] = float(np.log(z[2]/z[0])/np.log(10.))
    print(json.dumps(result, indent=1))
    store('early', result)


if __name__ == '__main__':
    commands = {name[8:]: value for name, value in globals().items() if name.startswith('command_')}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit('usage: python3 extensions.py {' + '|'.join(commands) + '}')
    start_time = time.time()
    commands[sys.argv[1]]()
    print(f'[{sys.argv[1]}] done in {time.time() - start_time:.1f} s')
