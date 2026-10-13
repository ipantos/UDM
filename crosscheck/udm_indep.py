"""Independent re-implementation of the paper's likelihood (written from the
manuscript's equations, not imported from the repository code).  Equation numbers
are those of the manuscript; eps_H = dlnH/dlna = -(1 + q).

Background : Eqs. (23)-(26)  seven-node not-a-knot cubic spline of eps_H in x=ln(1+z)
Growth     : Eq. (30)        y_N = 3 w y + (1+w) v ;  v_N = -(1-q) v + [3/2 - cs2 k^2/((1+w) a^2 H^2)] y
Observable : Eq. (32)        sigma8 * v(k_eff, z) / D(k=0, z_ref),  z_ref = 0.51
Likelihood : Eq. (34)        BAO (13) + SN (22) + growth (17)

Usage:  python3 udm_indep.py <udm_data.txt> <n1_posterior.npz>
        (recomputes the chi^2 of the best stored sample; crosscheck.py uses the functions)
"""
import sys, csv
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

Z_NODES = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
X_NODES = np.log1p(Z_NODES)
Z_INI, Z_REF = 2.330, 0.51
C_OVER_H0 = 2997.92458            # c/H0 in Mpc/h


# ----------------------------------------------------------------- data
def read_sections(path):
    sec, name = {}, None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                name = s[1:-1]
                sec[name] = []
                continue
            sec[name].append(line)
    return sec


def load_data(path):
    sec = read_sections(path)
    aniso = np.array([[float(v) for v in ln.split(",")] for ln in sec["bao_anisotropic"][1:]])
    iso = np.array([float(v) for v in sec["bao_isotropic"][1].split(",")])
    rows = [r for r in csv.DictReader(sec["growth"]) if r["space"] == "fourier" and r["k_max"]]
    z = np.array([float(r["z_eff"]) for r in rows])
    val = np.array([float(r["fsigma8"]) for r in rows])
    sig = np.array([float(r["sigma"]) for r in rows])
    kmin = np.array([float(r["k_min"]) if r["k_min"] else 0.0 for r in rows])
    kmax = np.array([float(r["k_max"]) for r in rows])
    keff = np.sqrt(0.6 * (kmax**5 - kmin**5) / (kmax**3 - kmin**3))       # Eq. (33)
    o = np.argsort(z, kind="stable")
    z, val, sig, kmin, kmax, keff = z[o], val[o], sig[o], kmin[o], kmax[o], keff[o]
    cov = np.diag(sig**2)
    idx = {round(float(zz), 3): i for i, zz in enumerate(z)}
    wz = [idx[v] for v in (0.44, 0.6, 0.73)]
    W = 1e-3 * np.array([[float(v) for v in ln.split(",")] for ln in sec["growth_wigglez_covariance"]])
    for a in range(3):
        for b in range(3):
            cov[wz[a], wz[b]] = W[a, b]
    eb = [idx[v] for v in (0.978, 1.23, 1.526, 1.944)]
    Rm = np.array([[float(v) for v in ln.split(",")] for ln in sec["growth_eboss_correlation"]])
    for a in range(4):
        for b in range(4):
            cov[eb[a], eb[b]] = Rm[a, b] * sig[eb[a]] * sig[eb[b]]
    zsn = np.array([float(ln.split()[1]) for ln in sec["sn_union3"]])
    musn = np.array([float(ln.split()[4]) for ln in sec["sn_union3"]])
    csn = np.array([[float(v) for v in ln.split()] for ln in sec["sn_union3_covariance"]])
    return dict(aniso=aniso, iso=iso, zg=z, fs8=val, sig=sig, kmin=kmin, kmax=kmax, keff=keff,
                cinv_g=np.linalg.inv(cov), cov_g=cov, zsn=zsn, musn=musn, cinv_sn=np.linalg.inv(csn))


# ----------------------------------------------------------- background
class Background:
    """eps_H(x) spline; everything else follows analytically or by fine quadrature."""

    def __init__(self, eps_nodes, n=20001):
        self.sp = CubicSpline(X_NODES, np.asarray(eps_nodes, float), bc_type="not-a-knot", extrapolate=True)
        self.anti = self.sp.antiderivative()
        self.x = np.linspace(0.0, np.log1p(Z_INI), n)
        self.eps = self.sp(self.x)
        self.invE = np.exp(self.anti(self.x) - self.anti(0.0))            # H0/H, Eq. (25)
        f = np.exp(self.x) * self.invE                                    # dz/E = e^x dx / E
        h = self.x[1] - self.x[0]
        # cumulative Simpson (composite, even number of intervals) -> int_0^z dz'/E
        cum = np.zeros(n)
        cum[2::2] = np.cumsum(h / 3.0 * (f[0:-2:2] + 4 * f[1:-1:2] + f[2::2]))
        cum[1::2] = 0.5 * (cum[0:-1:2] + cum[2::2])                       # only even points are used below
        self._x_even, self._cum_even = self.x[::2], cum[::2]
        self.q = -1.0 - self.eps
        self.w = -1.0 - 2.0 * self.eps / 3.0

    def at(self, z, what):
        x = np.log1p(np.asarray(z, float))
        if what == "invE":
            return np.exp(self.anti(x) - self.anti(0.0))
        if what == "eps":
            return self.sp(x)
        if what == "chi":                                                 # int_0^z dz'/E
            return CubicSpline(self._x_even, self._cum_even)(x)
        raise KeyError(what)


def chi2_bao(bg, A, d):
    c2 = 0.0
    for z, dm, sm, dh, sh, r in d["aniso"]:
        res = np.array([dm - A * bg.at(z, "chi"), dh - A * bg.at(z, "invE")])
        cov = np.array([[sm**2, r * sm * sh], [r * sm * sh, sh**2]])
        c2 += res @ np.linalg.solve(cov, res)
    z, dv, sv = d["iso"]
    pred = (z * (A * bg.at(z, "chi"))**2 * A * bg.at(z, "invE"))**(1.0 / 3.0)
    return float(c2 + ((dv - pred) / sv)**2)


def chi2_sn(bg, A, M, d):
    mu = 5.0 * np.log10((1.0 + d["zsn"]) * A * bg.at(d["zsn"], "chi")) + M
    r = d["musn"] - mu
    return float(r @ d["cinv_sn"] @ r)


# ---------------------------------------------------------------- growth
def growth_shape(bg, d, cs2_of_x, n_models=1, rtol=1e-9, atol=1e-12, f_ini=1.0):
    """Return shape[m, i] = v(k_eff_i, z_i)/D(k=0, z_ref) for n_models sound-speed
    histories integrated together.  cs2_of_x(x) -> array (n_models,) at scalar x."""
    kuniq, kinv = np.unique(np.round(d["keff"], 6), return_inverse=True)
    k = np.concatenate([kuniq, [0.0]])
    nk = len(k)
    xg, epsg, invEg = bg.x, bg.eps, bg.invE

    def rhs(N, s):
        x = -N
        eps = np.interp(x, xg, epsg)
        q = -1.0 - eps
        opw = -2.0 * eps / 3.0                                            # 1 + w
        invE = np.interp(x, xg, invEg)
        cs2 = np.asarray(cs2_of_x(x), float).reshape(n_models, 1)
        Q = cs2 * (k[None, :] * C_OVER_H0 * invE * np.exp(x))**2          # cs2 k^2/(a H)^2
        y = s[: n_models * nk].reshape(n_models, nk)
        v = s[n_models * nk:].reshape(n_models, nk)
        dy = 3.0 * (opw - 1.0) * y + opw * v
        dv = -(1.0 - q) * v + (1.5 - Q / opw) * y
        return np.concatenate([dy.ravel(), dv.ravel()])

    opw_i = -2.0 * float(bg.at(Z_INI, "eps")) / 3.0
    s0 = np.concatenate([np.full(n_models * nk, opw_i), np.full(n_models * nk, f_ini)])
    t_eval = np.unique(np.concatenate([-np.log1p(d["zg"]), [-np.log1p(Z_REF)], [0.0]]))
    sol = solve_ivp(rhs, [-np.log1p(Z_INI), 0.0], s0, method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y[: n_models * nk].reshape(n_models, nk, -1)
    V = sol.y[n_models * nk:].reshape(n_models, nk, -1)
    iref = int(np.argmin(np.abs(sol.t + np.log1p(Z_REF))))
    opw_ref = -2.0 * float(bg.at(Z_REF, "eps")) / 3.0
    Dref = Y[:, nk - 1, iref] / opw_ref                                   # D(k=0, z_ref)
    shape = np.empty((n_models, len(d["zg"])))
    for i, z in enumerate(d["zg"]):
        it = int(np.argmin(np.abs(sol.t + np.log1p(z))))
        shape[:, i] = V[:, kinv[i], it] / Dref
    return shape


def chi2_growth(shape, sigma8, d):
    r = d["fs8"] - sigma8 * shape
    return float(r @ d["cinv_g"] @ r)


def best_sigma8(shape, d):
    return float((shape @ d["cinv_g"] @ d["fs8"]) / (shape @ d["cinv_g"] @ shape))


# ------------------------------------------------- sound-speed histories
def cs2_constant(values):
    values = np.asarray(values, float)
    return lambda x: values


def cs2_minimal(bg, s0_values):
    """Constant prefactor f = alpha:  c_s/(1-c_s^2) = alpha/(rho+p),  rho+p = -2 H^2 eps_H.
    Parametrised by s0 = c_s^2(z=0)."""
    c0 = np.sqrt(np.asarray(s0_values, float))
    r0 = c0 / (1.0 - c0**2)
    R0 = -2.0 * float(bg.at(0.0, "eps"))                                  # (rho+p)/H0^2 at z=0

    def f(x):
        R = -2.0 * np.interp(x, bg.x, bg.eps) / np.interp(x, bg.x, bg.invE)**2
        r = r0 * R0 / R
        c = np.where(r > 1e-8, (np.sqrt(1.0 + 4.0 * r**2) - 1.0) / (2.0 * np.maximum(r, 1e-300)), r)
        return c**2
    return f


def cs2_a6(s0_values):
    """Exact constant-f, constant-V law: s/(1-s) = s0/(1-s0) a^6."""
    s0 = np.asarray(s0_values, float)

    def f(x):
        R = s0 / (1.0 - s0) * np.exp(-6.0 * x)
        return R / (1.0 + R)
    return f


if __name__ == "__main__":
    data_path, post_path = sys.argv[1], sys.argv[2]
    d = load_data(data_path)
    post = np.load(post_path, allow_pickle=False)
    best = post["best_sampled_parameters"]
    eps, A, cs2, s8, M = best[:7], best[7], best[8], best[9], best[10]
    bg = Background(eps)
    sh = growth_shape(bg, d, cs2_constant([cs2]))[0]
    cb, csn, cg = chi2_bao(bg, A, d), chi2_sn(bg, A, M, d), chi2_growth(sh, s8, d)
    print("k_eff          :", np.round(d["keff"], 6))
    print("best sample    : chi2_BAO=%.4f  chi2_SN=%.4f  chi2_growth=%.4f  total=%.4f   (stored: %.4f)"
          % (cb, csn, cg, cb + csn + cg, float(post["best_sampled_chi2"])))
    # response of each growth point to raising cs2 from 0 to the paper's limit, sigma8 fixed
    lim = float(np.percentile(post["chain"][:, 8], 95))
    s2 = growth_shape(bg, d, cs2_constant([0.0, lim]), n_models=2)
    resp = s8 * (s2[0] - s2[1]) / d["sig"]
    print("response/sigma :", np.round(resp, 4))
    print("fraction       :", np.round((s2[0] - s2[1]) / s2[0], 5))
