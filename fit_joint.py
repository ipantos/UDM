#!/usr/bin/env python3
"""
JOINT FIT -- background, sound speed and sigma_8 in ONE likelihood.

Run in this order:

    python3 fit_joint.py grid
    python3 fit_joint.py start
    python3 fit_joint.py nodes
    python3 fit_joint.py mcmc 40000
    python3 fit_joint.py summary
    python3 fit_joint.py validate
    python3 fit_joint.py posterior   (headline numbers from the saved posterior)

The MCMC is resumable: `mcmc 10000` adds 10000 steps to the existing chain.

DATA  (all read from udm_data.txt, kept next to this script)
----
BAO      13  DESI DR2
growth   17  Fourier-space f sigma_8 measurements
SNe      22  Union3 binned distance moduli
total    52

BACKGROUND
----------
eps_H = dln(H)/dln(a), seven nodes and a cubic spline in x = ln(1+z) = -N.

    H0/H    = exp[int eps_H dx],   D_H/r_d = A H0/H,
    D_M/r_d = A int (H0/H) dz,     A       = c/(H0 r_d).

GROWTH
------
Exact linear equations of the single scalar (Appendix C), valid for any
background pressure history:
    y_N = 3 w y + (1+w) V,
    V_N = -(1-q) V + [3/2 - c_s^2 k^2/((1+w) a^2 H^2)] y,
    y = delta rho_com/rho = (1+w) D,   D = delta rho_com/(rho_phi + p_phi),
    V = -theta/(aH) = D_N + beta D,    beta = p_N/(rho+p).
Eliminating V gives Eq. (C8) for D.  The compressed growth measurements are
compared with the velocity:  [f sigma_8] = sigma_8 V(k,z)/D(k=0, z=Z_NORM),
with Z_NORM = 0.51, a redshift where the background is constrained by data.
Start at z = 2.33 with D = 1 and V/D = F_INI = 1.
`validate` compares this system at k=0 with the independent u-equation
(Eq. C2), which uses w_NN; the two must agree to integration accuracy.

PHYSICAL CONSTRAINTS, at every grid point
-----------------------------------------
    V >= 0  ->  c_s^2 <= -w   ->  eps_H - (3/2) c_s^2 >= -3/2
    f > 0   ->  w    > -1     ->  eps_H < 0
"""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


# ============================================================
# RUN IDENTITY, PARAMETER CONVERSION AND SAVED RESULTS
# ============================================================
ROOT = Path(__file__).resolve().parent
SCHEMA = 1


def output_root():
    return Path(os.environ.get('UDM_OUTPUT_DIR', ROOT)).expanduser().resolve()


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(config):
    return hashlib.sha256(canonical_json(config).encode('utf-8')).hexdigest()


def runtime_versions():
    versions = {'python': platform.python_version(), 'platform': platform.platform(),
                'machine': platform.machine(), 'implementation': platform.python_implementation()}
    for name in ('numpy', 'scipy', 'matplotlib', 'emcee'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_hashes(*names):
    return {name: digest_file(ROOT / name) for name in names}


def convert_parameters(parameters, nn, nodes, prior, to):
    """Convert on the last axis without silently clipping or mutating input.

    Bounds belong to the prior, not the coordinate transformation. Positive
    values outside the prior remain outside it after conversion. Zero and
    negative physical sound speeds cannot be represented in log coordinates.
    """
    array = np.array(parameters, dtype=float, copy=True)
    if prior != 'log' or to not in ('physical', 'sampling'):
        raise ValueError('Unknown prior or coordinate representation')
    if nodes < 1 or array.ndim < 1 or array.shape[-1] != nn + nodes + 3:
        raise ValueError('Parameter dimension does not match the node count')
    if not np.isfinite(array).all():
        raise ValueError('Parameters must be finite')
    slot = (..., slice(nn + 1, nn + 1 + nodes))
    if to == 'sampling':
        if np.any(array[slot] <= 0):
            raise ValueError('Log-prior conversion requires strictly positive c_s^2')
        array[slot] = np.log10(array[slot])
    else:
        with np.errstate(over='raise', under='ignore'):
            array[slot] = np.power(10.0, array[slot])
    return array


def posterior_indices(length, maximum, seed):
    """Uniform sampling without replacement; never stride a flattened ensemble."""
    if length < 1 or maximum < 1:
        raise ValueError('A nonempty chain and positive sample count are required')
    if maximum >= length:
        return np.arange(length)
    return np.random.default_rng(seed).choice(length, size=maximum, replace=False)


def posterior_subset(chain, maximum, seed):
    return np.asarray(chain)[posterior_indices(len(chain), maximum, seed)]


def _array_digest(array):
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise ValueError('Object arrays are not supported')
    h = hashlib.sha256()
    h.update(canonical_json({'dtype': array.dtype.str, 'shape': list(array.shape)}).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def save_bundle(path, arrays, config, kind, coordinates, nodes, extra=None):
    """Atomically save arrays and their metadata in one non-pickle archive.

    `extra` is recorded in the metadata but takes no part in the identity
    check performed by load_bundle.
    """
    path = Path(path)
    values = {name: np.asarray(value) for name, value in arrays.items()}
    if '_metadata' in values:
        raise ValueError('Reserved array name')
    metadata = {
        'schema': SCHEMA, 'configuration': config,
        'configuration_sha256': fingerprint(config), 'kind': kind,
        'coordinates': coordinates, 'cs_nodes': int(nodes),
        'array_sha256': {key: _array_digest(value) for key, value in values.items()},
        'provenance': extra or {},
    }
    values['_metadata'] = np.array(canonical_json(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_bundle(path, config, kind, coordinates, nodes):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['_metadata'].item()))
        arrays = {key: archive[key].copy() for key in archive.files if key != '_metadata'}
    expected = {'schema': SCHEMA, 'kind': kind, 'coordinates': coordinates,
                'cs_nodes': int(nodes), 'configuration_sha256': fingerprint(config)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Incompatible checkpoint {path}: {key} differs')
    if fingerprint(metadata['configuration']) != metadata['configuration_sha256']:
        raise ValueError(f'Corrupt configuration metadata: {path}')
    actual = {key: _array_digest(value) for key, value in arrays.items()}
    if actual != metadata['array_sha256']:
        raise ValueError(f'Checkpoint array checksum mismatch: {path}')
    return arrays


def pack_random_state(state):
    """Serialize emcee/NumPy MT19937 state without pickle."""
    algorithm, keys, position, has_gauss, cached_gaussian = state
    if algorithm != 'MT19937':
        raise ValueError(f'Unsupported random-state algorithm: {algorithm}')
    return {'rng_keys': np.asarray(keys, dtype=np.uint32),
            'rng_position': np.array(position, dtype=np.int64),
            'rng_has_gauss': np.array(has_gauss, dtype=np.int64),
            'rng_cached_gaussian': np.array(cached_gaussian, dtype=float)}


def unpack_random_state(arrays):
    state = ('MT19937', arrays['rng_keys'].astype(np.uint32),
             int(arrays['rng_position']), int(arrays['rng_has_gauss']),
             float(arrays['rng_cached_gaussian']))
    probe = np.random.RandomState()
    probe.set_state(state)  # Validate all components before handing them to emcee.
    return state


def retained_segment(chain, done, thin):
    """Keep raw steps 1, 1+thin, ... independently of continuation boundaries."""
    if done < 0 or thin < 1:
        raise ValueError('Invalid step count or thinning interval')
    return np.asarray(chain)[(-done) % thin::thin]


def post_burn_chain(thin_chain, burn, thin):
    # Stored row j belongs to raw step 1+j*thin. Discard steps <= burn.
    return np.asarray(thin_chain)[(burn + thin - 1) // thin:]


def validate_sampler_arrays(arrays, walkers, dimension, thin):
    done = int(arrays['done'].item())
    expected_rows = (done + thin - 1) // thin
    expected = {'coords': (walkers, dimension), 'log_prob': (walkers,),
                'thin_chain': (expected_rows, walkers, dimension)}
    if done < 1:
        raise ValueError('Invalid completed step count')
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid sampler array: {key}')
    unpack_random_state(arrays)
    return done


@contextmanager
def exclusive_run(directory):
    """Refuse concurrent writers; a stale lock is explicit after a crash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.writer.lock'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f'Run is locked: {path}. Check for a running process before removing a stale lock.') from error
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


# ============================================================
# OBSERVATIONAL DATA FILE AND POSTERIOR
# ============================================================
# All measurements are read from the plain-text file udm_data.txt, kept next
# to this script.  Its SHA-256 digest is part of the run identity.
DATA_FILE = ROOT / 'udm_data.txt'


def read_data_sections(path=DATA_FILE):
    """Return {section name: list of non-comment lines} from udm_data.txt."""
    sections, current = {}, None
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            line = line.rstrip('\n')
            if line.startswith('#') or not line.strip():
                continue
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                sections[current] = []
            elif current is None:
                raise ValueError(f'Data line outside a section in {path}: {line}')
            else:
                sections[current].append(line)
    return sections


def data_hashes():
    return {DATA_FILE.name: digest_file(DATA_FILE)}


DATA = read_data_sections()


def _numbers(section, separator=','):
    return np.array([[float(value) for value in line.split(separator)]
                     for line in DATA[section]])


def posterior_chain():
    """Physical posterior saved by `summary` for the selected node number."""
    number_of_nodes = selected_node_number()
    try:
        return load_result(f'n{number_of_nodes}_posterior', 'posterior', 'physical',
                           number_of_nodes)['chain']
    except FileNotFoundError as error:
        raise FileNotFoundError('No posterior found: run start, nodes, mcmc 40000 and '
                                'summary first.') from error


# ============================================================
# CHOICES
# ============================================================
BC_TYPE = "not-a-knot"
NGRID, CONV_TARGET = 3201, 1.0e-3
N_RESTARTS, NODE_RESTARTS = 12, 6
# The constrained minimiser of the start stage satisfies its constraints
# only to rounding accuracy, so it is given the physical domain shrunk by
# this margin; its results then always pass the strict tests of log_prob.
START_MARGIN = 1.0e-6
# Initial guess of the start stage for eps_H (all nodes), A and c_s^2.
# sigma_8 and the supernova offset are linear in the model and are set
# analytically at the seed background, so they need no initial value.
START_EPS, START_A, START_CS2 = -1.0, 30.0, 4.0e-7
Z_INI, F_INI = 2.330, 1.0
K_EFF_RULE = "RMS of the published window with mode-count weighting k^2 dk"
PRIOR_EPS = (-10.0, 2.0)
PRIOR_A = (1.0, 500.0)
# --- c_s^2 prior ---------------------------------------------------------
# The sampling parameter for each c_s^2 node is u = log10(c_s^2), uniform on
# PRIOR_U: equal probability per decade.  CS2_PRIOR names this choice in the
# run identity and in the checkpoint metadata.  PRIOR_CS2 is the admissible
# range of c_s^2 itself, checked on every node and on the whole grid.
CS2_PRIOR = "log"
PRIOR_U = (-12.0, -2.0)
PRIOR_CS2 = (0.0, 10.0**PRIOR_U[1])


def cs_slot(number_of_cs_nodes):
    return slice(NN + 1, NN + 1 + number_of_cs_nodes)


def to_physical(parameters, number_of_cs_nodes):
    """Sampling -> physical parameters; supports vectors or arrays of vectors."""
    return convert_parameters(parameters, NN, number_of_cs_nodes,
                                    CS2_PRIOR, 'physical')


def to_sampling(parameters, number_of_cs_nodes):
    """Physical -> sampling parameters; never clip values to the prior."""
    return convert_parameters(parameters, NN, number_of_cs_nodes,
                                    CS2_PRIOR, 'sampling')
PRIOR_S8 = (0.1, 2.0)
PRIOR_CALM = (-10.0, 60.0)
NWALKERS, NSTEPS, NBURN, THIN = 40, 40000, 20000, 20
SEED = 20260809
RTOL, ATOL = 1.0e-6, 1.0e-9
VALIDATE_RTOL, VALIDATE_ATOL = 1.0e-9, 1.0e-12
VALIDATE_SAMPLES, VALIDATE_W_FLOOR = 300, 1.0e-8

# Tags the checkpoint files, so that chains produced under different
# likelihoods, data, priors, k_eff prescriptions or covariances never mix.
RUN_TAG = "joint_norm051"

# Hubble distance c/H0 in h^-1 Mpc, i.e. c/(100 km/s).  The tabulated
# wavenumbers are in h Mpc^-1, so k c/H0 is independent of h.
# It enters only through the pressure term.
C_OVER_H0 = 2997.92458

# Growth equations: exact single-scalar system (continuity + Euler), compared
# with the data through the velocity.  Part of the run identity.
GROWTH_SYSTEM = "exact y-V system (Appendix C), velocity observable, V/D=1 at z_ini"
# Redshift at which sigma_8 normalises D (k=0 mode).  It must be a growth-data
# redshift so that it is an output time of the integration.
Z_NORM = 0.51

# ============================================================
# DATA
# ============================================================
Z_NODES = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
X_NODES = np.log(1.0 + Z_NODES)
NN = len(Z_NODES)

# z, D_M/r_d, sigma_M, D_H/r_d, sigma_H, rho_MH   (udm_data.txt, [bao_anisotropic])
ANISO = [tuple(float(v) for v in line.split(',')) for line in DATA['bao_anisotropic'][1:]]
# z, D_V/r_d, sigma_V   (udm_data.txt, [bao_isotropic])
BGS = tuple(float(v) for v in DATA['bao_isotropic'][1].split(','))

CINVB = []
for _, _, sigma_m, _, sigma_h, rho_mh in ANISO:
    covariance = np.array([[sigma_m**2, rho_mh*sigma_m*sigma_h],
                           [rho_mh*sigma_m*sigma_h, sigma_h**2]])
    CINVB.append(np.linalg.inv(covariance))


def k_effective(k_min, k_max):
    """RMS wavenumber of a top-hat k-window with mode-count weighting k^2 dk:
    k_eff^2 = [int k^4 dk]/[int k^2 dk] = (3/5)(kx^5-kn^5)/(kx^3-kn^3)."""
    k_min = np.asarray(k_min, dtype=float)
    k_max = np.asarray(k_max, dtype=float)
    if np.any(k_max <= k_min):
        raise ValueError("Every growth window must satisfy k_max > k_min")
    return np.sqrt((3.0/5.0)*(k_max**5 - k_min**5)/(k_max**3 - k_min**3))


def load_growth():
    rows = [row for row in csv.DictReader(DATA['growth'])
            if row["space"] == "fourier" and row["k_max"]]
    z = np.array([float(row["z_eff"]) for row in rows])
    value = np.array([float(row["fsigma8"]) for row in rows])
    sigma = np.array([float(row["sigma"]) for row in rows])
    k_min = np.array([float(row["k_min"]) if row["k_min"] else 0.0 for row in rows])
    k_max = np.array([float(row["k_max"]) for row in rows])
    k_eff = k_effective(k_min, k_max)
    order = np.argsort(z)
    # k_max (largest wavenumber of each window) is returned as well; it is
    # not used by the likelihood, which uses k_eff.
    return z[order], value[order], sigma[order], k_eff[order], k_max[order]


ZG_D, FS8, SIG8, K_EFF, K_MAX = load_growth()
NG = len(ZG_D)

# Published covariance of the three WiggleZ measurements (udm_data.txt).
WIGGLEZ_Z = (0.44, 0.60, 0.73)
WIGGLEZ_COV = 1.0e-3*_numbers('growth_wigglez_covariance')
# Published correlations of the four eBOSS DR14Q measurements (udm_data.txt).
EBOSS_Z = (0.978, 1.230, 1.526, 1.944)
EBOSS_CORR = _numbers('growth_eboss_correlation')


def build_cov_growth():
    covariance = np.diag(SIG8**2)
    index = {round(float(z), 3): i for i, z in enumerate(ZG_D)}
    wigglez_index = [index[round(z, 3)] for z in WIGGLEZ_Z]
    for i in range(3):
        for j in range(3):
            covariance[wigglez_index[i], wigglez_index[j]] = WIGGLEZ_COV[i, j]
    eboss_index = [index[round(z, 3)] for z in EBOSS_Z]
    for i in range(4):
        for j in range(4):
            covariance[eboss_index[i], eboss_index[j]] = (
                EBOSS_CORR[i, j]*SIG8[eboss_index[i]]*SIG8[eboss_index[j]])
    return covariance


COV_G = build_cov_growth()
assert np.allclose(COV_G, COV_G.T), "Growth covariance is not symmetric"
assert np.all(np.linalg.eigvalsh(COV_G) > 0.0), "Growth covariance is not positive definite"
CINV_G = np.linalg.inv(COV_G)


def load_sne():
    z, mu = [], []
    for line in DATA['sn_union3']:
        parts = line.split()
        z.append(float(parts[1]))      # zcmb
        mu.append(float(parts[4]))     # mb (binned distance modulus)
    z, mu = np.asarray(z), np.asarray(mu)
    covariance = _numbers('sn_union3_covariance', separator=None)
    n = covariance.shape[0]
    assert covariance.shape == (n, n), "SNe covariance is not square"
    assert n == len(z), ("SNe covariance/data size mismatch", n, len(z))
    assert np.allclose(covariance, covariance.T), "SNe covariance is not symmetric"
    return z, mu, np.linalg.inv(covariance)


Z_SN, MU_SN, CINV_SN = load_sne()
NSN = len(Z_SN)
NDATA = 13 + NG + NSN

# ============================================================
# BACKGROUND
# ============================================================
XG = np.linspace(0.0, np.log(1.0 + Z_NODES[-1]), NGRID)
ZGRID = np.exp(XG) - 1.0
DZ = np.diff(ZGRID)

assert ZGRID[0] <= Z_SN.min() and Z_SN.max() <= ZGRID[-1], "SNe outside the grid"
assert ZGRID[0] <= ZG_D.min() and ZG_D.max() <= ZGRID[-1], "Growth data outside the grid"
assert all(ZGRID[0] <= z <= ZGRID[-1] for z, *_ in ANISO), "BAO data outside the grid"


def _basis(x_grid, derivative=0):
    columns = []
    for index in range(NN):
        unit = np.zeros(NN)
        unit[index] = 1.0
        spline = CubicSpline(X_NODES, unit, bc_type=BC_TYPE, extrapolate=True)
        columns.append(spline(x_grid) if derivative == 0
                       else spline.derivative(derivative)(x_grid))
    return np.asarray(columns).T


SMAT = _basis(XG)
SMAT_D1 = _basis(XG, derivative=1)
SMAT_D2 = _basis(XG, derivative=2)
IMAT = np.array([np.concatenate([[0.0],
        np.cumsum(0.5*(SMAT[1:, i] + SMAT[:-1, i])*np.diff(XG))])
        for i in range(NN)]).T


def background(eps_nodes, amplitude):
    """Return H0/H, D_M/r_d, D_H/r_d and q on the reconstruction grid."""
    exponent = IMAT @ eps_nodes
    if np.any(exponent > 700.0):
        raise FloatingPointError("Background exponential overflow")
    inv_hubble = np.exp(exponent)
    distance_h = amplitude*inv_hubble
    distance_m = amplitude*np.concatenate([[0.0],
        np.cumsum(0.5*(inv_hubble[1:] + inv_hubble[:-1])*DZ)])
    q_grid = -1.0 - (SMAT @ eps_nodes)
    return inv_hubble, distance_m, distance_h, q_grid


def cs2_on_grid(cs2_nodes, z_nodes):
    """c_s^2 on the full grid, without clipping.  Negative values or values
    outside the prior caused by spline overshoot are rejected by log_prob."""
    cs2_nodes = np.asarray(cs2_nodes)
    z_nodes = np.asarray(z_nodes)
    if len(z_nodes) == 1:
        return np.full_like(XG, float(cs2_nodes[0]))
    spline = CubicSpline(np.log(1.0 + z_nodes), cs2_nodes,
                         bc_type="natural", extrapolate=True)
    return spline(XG)


def node_positions(number_of_nodes):
    if number_of_nodes == 1:
        return np.array([np.sqrt(ZG_D[0]*ZG_D[-1])])
    return np.expm1(np.linspace(np.log(1.0 + ZG_D[0]),
                                np.log(1.0 + ZG_D[-1]), number_of_nodes))


def selected_node_number():
    if not result_path('node_choice').exists():
        return 1
    number = int(load_result('node_choice', 'node_choice', 'none', 0)['chosen'].item())
    if number not in (1, 2, 3):
        raise ValueError('Invalid saved node choice')
    return number


def configuration():
    """Identity of this fit.

    Only quantities that change the result enter: settings, data, growth
    equations and the normalisation redshift.  Library versions and the
    bytes of this script are recorded separately by provenance(); they are
    saved with every result but do not change its identity, so the results
    stay readable on another computer and after editing a comment.
    """
    settings = {name: globals()[name] for name in (
        'BC_TYPE', 'NGRID', 'CONV_TARGET', 'N_RESTARTS', 'NODE_RESTARTS',
        'Z_INI', 'F_INI', 'K_EFF_RULE', 'PRIOR_EPS', 'PRIOR_A', 'CS2_PRIOR',
        'PRIOR_U', 'PRIOR_CS2', 'PRIOR_S8', 'PRIOR_CALM', 'NWALKERS',
        'NSTEPS', 'NBURN', 'THIN', 'SEED', 'RTOL', 'ATOL', 'C_OVER_H0',
        'GROWTH_SYSTEM', 'Z_NORM', 'START_MARGIN', 'START_EPS', 'START_A',
        'START_CS2')}
    return {'run_tag': RUN_TAG, 'settings': settings,
            'background_nodes': Z_NODES.tolist(),
            'bao_anisotropic': ANISO, 'bao_isotropic': BGS,
            'growth_covariance': COV_G.tolist(),
            'growth_redshifts': ZG_D.tolist(), 'growth_k_eff': K_EFF.tolist(),
            'data_sha256': data_hashes(),
            'chain_layout': 'step, walker, sampling-parameter',
            'thinning': 'raw steps 1+j*THIN; discard raw steps <= NBURN'}


def provenance():
    """Where and with what a result was produced; informative only."""
    return {'source_sha256': source_hashes(Path(__file__).name),
            'runtime': runtime_versions()}


def run_directory():
    return output_root() / 'runs' / f'{RUN_TAG}_{CS2_PRIOR}_{fingerprint(configuration())[:16]}'


def result_path(name):
    return run_directory() / f'{name}.npz'


def save_result(name, arrays, kind, coordinates, nodes):
    save_bundle(result_path(name), arrays, configuration(), kind, coordinates, nodes,
                extra=provenance())


def load_result(name, kind, coordinates, nodes):
    return load_bundle(result_path(name), configuration(), kind, coordinates, nodes)


def chain_tag(number_of_nodes):
    return str(run_directory() / f'n{number_of_nodes}')


# ============================================================
# CHI-SQUARED
# ============================================================
def chi2_bao(distance_m, distance_h):
    chi2 = 0.0
    for index, (z, obs_m, sig_m, obs_h, sig_h, corr) in enumerate(ANISO):
        residual = np.array([obs_m - np.interp(z, ZGRID, distance_m),
                             obs_h - np.interp(z, ZGRID, distance_h)])
        chi2 += residual @ CINVB[index] @ residual
    z, obs_v, sig_v = BGS
    argument = z*np.interp(z, ZGRID, distance_m)**2*np.interp(z, ZGRID, distance_h)
    if argument <= 0.0:
        return np.inf
    chi2 += ((obs_v - np.cbrt(argument))/sig_v)**2
    return float(chi2)


def chi2_sne(distance_m, cal_m):
    interpolated = np.interp(Z_SN, ZGRID, distance_m)
    if np.any(interpolated <= 0.0):
        return np.inf
    predicted_mu = 5.0*np.log10((1.0 + Z_SN)*interpolated) + cal_m
    residual = MU_SN - predicted_mu
    return float(residual @ CINV_SN @ residual)


K_UNIQ, K_IDX = np.unique(np.round(K_EFF, 6), return_inverse=True)
# solve_ivp requires strictly increasing, unique t_eval values.
T_EVAL = np.unique(np.concatenate([-np.log(1.0 + ZG_D), [0.0]]))


def growth(cs2_grid, q_grid, inv_hubble, amplitude):
    """Integrate every distinct k_eff simultaneously, plus a k=0 reference.

    Exact linear system of Appendix C, with no constant-pressure assumption:
        y = delta rho_com / rho = (1+w) D,   V = -theta/(aH) = D_N + beta D,
        y_N = 3 w y + (1+w) V                         (energy conservation)
        V_N = -(1-q) V + [3/2 - Q/(1+w)] y            (momentum conservation)
    with Q = c_s^2 k^2/(a^2 H^2).  Eliminating V gives Eq. (C8) for D.
    Only w and q enter; no derivative of the background spline is used.
    Initial conditions at Z_INI: D = 1 (so y = 1+w) and V/D = F_INI.
    """
    k_values = np.concatenate([K_UNIQ, [0.0]])
    number_of_modes = len(k_values)

    def rhs(N, state):
        x = -N
        q = np.interp(x, XG, q_grid)
        one_plus_w = 2.0*(1.0 + q)/3.0          # q = (1+3w)/2
        cs2 = np.interp(x, XG, cs2_grid)
        h0_over_h = np.interp(x, XG, inv_hubble)
        z = np.expm1(x)
        pressure = cs2*(k_values*C_OVER_H0*h0_over_h)**2*(1.0 + z)**2
        y = state[:number_of_modes]
        V = state[number_of_modes:]
        return np.concatenate([3.0*(one_plus_w - 1.0)*y + one_plus_w*V,
                               -(1.0 - q)*V + (1.5 - pressure/one_plus_w)*y])

    one_plus_w_initial = 2.0*(1.0 + float(np.interp(np.log(1.0 + Z_INI), XG, q_grid)))/3.0
    initial_state = np.concatenate([np.full(number_of_modes, one_plus_w_initial),
                                    np.full(number_of_modes, F_INI)])
    solution = solve_ivp(rhs, [-np.log(1.0 + Z_INI), 0.0], initial_state,
                         rtol=RTOL, atol=ATOL, t_eval=T_EVAL)
    return solution, number_of_modes


def growth_shape(cs2_grid, q_grid, inv_hubble, amplitude):
    """V(k_i, z_i)/D(k=0, Z_NORM) for the 17 measurements, i.e. the growth
    prediction for sigma_8 = 1; None if the integration fails."""
    solution, number_of_modes = growth(cs2_grid, q_grid, inv_hubble, amplitude)
    if not solution.success:
        return None
    if len(np.atleast_1d(solution.t)) != len(T_EVAL):
        return None
    times = np.asarray(solution.t)
    values = np.asarray(solution.y)
    if not np.all(np.isfinite(values)):
        return None
    norm_N = -np.log(1.0 + Z_NORM)
    norm_index = int(np.argmin(np.abs(times - norm_N)))
    if abs(times[norm_index] - norm_N) > 1.0e-12:
        return None
    # The final mode is the k=0 reference; D = y/(1+w) at z = Z_NORM.
    q_norm = float(np.interp(np.log(1.0 + Z_NORM), XG, q_grid))
    D_reference = values[number_of_modes - 1, norm_index]/(2.0*(1.0 + q_norm)/3.0)
    if not np.isfinite(D_reference) or D_reference == 0.0:
        return None
    shape = np.empty(NG)
    for index in range(NG):
        target_N = -np.log(1.0 + ZG_D[index])
        time_index = int(np.argmin(np.abs(times - target_N)))
        # RSD measure the velocity: V = D_N + beta D (Appendix C).
        velocity = values[number_of_modes + K_IDX[index], time_index]
        shape[index] = velocity/D_reference
    return shape


def chi2_growth(cs2_grid, q_grid, inv_hubble, amplitude, sigma8):
    shape = growth_shape(cs2_grid, q_grid, inv_hubble, amplitude)
    if shape is None:
        return np.inf
    residual = FS8 - sigma8*shape
    return float(residual @ CINV_G @ residual)


# ============================================================
# JOINT LIKELIHOOD
# ============================================================
def unpack(parameters, number_of_cs_nodes):
    eps = parameters[:NN]
    amplitude = parameters[NN]
    cs_start = NN + 1
    cs_stop = cs_start + number_of_cs_nodes
    return (eps, amplitude, parameters[cs_start:cs_stop],
            parameters[cs_stop], parameters[cs_stop + 1])


def chi2_total(parameters, z_nodes):
    eps, amplitude, cs2_nodes, sigma8, cal_m = unpack(parameters, len(z_nodes))
    try:
        inv_hubble, distance_m, distance_h, q_grid = background(eps, amplitude)
    except Exception:
        return np.inf
    if not (np.all(np.isfinite(inv_hubble)) and np.all(np.isfinite(distance_m))
            and np.all(np.isfinite(distance_h)) and np.all(np.isfinite(q_grid))):
        return np.inf
    chi2 = chi2_bao(distance_m, distance_h)
    if not np.isfinite(chi2):
        return np.inf
    sne_contribution = chi2_sne(distance_m, cal_m)
    if not np.isfinite(sne_contribution):
        return np.inf
    chi2 += sne_contribution
    growth_contribution = chi2_growth(cs2_on_grid(cs2_nodes, z_nodes), q_grid,
                                      inv_hubble, amplitude, sigma8)
    if not np.isfinite(growth_contribution):
        return np.inf
    return float(chi2 + growth_contribution)


def log_prob_physical(parameters, z_nodes):
    """Prior and likelihood evaluated at a physical vector (no Jacobian change)."""
    try:
        sampling = to_sampling(parameters, len(z_nodes))
    except (ValueError, FloatingPointError):
        return -np.inf
    return log_prob(sampling, z_nodes)


def log_prob(parameters, z_nodes):
    """Log probability in the declared sampling coordinates only."""
    number_of_cs_nodes = len(z_nodes)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (NN + 3 + number_of_cs_nodes,) or not np.isfinite(parameters).all():
        return -np.inf
    u_nodes = np.asarray(parameters)[cs_slot(number_of_cs_nodes)]
    if np.any(u_nodes < PRIOR_U[0]) or np.any(u_nodes > PRIOR_U[1]):
        return -np.inf
    parameters = to_physical(parameters, number_of_cs_nodes)
    eps, amplitude, cs2_nodes, sigma8, cal_m = unpack(parameters, len(z_nodes))
    if np.any(eps <= PRIOR_EPS[0]) or np.any(eps >= PRIOR_EPS[1]):
        return -np.inf
    if not (PRIOR_A[0] < amplitude < PRIOR_A[1]):
        return -np.inf
    if np.any(cs2_nodes < PRIOR_CS2[0]) or np.any(cs2_nodes > PRIOR_CS2[1]):
        return -np.inf
    if not (PRIOR_S8[0] < sigma8 < PRIOR_S8[1]):
        return -np.inf
    if not (PRIOR_CALM[0] < cal_m < PRIOR_CALM[1]):
        return -np.inf
    cs2_grid = cs2_on_grid(cs2_nodes, z_nodes)
    if not np.all(np.isfinite(cs2_grid)):
        return -np.inf
    if np.any(cs2_grid < PRIOR_CS2[0]) or np.any(cs2_grid > PRIOR_CS2[1]):
        return -np.inf
    w_grid = -1.0 - (2.0/3.0)*(SMAT @ eps)
    if np.any(w_grid > -cs2_grid):        # V >= 0:  c_s^2 <= -w
        return -np.inf
    if np.any(w_grid <= -1.0):            # regular branch (R > 0):  w > -1
        return -np.inf
    try:
        chi2 = chi2_total(parameters, z_nodes)
    except Exception:
        return -np.inf
    return -np.inf if not np.isfinite(chi2) else -0.5*chi2


# ============================================================
# CLOSURE DIAGNOSTIC
# ============================================================
def validate_zero_mode(eps, amplitude):
    """Compare at k=0 the production growth system with the u-equation (C2).
    The production system (y, V) uses only w and q; the u-equation contains
    theta''/theta and therefore w_NN.  Both are exact, so their growth rates
    D_N/D must agree to integration accuracy: an independent check of the
    implementation, not an extra likelihood term.  Samples too close to
    w=-1 are skipped because u is singular when rho+p vanishes."""
    _, _, _, q_grid = background(eps, amplitude)
    w = -1.0 - (2.0/3.0)*(SMAT @ eps)
    one_plus_w = 1.0 + w
    if np.min(one_plus_w) < VALIDATE_W_FLOOR:
        return None
    # Since x = -N:  w_N = +(2/3) eps_x,  w_NN = -(2/3) eps_xx.
    w_N = (2.0/3.0)*(SMAT_D1 @ eps)
    w_NN = -(2.0/3.0)*(SMAT_D2 @ eps)
    h = -1.5*one_plus_w
    L1 = -1.0 - w_N/(2.0*one_plus_w)                       # dln(theta)/dN
    L1_N = -0.5*(w_NN*one_plus_w - w_N**2)/one_plus_w**2
    theta_term = L1**2 + L1_N + L1*(1.0 + h)
    g_grid = 2.0 + h + w_N/(2.0*one_plus_w)                # u = G D, G ~ a^2 H sqrt(1+w)
    N_initial = -np.log(1.0 + Z_INI)
    g_initial = float(np.interp(-N_initial, XG, g_grid))

    beta_grid = w_N/one_plus_w - 3.0*w                     # Eq. (C5)

    def rhs_D(N, state):
        # Production system at k=0 (same equations as growth()).
        q = np.interp(-N, XG, q_grid)
        opw = 2.0*(1.0 + q)/3.0
        y, V = state
        return [3.0*(opw - 1.0)*y + opw*V, -(1.0 - q)*V + 1.5*y]

    def rhs_u(N, state):
        h_value = np.interp(-N, XG, h)
        theta_value = np.interp(-N, XG, theta_term)
        u, u_N = state
        return [u_N, -(1.0 + h_value)*u_N + theta_value*u]

    opw_initial = float(np.interp(-N_initial, XG, one_plus_w))
    beta_initial = float(np.interp(-N_initial, XG, beta_grid))
    solution_D = solve_ivp(rhs_D, [N_initial, 0.0], [opw_initial, F_INI],
                           rtol=VALIDATE_RTOL, atol=VALIDATE_ATOL, dense_output=True)
    # Same initial state for u:  D_N/D = V/D - beta = F_INI - beta,
    # and u_N/u = D_N/D + g.
    solution_u = solve_ivp(rhs_u, [N_initial, 0.0],
                           [1.0, F_INI - beta_initial + g_initial],
                           rtol=VALIDATE_RTOL, atol=VALIDATE_ATOL, dense_output=True)
    if not solution_D.success or not solution_u.success:
        return None
    output = []
    for z in ZG_D:
        N = -np.log(1.0 + z)
        y, V = solution_D.sol(N)
        u, u_N = solution_u.sol(N)
        opw = float(np.interp(-N, XG, one_plus_w))
        D = y/opw
        if D == 0.0 or u == 0.0:
            return None
        beta_value = float(np.interp(-N, XG, beta_grid))
        g_value = float(np.interp(-N, XG, g_grid))
        output.append((float(z), float(V/D - beta_value), float(u_N/u - g_value)))
    return output


# ============================================================
# CHAIN HELPERS
# ============================================================
def load_sampler_checkpoint(number_of_nodes):
    arrays = load_result(f'n{number_of_nodes}_mcmc', 'mcmc', 'sampling', number_of_nodes)
    validate_sampler_arrays(arrays, NWALKERS, NN + 3 + number_of_nodes, THIN)
    return arrays


def load_flat_chain(number_of_nodes):
    arrays = load_sampler_checkpoint(number_of_nodes)
    post = post_burn_chain(arrays['thin_chain'], NBURN, THIN)
    if not len(post):
        raise ValueError('The chain is shorter than the requested burn-in')
    return to_physical(post.reshape(-1, NN + 3 + number_of_nodes), number_of_nodes)


# ============================================================
# COMMANDS
# ============================================================
def command_grid():
    try:
        parameters = load_result("start", "start", "physical", 1)["parameters"]
        print("(testing at the stored starting point)")
    except FileNotFoundError:
        parameters = np.concatenate([np.full(NN, START_EPS), [START_A], [START_CS2],
                                     [np.mean(PRIOR_S8)], [np.mean(PRIOR_CALM)]])
        print(f"(no starting point found; testing at eps_H={START_EPS} and A={START_A})")
    print(f"\nBackground-grid convergence"
          f"\nTarget: maximum distance change < {CONV_TARGET:.1e}")
    print(f"{'NGRID':>8}{'maximum change from previous grid':>38}")
    previous, chosen = None, None
    for number in [201, 401, 801, 1601, 3201, 6401, 12801]:
        x_grid = np.linspace(0.0, np.log(1.0 + Z_NODES[-1]), number)
        basis = _basis(x_grid)
        integral = np.array([np.concatenate([[0.0],
            np.cumsum(0.5*(basis[1:, i] + basis[:-1, i])*np.diff(x_grid))])
            for i in range(NN)]).T
        z_grid = np.exp(x_grid) - 1.0
        inv_hubble = np.exp(integral @ parameters[:NN])
        amplitude = parameters[NN]
        distance_h = amplitude*inv_hubble
        distance_m = amplitude*np.concatenate([[0.0],
            np.cumsum(0.5*(inv_hubble[1:] + inv_hubble[:-1])*np.diff(z_grid))])
        values = np.concatenate([np.interp(Z_NODES, z_grid, distance_m),
                                 np.interp(Z_NODES, z_grid, distance_h)])
        if previous is not None:
            difference = np.max(np.abs(values - previous))
            print(f"{number:8d}{difference:38.4e}")
            if difference < CONV_TARGET and chosen is None:
                chosen = number
        previous = values
    print()
    print("Target not reached on the tested grids." if chosen is None
          else f"First grid satisfying target: {chosen}")
    print(f"Grid adopted by the fit:       {NGRID}")


def command_start():
    from scipy.optimize import Bounds, LinearConstraint, minimize
    z_nodes = node_positions(1)
    number_of_parameters = NN + 4
    cs_index = NN + 1
    lower_bounds = np.concatenate([np.full(NN, PRIOR_EPS[0]),
        [PRIOR_A[0], 10.0**PRIOR_U[0], PRIOR_S8[0], PRIOR_CALM[0]]])
    upper_bounds = np.concatenate([np.full(NN, PRIOR_EPS[1]),
        [PRIOR_A[1], PRIOR_CS2[1], PRIOR_S8[1], PRIOR_CALM[1]]])
    bounds = Bounds(lower_bounds, upper_bounds)
    # Exact full-grid linear constraints:
    #   1. eps_H <= 0
    #   2. eps_H - 1.5 c_s^2 >= -1.5
    # The domain is shrunk by START_MARGIN (see CHOICES) so that points
    # returned by the minimiser lie strictly inside the physical domain.
    matrix = np.zeros((2*NGRID, number_of_parameters))
    matrix[:NGRID, :NN] = SMAT
    matrix[NGRID:, :NN] = SMAT
    matrix[NGRID:, cs_index] = -1.5
    lower = np.concatenate([np.full(NGRID, -np.inf), np.full(NGRID, -1.5 + START_MARGIN)])
    upper = np.concatenate([np.full(NGRID, -START_MARGIN), np.full(NGRID, np.inf)])
    physical_constraint = LinearConstraint(matrix, lower, upper)

    def objective(vector):
        # Same parameter ranges and physical domain as log_prob: points
        # outside them are rejected before the growth equation is solved.
        # Inside the domain the objective is unchanged.
        if np.any(vector < lower_bounds) or np.any(vector > upper_bounds):
            return 1.0e100
        eps_grid = SMAT @ vector[:NN]
        if np.any(eps_grid > 0.0) or np.any(eps_grid - 1.5*vector[cs_index] < -1.5):
            return 1.0e100
        try:
            value = chi2_total(vector, z_nodes)
        except Exception:
            return 1.0e100
        return 1.0e100 if not np.isfinite(value) else value

    def admissible_initial_point(vector):
        if np.any(vector <= lower_bounds) or np.any(vector >= upper_bounds):
            return False
        eps_grid = SMAT @ vector[:NN]
        if np.any(eps_grid > 0.0):
            return False
        if np.any(eps_grid - 1.5*vector[cs_index] < -1.5):
            return False
        return True

    # PREPARATION.  The eleven-parameter problem has a rough landscape and
    # SLSQP stalls when started far from the solution.  The background alone,
    # constrained by the thirteen BAO measurements, is a well-behaved
    # eight-parameter problem, so it is solved first and used to seed the
    # restarts.  This changes no assumption and no likelihood: it only decides
    # where the search begins.  Restart 0 uses the seed; the others are
    # scattered around it, and the best of all is kept.
    def objective_bao(vector):
        try:
            inv_h, d_m, d_h, _ = background(vector[:NN], vector[NN])
        except Exception:
            return 1.0e100
        if not (np.all(np.isfinite(d_m)) and np.all(np.isfinite(d_h))):
            return 1.0e100
        value = chi2_bao(d_m, d_h)
        return 1.0e100 if not np.isfinite(value) else value

    seed_vector = np.concatenate([np.full(NN, START_EPS), [START_A], [START_CS2],
                                  [np.mean(PRIOR_S8)], [np.mean(PRIOR_CALM)]])
    bao_only = minimize(objective_bao, seed_vector, method="SLSQP",
                        bounds=bounds, constraints=[physical_constraint],
                        options={"maxiter": 1000, "ftol": 1.0e-12})
    if np.isfinite(bao_only.fun) and bao_only.fun < 1.0e90:
        seed_vector = bao_only.x.copy()
        print(f"\nBackground-only preparation: BAO chi2 = {bao_only.fun:.4f}")
    else:
        print("\nBackground-only preparation failed; using the flat seed")
    # CALM and sigma_8 are linear in the model, so their optima at the seed
    # background are found directly rather than searched for.
    inv_h, d_m, d_h, q_seed = background(seed_vector[:NN], seed_vector[NN])
    good = np.interp(Z_SN, ZGRID, d_m) > 0.0
    if np.all(good):
        base_mu = 5.0*np.log10((1.0 + Z_SN)*np.interp(Z_SN, ZGRID, d_m))
        ones = np.ones(NSN)
        seed_vector[-1] = float((ones @ CINV_SN @ (MU_SN - base_mu))
                                /(ones @ CINV_SN @ ones))
    shape = growth_shape(cs2_on_grid(seed_vector[cs_index:cs_index + 1], z_nodes),
                         q_seed, inv_h, seed_vector[NN])
    if shape is not None:
        sigma8_seed = float((shape @ CINV_G @ FS8)/(shape @ CINV_G @ shape))
        seed_vector[-2] = float(np.clip(sigma8_seed, PRIOR_S8[0] + 1.0e-6,
                                        PRIOR_S8[1] - 1.0e-6))
    print(f"Seed: A = {seed_vector[NN]:.4f}, sigma_8 = {seed_vector[-2]:.4f}, "
          f"CALM = {seed_vector[-1]:.4f}, seed chi2 = {objective(seed_vector):.4f}")

    rng = np.random.default_rng(SEED)
    best_result = None
    print(f"\nConstrained minimisation with {N_RESTARTS} independent starts")
    for restart in range(N_RESTARTS):
        if restart == 0:
            initial = seed_vector.copy()
        else:
            spread = np.concatenate([np.full(NN, 0.06), [0.30],
                                     [0.0], [0.03], [0.05]])
            for _ in range(10000):
                candidate = seed_vector + spread*rng.standard_normal(NN + 4)
                candidate[NN + 1] = 10.0**rng.uniform(-7.5, -6.0)
                if admissible_initial_point(candidate):
                    initial = candidate
                    break
            else:
                raise RuntimeError("Could not generate an admissible start")
        result = minimize(objective, initial, method="SLSQP", bounds=bounds,
                          constraints=[physical_constraint],
                          options={"maxiter": 1000, "ftol": 1.0e-12, "disp": False})
        valid = (np.isfinite(result.fun) and result.fun < 1.0e90
                 and np.isfinite(log_prob_physical(result.x, z_nodes)))
        print(f"  restart {restart:2d}: chi2 = {result.fun:12.4f}   "
              f"{'accepted' if valid else 'rejected'}", flush=True)
        if valid and (best_result is None or result.fun < best_result.fun):
            best_result = result
    if best_result is None:
        raise RuntimeError("No admissible minimum was found")
    parameters = best_result.x
    eps, amplitude, cs2_nodes, sigma8, cal_m = unpack(parameters, 1)
    inv_hubble, distance_m, distance_h, q_grid = background(eps, amplitude)
    cs2_grid = cs2_on_grid(cs2_nodes, z_nodes)
    w_grid = -1.0 - (2.0/3.0)*(SMAT @ eps)
    number_of_parameters = len(parameters)
    degrees_of_freedom = NDATA - number_of_parameters
    print(f"\nBest chi2 = {best_result.fun:.4f}")
    print(f"Measurements: {NDATA} | parameters: {number_of_parameters}"
          f" | dof: {degrees_of_freedom}")
    print(f"Reduced chi2 = {best_result.fun/degrees_of_freedom:.4f}")
    print(f"BAO chi2    = {chi2_bao(distance_m, distance_h):.4f}")
    print(f"SNe chi2    = {chi2_sne(distance_m, cal_m):.4f}")
    print(f"growth chi2 = {chi2_growth(cs2_grid, q_grid, inv_hubble, amplitude, sigma8):.4f}")
    print(f"c_s^2       = {cs2_nodes[0]:.6e}")
    print(f"sigma_8     = {sigma8:.6f}")
    print(f"A           = {amplitude:.6f}")
    print(f"CALM        = {cal_m:.6f}")
    print("\nFull-grid physical check:")
    print(f"  max(w+c_s^2) = {np.max(w_grid + cs2_grid):+.4e}  [must be <= 0]")
    print(f"  min(w+1)     = {np.min(w_grid + 1.0):+.4e}  [must be >= 0]")
    save_result('start', {'parameters': parameters}, 'start', 'physical', 1)
    print(f"\nSaved physical starting point: {result_path('start')}")


def command_nodes():
    from scipy.optimize import minimize
    initial_one_node = load_result('start', 'start', 'physical', 1)['parameters']
    # BIC penalty per extra parameter uses the number of data points in the
    # likelihood being maximised.  The likelihood here is JOINT, so that is
    # NDATA = 52, not the 17 growth measurements alone.
    threshold = np.log(NDATA)
    print(f"\nNode scan on the joint likelihood")
    print(f"Penalty per extra c_s node: ln({NDATA}) = {threshold:.4f}")
    rng = np.random.default_rng(SEED)
    results = {}
    for number_of_nodes in (1, 2, 3):
        z_nodes = node_positions(number_of_nodes)
        base = np.concatenate([initial_one_node[:NN + 1],
            np.full(number_of_nodes, initial_one_node[NN + 1]),
            initial_one_node[-2:]])
        base = to_sampling(base, number_of_nodes)

        def objective(vector, z_nodes=z_nodes):
            probability = log_prob(vector, z_nodes)
            return 1.0e100 if not np.isfinite(probability) else -2.0*probability

        best = None
        for restart in range(NODE_RESTARTS):
            if restart == 0:
                initial = base.copy()
            else:
                scale = np.abs(base)*2.0e-3 + 1.0e-9
                for _ in range(10000):
                    candidate = base + scale*rng.standard_normal(len(base))
                    if np.isfinite(log_prob(candidate, z_nodes)):
                        initial = candidate
                        break
                else:
                    continue
            result = minimize(objective, initial, method="Nelder-Mead",
                options={"maxiter": 30000, "maxfev": 30000, "fatol": 1.0e-8,
                         "xatol": 1.0e-10, "disp": False})
            valid = (np.isfinite(result.fun) and result.fun < 1.0e90
                     and np.isfinite(log_prob(result.x, z_nodes)))
            if valid and (best is None or result.fun < best.fun):
                best = result
        if best is None:
            print(f"  {number_of_nodes} node(s): minimisation failed")
            continue
        chi2 = float(best.fun)
        # Terms common to all models cancel; only the node count changes.
        bic_relative = chi2 + number_of_nodes*threshold
        physical_best = to_physical(best.x, number_of_nodes)
        results[number_of_nodes] = {"chi2": chi2, "bic": bic_relative,
                                    "parameters": physical_best}
        save_result(f'best_{number_of_nodes}', {'parameters': physical_best},
                    'best_start', 'physical', number_of_nodes)
        print(f"  {number_of_nodes} node(s): chi2 = {chi2:10.4f}"
              f" | relative BIC = {bic_relative:10.4f}", flush=True)
    if not results:
        raise RuntimeError("The node scan failed for every model")
    chosen = min(results, key=lambda number: results[number]["bic"])
    tested = sorted(results)
    save_result('node_choice', {'chosen': np.array(chosen), 'tested': np.array(tested),
                'chi2': np.array([results[n]['chi2'] for n in tested]),
                'bic': np.array([results[n]['bic'] for n in tested])},
                'node_choice', 'none', 0)
    print(f"\nSelected number of c_s^2 nodes: {chosen}")
    reference = results[min(results)]["chi2"]
    print("\nImprovement relative to the fewest nodes:")
    for number in sorted(results):
        print(f"  {number} node(s): Delta chi2 = "
              f"{reference - results[number]['chi2']:+.4f}")


def command_mcmc():
    import emcee
    number_of_nodes = selected_node_number()
    z_nodes = node_positions(number_of_nodes)
    dimension = NN + 3 + number_of_nodes
    name = f'n{number_of_nodes}_mcmc'
    if result_path(name).exists():
        arrays = load_sampler_checkpoint(number_of_nodes)
        done = int(arrays['done'].item())
        kept = arrays['thin_chain']
        state = emcee.State(arrays['coords'], log_prob=arrays['log_prob'],
                            random_state=unpack_random_state(arrays))
        print(f'Resuming from {done} completed steps, with saved random state')
    else:
        done, kept, state = 0, np.empty((0, NWALKERS, dimension)), None
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else max(0, NSTEPS - done)
    if steps < 0:
        raise ValueError('Additional steps must be nonnegative')
    if steps == 0:
        print(f'No additional steps requested; {done} completed')
        return
    if state is None:
        if result_path(f'best_{number_of_nodes}').exists():
            central = load_result(f'best_{number_of_nodes}', 'best_start',
                                  'physical', number_of_nodes)['parameters']
        else:
            one = load_result('start', 'start', 'physical', 1)['parameters']
            central = np.concatenate([one[:NN + 1],
                       np.full(number_of_nodes, one[NN + 1]), one[-2:]])
        central = to_sampling(central, number_of_nodes)
        scale = np.abs(central)*1.0e-3 + 1.0e-9
        coords = np.empty((NWALKERS, dimension))
        rng = np.random.default_rng(SEED)
        for walker in range(NWALKERS):
            for _ in range(100000):
                candidate = central + scale*rng.standard_normal(dimension)
                candidate[cs_slot(number_of_nodes)] = rng.uniform(*PRIOR_U, number_of_nodes)
                if np.isfinite(log_prob(candidate, z_nodes)):
                    coords[walker] = candidate
                    break
            else:
                raise RuntimeError(f'Could not initialise walker {walker}')
        state = emcee.State(coords, random_state=np.random.RandomState(SEED).get_state())
    sampler = emcee.EnsembleSampler(NWALKERS, dimension, log_prob, args=(z_nodes,))
    sampler.random_state = state.random_state
    start_time = time.time()
    final_state = sampler.run_mcmc(state, steps, progress=False)
    kept = np.concatenate([kept, retained_segment(sampler.get_chain(), done, THIN)], axis=0)
    arrays = {'coords': final_state.coords, 'log_prob': final_state.log_prob,
              'thin_chain': kept, 'done': np.array(done + steps),
              **pack_random_state(final_state.random_state)}
    validate_sampler_arrays(arrays, NWALKERS, dimension, THIN)
    save_result(name, arrays, 'mcmc', 'sampling', number_of_nodes)
    print(f'Completed raw steps: {done + steps}; elapsed {time.time() - start_time:.1f} s')
    print(f'Mean acceptance fraction (this segment): {np.mean(sampler.acceptance_fraction):.4f}')
    print(f'Saved checkpoint: {result_path(name)}')
    report_convergence(kept, done + steps)


def report_convergence(thin_chain, raw_steps_done):
    """The rule of fifty autocorrelation lengths, printed after every run."""
    import emcee
    retained_raw = raw_steps_done - NBURN
    if retained_raw <= 0:
        print(f"Convergence: still inside the burn-in ({raw_steps_done} of "
              f"{NBURN} burn-in steps).")
        return
    post = post_burn_chain(thin_chain, NBURN, THIN)
    if len(post) < 50:
        print("Convergence: retained chain too short to estimate the "
              "autocorrelation time yet.")
        return
    try:
        tau_thin = emcee.autocorr.integrated_time(post, quiet=True)
    except Exception as error:
        print(f"Convergence: autocorrelation estimate unavailable ({error}).")
        return
    tau_raw = float(np.max(tau_thin))*THIN
    needed_raw = 50.0*tau_raw
    print(f"Convergence: integrated autocorrelation time {tau_raw:.0f} raw "
          f"steps; retained {retained_raw} of the {needed_raw:.0f} the rule "
          f"of fifty lengths asks for "
          f"({'MET' if retained_raw >= needed_raw else 'NOT met: run '
             f'{max(0, needed_raw - retained_raw):.0f} more raw steps'}).")


def command_converged():
    arrays = load_sampler_checkpoint(selected_node_number())
    report_convergence(arrays['thin_chain'], int(arrays['done'].item()))


def command_summary():
    number_of_nodes = selected_node_number()
    z_nodes = node_positions(number_of_nodes)
    try:
        flat_chain = load_flat_chain(number_of_nodes)
    except FileNotFoundError:
        print("No chain found. Run `mcmc` first.")
        return
    lower, median, upper = np.percentile(flat_chain, [16.0, 50.0, 84.0], axis=0)
    print(f"\nPosterior samples after burn-in: {len(flat_chain)}")
    print(f"Number of c_s^2 nodes: {number_of_nodes}")
    for node in range(number_of_nodes):
        index = NN + 1 + node
        print(f"\nc_s^2 node {node + 1} at z={z_nodes[node]:.4f}:")
        print(f"  {median[index]:.6e} -{median[index]-lower[index]:.2e}"
              f" +{upper[index]-median[index]:.2e}")
        print(f"  95% upper limit: {np.percentile(flat_chain[:, index], 95):.6e}")
    sigma8_index = NN + 1 + number_of_nodes
    cal_m_index = sigma8_index + 1
    print(f"\nsigma_8:\n  {median[sigma8_index]:.6f}"
          f" -{median[sigma8_index]-lower[sigma8_index]:.6f}"
          f" +{upper[sigma8_index]-median[sigma8_index]:.6f}")
    print(f"\nA    = {median[NN]:.6f}")
    print(f"CALM = {median[cal_m_index]:.6f}")
    # chi^2 of every stored sample; the lowest is reported as the best sampled value.
    chi2_values = np.array([chi2_total(p, z_nodes) for p in flat_chain])
    finite = np.isfinite(chi2_values)
    if not np.any(finite):
        raise RuntimeError("No finite chi-squared values in the chain")
    finite_indices = np.where(finite)[0]
    local_best = finite_indices[np.argmin(chi2_values[finite])]
    best_parameters, best_chi2 = flat_chain[local_best], chi2_values[local_best]
    number_of_parameters = NN + 3 + number_of_nodes
    degrees_of_freedom = NDATA - number_of_parameters
    print(f"\nBest sampled chi2 = {best_chi2:.4f}")
    print(f"Measurements: {NDATA} | parameters: {number_of_parameters}"
          f" | dof: {degrees_of_freedom}")
    print(f"Reduced chi2 = {best_chi2/degrees_of_freedom:.4f}")
    # This diagnostic refers only to the node values of w.
    W_nodes = -1.0 - (2.0/3.0)*flat_chain[:, :NN]
    monotonic_fraction = np.mean(np.all(np.diff(W_nodes, axis=1) >= -1.0e-12, axis=1))
    print(f"\nPosterior samples with monotonic w at the seven nodes: "
          f"{100.0*monotonic_fraction:.2f}%")
    save_result(f'n{number_of_nodes}_posterior',
                {'chain': flat_chain, 'chi2': chi2_values,
                 'best_sampled_parameters': best_parameters,
                 'best_sampled_chi2': np.array(best_chi2)},
                'posterior', 'physical', number_of_nodes)
    print(f"\nSaved physical posterior: {result_path(f'n{number_of_nodes}_posterior')}")


def command_validate():
    number_of_nodes = selected_node_number()
    try:
        flat_chain = load_flat_chain(number_of_nodes)
        sample = posterior_subset(flat_chain, VALIDATE_SAMPLES, SEED)
        count = len(sample)
        source = f"{count} posterior samples"
    except FileNotFoundError:
        sample = load_result('start', 'start', 'physical', 1)['parameters'][None, :]
        source = "the starting point only (no usable chain found)"
    print("\nClosure diagnostic at k=0")
    print(f"Source: {source}")
    worst_differences, first_valid, skipped = [], None, 0
    for parameters in sample:
        result = validate_zero_mode(parameters[:NN], parameters[NN])
        if result is None:
            skipped += 1
            continue
        differences = []
        for _, growth_D, growth_u in result:
            if (not np.isfinite(growth_D) or not np.isfinite(growth_u)
                    or abs(growth_u) < 1.0e-12):
                continue
            differences.append(abs(growth_D/growth_u - 1.0))
        if not differences:
            skipped += 1
            continue
        worst_differences.append(max(differences))
        if first_valid is None:
            first_valid = result
    if not worst_differences:
        print("No sample produced a usable diagnostic.")
        return
    worst_percent = 100.0*np.asarray(worst_differences)
    print(f"Usable samples: {len(worst_percent)}")
    print(f"Skipped samples: {skipped}")
    print("\nMaximum absolute fractional difference within each sample:")
    print(f"  median:          {np.median(worst_percent):.3f}%")
    print(f"  68th percentile: {np.percentile(worst_percent, 68):.3f}%")
    print(f"  95th percentile: {np.percentile(worst_percent, 95):.3f}%")
    print(f"  maximum:         {np.max(worst_percent):.3f}%")
    if first_valid is not None:
        print("\nFirst usable sample, redshift by redshift:")
        print(f"{'z':>8}{'f (growth)':>16}{'f (u-equation)':>18}{'difference %':>16}")
        for z, growth_D, growth_u in first_valid:
            difference = (np.nan if abs(growth_u) < 1.0e-12
                          else 100.0*abs(growth_D/growth_u - 1.0))
            print(f"{z:8.3f}{growth_D:16.6f}{growth_u:18.6f}{difference:16.3f}")


def print_usage():
    print("""
Usage:
    python3 fit_joint.py grid
    python3 fit_joint.py start
    python3 fit_joint.py nodes
    python3 fit_joint.py mcmc [additional_steps]
    python3 fit_joint.py summary
    python3 fit_joint.py validate
    python3 fit_joint.py posterior
""")


def command_posterior():
    """Headline numbers from the posterior saved by `summary`; no fitting."""
    chain = posterior_chain()
    cs2, sigma8 = chain[:, NN + 1], chain[:, NN + 2]
    print(f"Saved posterior: {chain.shape[0]} samples")
    print(f"c_s^2 95% upper limit: {np.percentile(cs2, 95):.4e}")
    for floor in (1e-12, 1e-10, 1e-9, 1e-8):
        print(f"  lower prior cutoff {floor:.0e}: 95% upper limit "
              f"{np.percentile(cs2[cs2 >= floor], 95):.4e}")
    low, mid, high = np.percentile(sigma8, [16.0, 50.0, 84.0])
    print(f"sigma_8 (normalising D at z = {Z_NORM}) = {mid:.4f} +{high - mid:.4f} -{mid - low:.4f} (68%)")
    print(f"correlation(c_s^2, sigma_8) = {np.corrcoef(cs2, sigma8)[0, 1]:.3f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    commands = {'grid': command_grid, 'start': command_start, 'nodes': command_nodes,
                'mcmc': command_mcmc, 'converged': command_converged,
                'summary': command_summary, 'validate': command_validate,
                'posterior': command_posterior}
    if command not in commands:
        print_usage()
        sys.exit(f'Unknown command: {command}')
    print(f'Prior: {CS2_PRIOR}; data points: {NDATA}; run directory: {run_directory()}')
    if command in ('start', 'nodes', 'mcmc', 'summary'):
        with exclusive_run(run_directory()):
            commands[command]()
    else:
        commands[command]()
