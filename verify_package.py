"""Short input, solver and restart checks; this is not an MCMC convergence test."""
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import scipy
import emcee
import extensions as ext
import assumptions as asm
import fit_joint as fj
sys.path.insert(0, str(Path(__file__).resolve().parent/'crosscheck'))
import udm_indep as independent
ROOT = Path(__file__).resolve().parent

def verify():
    result = {'python': platform.python_version(), 'numpy': np.__version__,
              'scipy': scipy.__version__, 'emcee': emcee.__version__}
    data = independent.load_data(ROOT/'udm_data.txt')
    assert fj.NDATA == 52 and len(data['zg']) == 17
    assert data['zg'][0] == .07 and data['fs8'][0] == .44 and data['sig'][0] == .088
    assert data['kmin'][0] == .025 and data['kmax'][0] == .30
    assert fj.selected_node_number() == 1
    assert 'without the six DESI full-shape amplitudes' not in asm.scale_cases()
    result['qin'] = {key: float(data[key][0]) for key in ('zg', 'fs8', 'sig', 'kmin', 'kmax', 'keff')}
    official = (ROOT/'provenance/union3_official.fits').read_bytes()
    import hashlib
    assert hashlib.sha256(official).hexdigest() == 'ef98b7dde1025ee7134f3242aaea3f1f0ed3f2b29a718458c550a8e4e24d0349'
    array = np.frombuffer(official, dtype='>f8', count=23*23, offset=2880).reshape(23,23)
    covariance = np.linalg.inv(array[1:,1:])
    assert np.max(abs(data['zsn']-array[0,1:])) < 5.1e-7
    assert np.max(abs(data['musn']-array[1:,0])) < 5.1e-7
    assert np.max(abs(np.linalg.inv(data['cinv_sn'])-covariance)) < 1e-12
    assert np.linalg.eigvalsh(covariance).min() > 0
    result['union3_official_fits_match'] = True
    source_lines = json.loads((ROOT/'provenance/unchanged_inputs.json').read_text())
    sections = independent.read_sections(ROOT/'udm_data.txt')
    for section, expected in source_lines['sections'].items():
        assert sections[section] == expected, section
    actual_growth = [line for line in sections['growth'] if not line.startswith('29,0.07,')]
    assert actual_growth == source_lines['growth_except_qin']
    result['all_other_input_rows_unchanged'] = True
    eps = np.array([-.79,-.86,-1.02,-1.18,-1.32,-1.37,-1.45])
    bg = independent.Background(eps)
    errors=[]
    for cs2 in (1e-12, 3e-7, 3e-6):
        description=ext.spline_profile(eps[None,:],cs2,solver=ext.FINE)
        amp,sigma,cal = [float(description[k][0]) for k in ('A','sigma8','M')]
        value,_=ext.fit_chi2(eps,amp,cs2,sigma,cal)
        shape=independent.growth_shape(bg,data,independent.cs2_constant([cs2]))[0]
        alternate=(independent.chi2_bao(bg,amp,data)+independent.chi2_sn(bg,amp,cal,data)
                   +independent.chi2_growth(shape,sigma,data))
        errors.append(abs(value-alternate))
    assert max(errors)<.003, errors
    result['independent_chi2_abs_differences'] = errors
    # Exercise the real checkpoint writer/reader with a cheap Gaussian likelihood.
    # 17+23 steps deliberately cross the thinning boundary in two separate calls.
    saved = ext.ROOT, ext.OUT_JSON, ext.chain_log_prob, sys.argv.copy()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            ext.chain_log_prob=lambda positions,*args: -.5*np.sum(positions**2,axis=1)
            for name, segments in (('continuous',(40,)),('split',(17,23))):
                folder=root/name;folder.mkdir()
                ext.ROOT=folder;ext.OUT_JSON=folder/'extensions.json'
                ext.OUT_JSON.write_text(json.dumps({'bestfit':{'with_V_nonnegative':{'eps_nodes':eps.tolist()}}}))
                for steps in segments:
                    sys.argv=['extensions.py','chain','constant',str(steps)]
                    with contextlib.redirect_stdout(io.StringIO()): ext.command_chain()
            with np.load(root/'continuous/extensions_chains/constant.npz') as a, np.load(root/'split/extensions_chains/constant.npz') as b:
                for key in a.files:
                    assert np.array_equal(a[key],b[key]), key
    finally:
        ext.ROOT,ext.OUT_JSON,ext.chain_log_prob,sys.argv=saved
    result['chain_restart_17_plus_23_equals_40_bitwise'] = True
    result['full_scientific_runs_performed'] = False
    return result

if __name__ == '__main__':
    output=verify()
    print(json.dumps(output,indent=2))
    if '--save' in sys.argv:
        (ROOT/'verification_results.json').write_text(json.dumps(output,indent=2))
