# Supplementary checks

This folder contains the code and the results of three checks reported in
the Supplemental Material. They use the likelihood and the equations of the
code in the parent folder, which the scripts import directly; the stored fits
of the parent folder (`assumptions.json`, `extensions.json`) supply starting
points.

| Check | Supplemental Material | Scripts | Results in `results/` |
|---|---|---|---|
| Comparison of this driver with the code of the parent folder | | `audit_engine.py` | `validation.json` |
| Profiles without the three low-redshift or the six DESI growth measurements | Table SIII | `run_profiles.py`, `refine_profiles.py` | `profile_*.json`, `refined_limits.json` |
| Coverage of the $\Delta\chi^2=2.71$ level in the general and minimal models | Sec. SIV B, Table SVI | `run_coverage.py`, `coverage_minimal.py`, `audit_coverage_profiles.py`, `repair_coverage.py` | `coverage_*` |
| DESI DR1 windows and the Alcock–Paczyński effect | Sec. SIV C | `window_response.py` | `window_response.json`, `fiducial_linear_template.npz` |

## Running the checks

Install the requirements of this folder (they include CAMB and h5py) in a
Python 3.12 environment and run the commands from this folder:

```bash
python -m pip install -r requirements.txt
export UDM_AUDIT_RESULTS="$PWD/fresh_results"   # new, empty folder; leaves results/ unchanged
python audit_engine.py
python run_profiles.py
python refine_profiles.py
python run_coverage.py pressureless 1000
python run_coverage.py scalar 1000
python coverage_minimal.py pressureless
python coverage_minimal.py scalar
python audit_coverage_profiles.py pressureless
python audit_coverage_profiles.py scalar
python window_response.py
```

Without `UDM_AUDIT_RESULTS` the scripts write into `results/`. The full
sequence took about 40 minutes in our check; the coverage of the
general model takes most of it. `window_response.py` needs the profiles, so
it runs after `refine_profiles.py`.

The general coverage runs resume by simulation index; the minimal ones
recompute all simulated data sets. Random seeds are fixed in the scripts, so
a fresh run reproduces the stored simulations. No simulation in the stored
run failed; `repair_coverage.py` restarts a failed fit, if one occurs,
without discarding it.

CAMB's Fortran file loader can fail on paths with non-ASCII characters. In
that case set `UDM_CAMB_PACKAGE_ROOT` to the parent folder of an unchanged
`camb` package.

## Contents of the results

In all result files `cs` is the squared sound speed $c_s^2$, the present
value for the minimal model. `factor` is the true sound speed of the
simulations as a fraction of the limit from the data. The subsets `all`,
`no_lowz` and `no_desi` keep all growth measurements, remove those at
$z=0.07$, $0.18$ and $0.38$, or remove the six DESI amplitudes; the BAO and
supernova data are kept in every case.

`coverage_general_*.jsonl` lists every simulation of the general model with
its seed, test statistic and fitted parameters; the `.summary.json` files
give counts, binomial errors and Wilson intervals. The minimal-model `.npz`
files contain the test statistic and fitted sound speed of every simulation.
`coverage_final_consistency.json` compares two complete runs of the general
coverage.

## DESI data

`desi_data/` contains the six public DESI DR1 power-spectrum likelihood
files with their windows and covariances. Their download addresses and
SHA-256 checksums are in `provenance.json`.
