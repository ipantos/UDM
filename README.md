# Reconstructing a Born-Infeld scalar for unified dark matter and dark energy

Code, data, stored posterior and logs of the joint distance-and-growth
analysis described in the paper.

## Contents

| path | role |
|---|---|
| `udm_data.txt` | all observational inputs: DESI DR2 BAO, Union3 supernova moduli and covariance, the growth compilation with its scale windows, the WiggleZ covariance and the eBOSS correlation matrix |
| `fit_joint.py` | background, sound-speed and sigma_8 likelihood; starting point, node scan, MCMC, posterior summary and numerical validation |
| `figures.py` | figure data, PGFPlots sources and PDF figures from the stored posterior; writes `figure_data/numerical_validation.json` |
| `checks.py` | sensitivity of the sound-speed limit by importance reweighting of the stored posterior, response of each growth measurement to the sound speed, within-chain stability of posterior summaries; writes `checks.json` |
| `runs/joint_norm051_log_454989fb0b76b3f2/` | the supplied execution: `start.npz`, `best_{1,2,3}.npz`, `node_choice.npz`, `n1_mcmc.npz` (sampler checkpoint with the thinned chain of all 40 000 steps and the random-generator state), `n1_posterior.npz` (the 40 000 stored post-burn-in samples in physical coordinates) |
| `logs/` | the terminal output of every stage of that run, in the order it was produced: `grid.log`, `start.log`, `nodes.log`, `mcmc_seg1.log` ... `mcmc_seg8.log`, `summary.log`, `validate.log`, `checks.log`; `figures.log` records figure generation |
| `figure_data/`, `figure_tex/`, `figures/` | the output of `figures.py` from the stored run: the plotted quantities as CSV files and `numerical_validation.json`; the PGFPlots sources; the four generated figures |
| `checks.json` | the output of `checks.py` from the stored run |

Keep `udm_data.txt`, `fit_joint.py`, `figures.py` and `checks.py` in one
directory with `runs/` beside them. `figures.py` and `checks.py` read the
stored run directly, so the figures and all derived numbers can be
regenerated without rerunning the chain.

## Requirements

This execution uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and
emcee 3.1.6. Install the Python dependencies with
`python3 -m pip install -r requirements.txt`. The figures are compiled with `pdflatex` and need
the packages `standalone`, `pgfplots` (compat 1.18), `tikz`, `amsmath` and `lmodern`.
The numerical commands were run with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. Exact software and platform metadata are recorded in
the NPZ files. Matching package versions alone does not guarantee bitwise
identity across different platforms.

The supplied MCMC consists of eight completed segments of 5,000 steps per
walker, resumed using saved walker positions and random-generator state.
Every supplied NPZ records the SHA-256 of the included `fit_joint.py`.
`run_summary.json` records the principal results and provenance checks.

## Using the supplied posterior

Run commands from the directory containing this README. With
`UDM_OUTPUT_DIR` unset, the programs read the supplied `runs/` directory.
The following command prints the principal posterior summaries without
running the sampler:

```bash
python3 fit_joint.py posterior
```

The figure and sensitivity commands below also use this stored posterior.
Do not run `start`, `nodes`, `mcmc`, or `summary` in this directory merely to
inspect the supplied results: these commands write run files.

## Generating a new posterior

Use a separate, initially empty output directory to preserve the supplied
results. In a Linux/macOS shell, set:

```bash
export UDM_OUTPUT_DIR="$PWD/reproduction"
```

In Windows PowerShell, use instead:

```powershell
$env:UDM_OUTPUT_DIR = Join-Path (Get-Location) "reproduction"
```

Choose another directory name if `reproduction` already contains a run.
Keep this environment setting active for all commands in the new calculation.
On systems where the executable is named `python` rather than `python3`,
substitute `python` in the commands below.

Run the commands in this order. Runtime depends on the machine and software environment.

```
python3 fit_joint.py grid          # background grid and spline checks
python3 fit_joint.py start         # BAO-only seed, analytic sigma_8 and M, 12 constrained minimizations
python3 fit_joint.py nodes         # local searches for 1, 2 and 3 nodes; heuristic BIC selection
python3 fit_joint.py mcmc 40000    # 40 walkers x 40 000 steps (see below)
python3 fit_joint.py summary       # medians, intervals, 95% upper limit, best sampled chi^2
python3 fit_joint.py validate      # growth rate at k = 0 from the u-equation for 300 samples
```

The argument to `mcmc` specifies **additional steps per walker**. Thus
`mcmc 40000` starts a 40,000-step chain only when the selected output directory
contains no sampler checkpoint. With an existing checkpoint it resumes the
chain and adds another 40,000 steps. Running `mcmc` without a step argument
instead adds only the steps needed to reach the configured total of 40,000.
The checkpoint preserves the walker positions and random-generator state.

Results are stored under `UDM_OUTPUT_DIR/runs/<run identity>/`, or under
`runs/<run identity>/` beside the scripts when the variable is unset.
The run identity hashes the configured settings and input data. Source-code
hashes and library versions are recorded in the NPZ metadata but do not
change that identity. Use a separate output directory for each new calculation.

The node scan uses six Nelder--Mead searches per node number. The first starts
at the base vector; the others use Gaussian perturbations in sampling
coordinates with standard deviations `0.002 * abs(base) + 1e-9`.
It writes the optimized vectors and a heuristic BIC-based node choice.
The supplied results select one sound-speed node, and the optimized
constant-sound-speed vector supplies the central values of the other
coordinates for walker initialization. The sampled log10 sound speed is
drawn uniformly over its prior range. These local searches
do not establish a robust preference for the number of nodes; the paper's
posterior is conditional on the constant-sound-speed model. Check the node
choice before sampling in a new calculation, since the heuristic can select
a different model. The figure and sensitivity scripts supplied here are
intended for the one-node model.

To return to the supplied posterior, unset `UDM_OUTPUT_DIR` (`unset
UDM_OUTPUT_DIR` in a Linux/macOS shell, or `Remove-Item Env:UDM_OUTPUT_DIR`
in PowerShell).

## Figures and checks

These commands read the posterior selected by `UDM_OUTPUT_DIR` and overwrite
`figure_data/`, `figure_tex/`, `figures/`, and `checks.json` beside the scripts.
Use a separate copy of the package if you want to preserve the supplied
figure and check outputs while regenerating them.

```
python3 figures.py     # figure_data/, figure_tex/, figures/fig01-fig04.pdf
python3 checks.py      # checks.json
```

`figure_data/numerical_validation.json` records the numerical checks for this posterior (spline tests, step refinement, solver
comparison, prior-cutoff limits, reconstruction medians at the nodes, Jeans
estimate) together with the SHA-256 of the posterior chain (of the 40 000 x 11
array of physical parameter samples, excluding the separately stored
chi-squared values; not of the `.npz` file), of the data file and of
`figures.py` itself. `checks.json` records the reweighted upper limits for
the alternative initial velocity rates and for the two treatments of the
z = 0.07 measurement, their effective sample sizes, the response of every
growth measurement to the sound speed at the best stored sample, and the
upper limits and sigma_8 medians obtained from the two halves of the retained
chain and from the two groups of twenty walkers split by their initial
log10 c_s^2. These are within-chain stability diagnostics; they do not
by themselves establish independence from the initialization.

## Settings

All settings are constants at the top of `fit_joint.py`: the seven background
nodes, the 3201-point grid in ln(1+z), the priors, the physical restrictions,
the integrator tolerances, the normalization redshift z = 0.51, the initial
conditions at z = 2.33, and the sampler configuration (40 walkers, 40 000
steps, 20 000 burn-in, thinning 20, seed 20260809).

## File integrity

`SHA256SUMS.txt` records the SHA-256 of every other supplied file, with paths
relative to this directory. On systems with `sha256sum`, verify the package
from this directory with:

```bash
sha256sum -c SHA256SUMS.txt
```

Regenerating outputs changes their hashes. The manifest describes the
supplied package, not subsequently regenerated files.

## License

The source code is distributed under the MIT License; see `LICENSE`.
Observational data retain their original source attribution in `udm_data.txt`.
