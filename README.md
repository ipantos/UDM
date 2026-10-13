# Reconstructing a Born–Infeld scalar for unified dark matter and dark energy

This repository contains the code, the input data and the complete run on
which the paper is based: chains, profiles, all results, the log of every
stage and the five figures of the paper.

## The run of the paper

The run of the paper is the default run of `run_pipeline.py` with three
additions. All of them are recorded in the logs.

1. **Main chain extended to 60,000 steps per walker.** After the pipeline
   had finished (40,000 steps), the chain was continued from its checkpoint
   by four runs of `python fit_joint.py mcmc 5000` (`logs/mcmc.log`). The
   burn-in stayed at 20,000 steps, so 80,000 samples are retained. The
   stages that depend on the chain were then run again, in this order:
   `fit_joint.py converged`, `fit_joint.py summary`, `fit_joint.py validate`,
   `figures.py --data-only`, `checks.py`, `extensions.py posterior`,
   `extensions.py chains`, `extensions.py figure`, `export_results.py`.
2. **The fit without the six DESI full-shape amplitudes** (a row of
   Table II of the paper). It was run on a copy of this folder with
   `python assumptions.py scales --legacy-desi-diagnostic`. The result is in
   `assumptions_desi_diagnostic.json` and the full output in
   `logs/asm_scales_desi_diagnostic.log`. All other cases that the same
   command recomputed were identical to those in `assumptions.json`.
3. **The Monte Carlo uncertainty of the posterior limit** (batch means):
   `python mc_uncertainty.py`, with its output in `mc_uncertainty_output.txt`.

The five figures of the paper are in `paper_output/` and are produced by
`python paper_scripts/paper_figures.py . paper_output`.

## Getting started

Download the repository into a **new folder** and open a terminal in it.
Python 3.12, which was used for the run, is recommended.

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_pipeline.py
```

Linux / macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_pipeline.py
```

With a suitable Python environment already in place:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

Check of the package only, without the full chains:

```bash
python verify_package.py
```

List of the stages without running them:

```bash
python run_pipeline.py --plan
```

## What the pipeline runs

1. Checks of the inputs, of Union3, of an independent computation of
   $\chi^2$ and of the continuation of chains.
2. Initialization, a diagnostic comparison of the number of sound-speed
   nodes, and the main chain for a **constant** $c_s^2$: 40 walkers,
   40,000 steps per walker, burn-in 20,000, thinning 20 (for the paper the
   chain was extended to 60,000 steps; see "The run of the paper"). One node
   is the assumption of this analysis; the diagnostic comparison does not
   change it.
3. Summary, completion checks, the numerical data of the figures and the
   sensitivity checks.
4. All stages of `extensions.py`, with the three supplementary chains of
   12,000 steps per walker and their summaries.
5. The stages of `assumptions.py`, including `tracer_general`.
6. The two independent numerical cross-checks and the export of the
   results.

The default run keeps the six DESI full-shape amplitudes. The fit without
them is optional: `python assumptions.py scales --legacy-desi-diagnostic`.

## Resuming after an interruption

Run **the same command** `python run_pipeline.py` again. The pipeline skips
the completed stages and continues the chains up to the lengths given
above. The main chain writes a checkpoint every 5,000 steps and the
supplementary chains every 500; steps after the last checkpoint are
repeated after an interruption.

Do not run two pipelines at the same time in the same folder, do not copy
outputs of other runs into the folder, do not set `UDM_OUTPUT_DIR`, and do
not change code or data in between: the pipeline refuses inputs whose
identity has changed. For a different analysis use a fresh copy of the
repository.

The individual commands `extensions.py chain NAME 12000` **add** 12,000
steps. The pipeline computes by itself how many remain; use it for normal
starts and continuations.

`tracer_general` also stores the points already computed in the files
`tracer_general_*_partial.json`, marked `complete`. These are diagnostic
checkpoints: if the stage is interrupted, the pipeline repeats it from the
beginning, not from the last point.

## Results and figures

- `logs/`: the full output of every stage. Restarts are appended to the
  corresponding log.
- `runs/`, `extensions_chains/`, `assumptions_grids/`: chains, grids and
  checkpoints.
- `extensions.json`, `assumptions.json`, `checks.json`: the results of each
  analysis.
- `figure_data/`: the numerical data of the figures and the numerical
  validation.
- `crosscheck_extensions.json`, `crosscheck_assumptions.json`: the
  differences from the independent computations.
- `new_results.json`, `new_results.csv`: the main results collected in one
  place.
- `provenance/`: the sources of the inputs and their checks, including the
  official Union3 file.

The five PDF figures need `pdflatex` with `standalone`, `pgfplots`, `tikz`,
`amsmath`, `lmodern` and the PGFPlots libraries of the TeX installation.
TeX is not needed for the numerical run. With TeX installed:

```bash
python run_pipeline.py --figures
```

After the numerical stages this command skips the completed stages and
writes the PDF files to `paper_output/`.

Completing the preset number of steps does **not by itself establish
convergence** of the MCMC. See `logs/converged.log`, `checks.json`, the
diagnostics of the supplementary chains and the independent cross-checks.
