# NeTT — Neural Tensor Tomography

NeTT reconstructs, for every voxel of a sample, the **reciprocal-space map (RSM)** of small-/wide-angle X-ray
scattering tensor tomography (SAXS/WAXS-TT) as a set of spherical-harmonic (SH) coefficients. The coefficients are
the output of an *implicit neural field*: a multi-resolution feature grid and a small MLP evaluated at the voxel
coordinates, trained against the measured projections of one scan, with no external data.

This repository contains **everything needed to reproduce the results of the paper**: the NeTT method (one fixed
recipe), the [mumott](https://mumott.org) baselines (SH, GK, ZH), the data preparation, the experiment grid
(tables, ablations of the Supplementary Information, noise and sparsity studies), the evaluation, and the table /
figure generators.

```python
from nett import NeTTConfig, fit
from nett.data import registry, splits

ds = registry.get("narwhal-tusk")                       # any mumott DataContainer works, see below
train, held = splits.split(ds.dc(), "random", 0.15, 42)
result = fit(train, NeTTConfig(ell_max=8), held_dc=held)     # ell_max is the only parameter you choose
coeffs = result.coefficients                            # (X, Y, Z, C) float32 SH coefficients, mumott ordering
```

For your own data, load a mumott `DataContainer` and call `fit(dc, NeTTConfig(ell_max=...))`.
`ell_max` (even, 2–12 in the paper) is the angular band limit and depends on the sample; everything else is fixed
by the recipe (see `nett/config.py`).

## Install

NeTT needs a CUDA GPU. Dependencies: PyTorch, [mumott 2.3](https://mumott.org), the
[ASTRA toolbox](https://astra-toolbox.com) (CUDA build, for the X-ray transform) and CuPy.

```bash
conda env create -f environment.yml && conda activate nett      # or: see Dockerfile
nett check                                                       # prints versions and whether CUDA works
```

The paper's numbers were produced with torch 2.8, mumott 2.3 and astra-toolbox 2.3; the validation (`docs/VALIDATION.md`) was run with torch 2.11, mumott 2.3 and astra-toolbox 2.3.0 and agrees.

## Reproduce the paper

```bash
export NETT_DATA=$PWD/data NETT_RESULTS=$PWD/results
nett data download && nett data prepare       # Zenodo -> $NETT_DATA (see docs/DATA.md)

nett plan                                      # the 464 runs behind all tables/figures (experiment, dataset, method, seed ...)
nett run table1/nielsen-t/NeTT_s0              # one run (~1 GPU-hour for NeTT, <1 h for baselines)
nett worker --experiment table1 table2 bone    # work through experiments; start several workers in parallel
sbatch scripts/slurm/node_workers.sbatch       # SLURM: one node, 4 GPUs x N workers pulling from the same queue

nett tables                                    # Tables 1/2, SI tables, numbers.tex, summary.csv
nett figures                                   # all figures
nett validate                                  # compare with the numbers printed in the paper
```

| experiment | content | runs |
|---|---|---|
| `table1` | 3 simulated phantoms × {NeTT ×4 seeds, SH, GK, ZH}, ground-truth metrics (Table 1, SI isotropic/RSM-corr tables) | 21 |
| `table2` | 4 experimental datasets × {random, high-tilt holdout} × {NeTT ×4, SH, GK, ZH}, held-out NRMSE (Table 2) | 56 |
| `bone` | trabecular bone, two mounts: cross-mount NRMSE and orientation agreement | 18 |
| `noise` | Poisson rates 10 and 31.6 (100 = `table1`) × {NeTT, SH, GK} | 18 |
| `sparsity` | {Max-Min, 3 random} pruning × levels 100…20 % (+10 %) × {NeTT, SH, GK} | 342 |
| `control` | SI "no network": free per-voxel coefficients, learning-rate multiplier ×3/×10/×30 | 9 |

Every run writes `<results>/<id>.json` (configuration, metrics, timings, software versions) and `<id>.npy` (the
reconstruction). Everything downstream (`nett tables`, `nett figures`) reads only these files; `expected/paper_numbers.json`
holds the numbers of the paper, and `nett validate` compares them with your results (rule: the mean of the new
NeTT runs lies within 2 paper standard deviations of the paper mean; baselines within 2 %).

A quick check that everything works on a new machine (minutes): `nett fit nielsen-t --set n_iterations=500`.

## What is where

```
nett/
  config.py        the recipe (NeTTConfig): ell_max + frozen defaults; ablations are `--set key=value` overrides
  field.py         NeTT field (feature grid + MLP + SH head) and the "no network" control field
  encoding.py      multi-resolution feature grid;  schedule.py  coarse-to-fine spatial/angular reveals
  sh.py            real spherical harmonics, SH -> detector-segment integration matrix (SAXS flat plane / WAXS Ewald curvature)
  projector.py     differentiable X-ray transform (ASTRA, adjoint as backward pass)
  train.py         fit(): the training loop (the whole method is ~200 lines)
  baselines/       mumott SH / GK / ZH exactly as run for the paper (shipped SAXSProjectorCUDA, unregularised, L-BFGS)
  data/            registry (datasets, ground truth), Zenodo download, preparation, splits (hold-out, pruning)
  metrics.py       ground-truth metrics, held-out NRMSE, orientation agreement
  experiments/     plan.py (the grid), execute.py (one run), worker.py (file-based queue)
  report/          tables, numbers.tex macros, figures, validate
tests/             CPU unit tests (`pytest`) and GPU tests (`pytest -m gpu`, needs CUDA + ASTRA)
scripts/slurm/     example SLURM job
expected/          numbers of the paper
docs/              DATA.md, DESIGN_NOTES.md, VALIDATION.md
```

## Things worth knowing

* **Two forward operators, on purpose.** NeTT uses the ASTRA toolbox for the X-ray transform; the mumott baselines use
  mumott's own `SAXSProjectorCUDA` as released. All methods are *evaluated* with the same operator (ASTRA), so
  held-out scores are comparable. Details in `docs/DESIGN_NOTES.md`.
* **Baselines are unregularised and run to convergence** (Laplacian weight 0; L-BFGS limits raised from mumott's
  10/20 to 500/2000 iterations/evaluations), see `nett/baselines/mumott_baselines.py`.
* **Seeds.** The paper reports the mean ± SD of 4 NeTT runs (`seed = 0…3`). GPU kernels are not bit-reproducible; expect
  run-to-run differences at the level of the reported SD.
* **Data preparation changes two files** (aragonite 96 → 48 azimuthal bins; one hot pixel of the steel-wire data
  masked). Both are in `nett/data/prepare.py`; without them the numbers differ. See `docs/DATA.md`.

## Citation

See `CITATION.cff`. mumott is MPL-2.0 and is used as an unmodified dependency; NeTT is MIT-licensed.
