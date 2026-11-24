# Validation of the standalone repository

Goal: show that `nett` (this repository, independent of the research code `smartt`) reproduces the numbers of the
paper. Run on Clariden (GH200, 1 node, 8 workers sharing 4 GPUs; python 3.12, torch 2.11 (NGC build), mumott 2.3, astra-toolbox 2.3.0; the paper's NeTT runs used an earlier torch 2.8 / astra 2.3 environment, so this is also a check across versions). Each NeTT run took 52-161 min with 8 runs sharing the node,
2026-11-23/24. Raw results of every run: `docs/validation_runs/` (JSON; volumes not stored).

## 1. Equivalence with the research code (`tests/gpu_equivalence.py`, nielsen-t, 150 steps, same seed)

| quantity | max difference |
|---|---|
| forward projection (ASTRA), `nett.projector` vs research code | 0.0 |
| adjoint projection | 0.0 |
| training loss, every step | 0.0 (relative) |
| final coefficients | identical (corr = 1, rel. L2 = 0) |

CPU-side comparisons (against the research code, on the real data): SH integration matrix (0.0), field and annealing
schedules (identical outputs), random pruning subsets at 20/50/90 % for seeds 0-2 and the 15 % hold-out (identical
data), all ground-truth metrics on a perturbed volume (0.0), the two prepared data files of `nett data prepare`
(identical content hashes).

## 2. The four paper results, 4 independent NeTT runs each (`nett validate`)

Rule: the mean of the new NeTT runs lies within 2 paper standard deviations of the paper mean (paper: mean +- SD of
4 runs); deterministic baselines within 2 %.

| result | new (mean of 4) | paper | |diff| / (2 SD) |
|---|---|---|---|
| nielsen-m, orientation error (deg) | 2.632 | 2.525 +- 0.122 | 0.11 / 0.24 |
| narwhal tusk, random-holdout NRMSE | 0.1497 | 0.1499 +- 0.00015 | 0.00014 / 0.0003 |
| steel wire, high-tilt-holdout NRMSE | 0.5778 | 0.5761 +- 0.0051 | 0.0016 / 0.010 |
| trabecular bone, cross-mount NRMSE | 0.1783 | 0.1773 +- 0.0039 | 0.0010 / 0.0077 |

All five other nielsen-m NeTT metrics (RSM correlation, RA MAE, PSNR, SSIM, NRMSE) pass the same rule.

## 3. mumott baselines (deterministic; same settings as the paper)

nielsen-m: SH 12.39 deg, GK 4.20 deg (identical to the paper to 4 digits), ZH 7.52 deg (paper 7.59, -0.9 %).
Narwhal random holdout: SH 0.1550, GK 0.1578 (identical), ZH 0.1874 (paper 0.1864, +0.5 %).
Bone cross-mount: SH 0.2797, GK 0.2162 (identical), ZH 0.1759 (paper 0.1761).

34 cells compared (`nett_validate_output.txt`): **all pass**.

## Not covered by this validation

Noise, sparsity and no-network-control runs, the other experimental datasets (aragonite, beaver tooth), the bone
orientation-agreement metric on full volumes, and a clean-environment install from `environment.yml`/`Dockerfile` and
a clean install from `environment.yml`/`Dockerfile` (`nett data download` itself was tested, see `DATA.md`). The unit tests
(`pytest`) and GPU tests (`pytest -m gpu`) cover the components.
