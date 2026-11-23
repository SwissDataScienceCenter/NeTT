# Design notes

## What "NeTT" is in this repository

The paper uses one recipe ("sp3" in the research code): a single training run of 10,500 AdamW steps, learning rate
3e-3 (MLP) and 3e-2 (feature grid), 50 warm-up steps then cosine decay, 100 projections per step, a feature grid of 8
levels × 8 features, an MLP of 2 × 64 ReLU units, a linear head with one output per SH coefficient divided by (l + 1), a
softplus on the isotropic coefficient, a zero-initialised head, an MSE loss on data scaled to unit mean, and two
coarse-to-fine reveals (spatial: first 50 %; angular, stochastic per voxel: first 70 %). The research code carried many
other options (a second training phase, regularisers, Huber loss, multi-q fields, …); none of them is part of the
paper and none is in this repository. `ell_max` is the only parameter of the method a user sets.

## Forward operators

* NeTT: ASTRA toolbox (`parallel3d_vec` geometry derived from the mumott `Geometry`), SH contraction with a precomputed
  integration matrix (flat-plane Simpson quadrature for SAXS, mumott's Ewald-curvature-corrected basis for WAXS).
* Baselines: mumott's `SAXSProjectorCUDA` as released (mumott 2.3), i.e. including its known low-edge bilinear
  interpolation behaviour. A corrected ("padded") projector exists in the research code; the paper's baselines do
  not use it.
* Evaluation of held-out NRMSE and cross-mount NRMSE re-projects every method's SH volume with the ASTRA operator and
  the same integration matrix, so all methods are scored with one operator.

## Reproducibility

* `nett.fit` is deterministic given the seed up to GPU non-determinism (atomic adds in the ASTRA kernels). Using the
  same seed, the standalone code and the research code produce bit-identical forward/adjoint projections, loss curves
  and coefficients over 150 training steps (see `tests/gpu_equivalence.py`, `docs/VALIDATION.md`).
* The research-code runs reported as "seed 0" were unseeded; here seed 0 is a fixed seed.
* mumott baselines: L-BFGS (scipy) on the GPU is not bit-reproducible; `nett validate` allows 2 %.
