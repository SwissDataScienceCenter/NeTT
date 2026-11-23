"""The mumott baselines of the paper: SH, GK and ZH reconstructions.

All three use mumott 2.3 as released:

* forward operator: mumott's own ``SAXSProjectorCUDA`` (unmodified);
* loss: squared residual with the data weights, **no regularisation** (Laplacian weight 0);
* optimiser: scipy L-BFGS-B on the flat coefficient vector, with mumott's default tolerances
  (``ftol = 1e-3`` relative loss decrease, ``gtol = 1e-5`` projected gradient) but iteration limits raised from
  mumott's 10 / 20 to ``maxiter = 500`` / ``maxfun = 2000``, so the solver stops on convergence rather than on the budget;
* ZH (zonal harmonics, Carlsen et al. 2024): starts from the SH reconstruction, fits a symmetry axis per voxel
  (``resolution = 20``, ``filter = 'square'``) and refines zonal coefficients + axes for 300 iterations of
  mumott's ``ZHTTOptimizer``; the result is rotated back to a full SH coefficient field.

The result is always an ``(X, Y, Z, C)`` float32 array of SH coefficients up to ``ell_max`` (mumott ordering), so it
is directly comparable with :func:`nett.fit` output and the metrics of :mod:`nett.metrics`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..sh import patch_nonconvergent_basis_set


@dataclass(frozen=True)
class BaselineSettings:
    ell_max: int = 8
    laplacian_weight: float = 0.0     # 0 = unregularised (the paper); > 0 adds mumott's Laplacian regulariser
    maxiter: int = 500
    maxfun: int = 2000
    maxcor: int = 10
    ftol: float = 1e-3
    gtol: float = 1e-5
    zh_resolution: int = 20
    zh_maxiter: int = 300
    zh_filter: str = "square"
    projector: str = "cuda"           # "cuda" = SAXSProjectorCUDA as shipped; "cpu" = SAXSProjector


def _projector(geometry, s: BaselineSettings):
    from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA
    return (SAXSProjectorCUDA if s.projector == "cuda" else SAXSProjector)(geometry)


def _lbfgs(loss, s: BaselineSettings, label: str) -> Tuple[np.ndarray, Dict]:
    """scipy L-BFGS-B on ``loss`` (a mumott ``SquaredLoss``); returns the coefficient array and a convergence report."""
    from scipy.optimize import minimize

    history, shape = [], {}

    def fun(x):
        d = loss.get_loss(x, get_gradient=True)
        shape.setdefault("shape", d["gradient"].shape)
        history.append(float(d["loss"]))
        return d["loss"], d["gradient"].ravel().astype(np.float64)

    res = dict(minimize(
        fun=fun, x0=loss.initial_values.ravel(), jac=True, method="L-BFGS-B",
        options=dict(maxiter=s.maxiter, maxcor=s.maxcor, maxfun=s.maxfun, ftol=s.ftol, gtol=s.gtol, disp=False),
    ))
    x = res["x"].reshape(shape["shape"])
    report = dict(nit=int(res["nit"]), nfev=int(res["nfev"]), status=int(res["status"]), success=bool(res["success"]),
                  message=str(res["message"]), final_loss=float(res["fun"]), loss_history=history)
    print(f"[{label}] L-BFGS: nit={report['nit']} nfev={report['nfev']} loss={report['final_loss']:.6g} ({report['message']})", flush=True)
    return x, report


def _loss(train_dc, basis, s: BaselineSettings, residual_calculator=None):
    from mumott.methods.residual_calculators import GradientResidualCalculator
    from mumott.optimization.loss_functions import SquaredLoss
    from mumott.optimization.regularizers import Laplacian

    rc = residual_calculator or GradientResidualCalculator(
        data_container=train_dc, basis_set=basis, projector=_projector(train_dc.geometry, s))
    loss = SquaredLoss(residual_calculator=rc, use_weights=True)
    if s.laplacian_weight > 0:
        loss.add_regularizer("laplacian", Laplacian(), regularization_weight=s.laplacian_weight)
    return loss


def run_sh(train_dc, s: BaselineSettings = BaselineSettings()) -> Tuple[np.ndarray, Dict]:
    from mumott.methods.basis_sets import SphericalHarmonics

    basis = SphericalHarmonics(ell_max=s.ell_max, probed_coordinates=train_dc.geometry.probed_coordinates)
    patch_nonconvergent_basis_set(basis)
    x, rep = _lbfgs(_loss(train_dc, basis, s), s, "SH")
    return x.astype("float32"), rep


def run_gk(train_dc, s: BaselineSettings = BaselineSettings()) -> Tuple[np.ndarray, Dict]:
    """Gaussian kernels; the optimised kernel amplitudes are converted to SH coefficients up to ``ell_max``."""
    from mumott.methods.basis_sets import GaussianKernels

    basis = GaussianKernels(probed_coordinates=train_dc.geometry.probed_coordinates)
    patch_nonconvergent_basis_set(basis)
    x, rep = _lbfgs(_loss(train_dc, basis, s), s, "GK")
    return basis.get_spherical_harmonic_coefficients(x, ell_max=s.ell_max).astype("float32"), rep


def run_zh(train_dc, s: BaselineSettings = BaselineSettings(), sh_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """Zonal harmonics, started from the SH reconstruction (computed here unless ``sh_coeffs`` is supplied)."""
    from mumott.methods.basis_sets import SphericalHarmonics
    from mumott.methods.residual_calculators import ZHTTResidualCalculator
    from mumott.methods.utilities.fiber_fit import find_approximate_symmetry_axis, symmetric_part_along_given_direction
    from mumott.optimization.loss_functions import SquaredLoss
    from mumott.optimization.optimizers.zonal_harmonics_optimizer import ZHTTOptimizer

    if sh_coeffs is None:
        sh_coeffs, _ = run_sh(train_dc, s)
    sh = np.asarray(sh_coeffs, dtype=np.float64)

    _, theta0, phi0 = find_approximate_symmetry_axis(sh, s.ell_max, s.zh_resolution, filter=s.zh_filter)
    zonal = symmetric_part_along_given_direction(sh, theta0, phi0, ell_max=s.ell_max)

    projector = _projector(train_dc.geometry, s)
    basis = SphericalHarmonics(ell_max=s.ell_max, probed_coordinates=train_dc.geometry.probed_coordinates)
    patch_nonconvergent_basis_set(basis)
    calc = ZHTTResidualCalculator(data_container=train_dc, basis_set=basis, projector=projector)
    assert not np.isnan(np.asarray(basis.projection_matrix)).any(), "nan in ZH basis matrix"
    loss = SquaredLoss(calc, use_weights=True)
    x = np.ascontiguousarray(np.concatenate((zonal, theta0[..., None], phi0[..., None]), axis=-1)).astype(projector.dtype)

    loss0 = float(loss.get_loss(x, gradient_part="full")["loss"])
    trace, step, done = [(0, loss0)], None, 0
    while done < s.zh_maxiter:  # run in chunks of 5 to log a loss trace; the step size is fixed once, so this equals one long run
        n = min(5, s.zh_maxiter - done)
        opt = ZHTTOptimizer(loss, x, step_size_parameter=step, maxiter=n)
        step = opt._step_size_parameter
        res = opt.optimize()
        x = np.asarray(res["x"])
        done += n
        trace.append((done, float(res["loss"])))
    print(f"[ZH] loss {loss0:.6g} -> {trace[-1][1]:.6g}", flush=True)

    calc.coefficients = x
    full = np.asarray(calc.rotated_coefficients)
    assert np.isfinite(full).all(), "non-finite ZH coefficients"
    return full.astype("float32"), dict(trace=trace, step=float(step), start_loss=loss0)


def run_baseline(method: str, train_dc, s: BaselineSettings = BaselineSettings(), **kw):
    """``method`` in {"SH", "GK", "ZH"} -> ``(coefficients, report)``; adds ``report["seconds"]``."""
    t0 = time.time()
    coeffs, rep = {"SH": run_sh, "GK": run_gk, "ZH": run_zh}[method.upper()](train_dc, s, **kw)
    rep["seconds"] = time.time() - t0
    return coeffs, rep
