"""Evaluation metrics of the paper.

Ground truth available (simulated phantoms) -> :func:`ground_truth_metrics`:
    orientation_error_mean   RA-weighted mean angular error of the local symmetry axis (degrees)
    rsm_corr_mean            mean per-voxel Pearson correlation of the RSM over K = 30 directions
    ra_mae                   mean absolute error of the relative anisotropy (RA = std / mean over the K RSM-locations)
    psnr, ssim, nrmse        of the isotropic (c00) volume
    All averages are over the sample-support mask only.

No ground truth (experimental data) -> :func:`heldout_nrmse`: re-project the reconstruction onto *held-out
projections* and report ``sqrt(mean(residual^2)) / std(data)`` over valid pixels (plain NRMSE), with
:func:`heldout_report` adding the bulk-vs-bright-pixel split used to diagnose where a gap comes from.

Two reconstructions of the same sample from different data -> :func:`orientation_agreement` (trabecular bone).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from .projector import build_projector
from .sh import evaluate_real_sh, integration_matrix


# --------------------------------------------------------------------------------------------------------
# Directions and RSM sampling
# --------------------------------------------------------------------------------------------------------

def fibonacci_hemisphere(k: int, half_space: str = "y") -> np.ndarray:
    """``k`` roughly equal-area unit vectors on a hemisphere (golden-ratio spiral); pole at +y for ``half_space='y'``."""
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    i = np.arange(k, dtype=np.float64)
    phi = 2.0 * math.pi * i / golden
    z = (i + 0.5) / k
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
    if half_space == "y":  # rotate the z-pole onto +y: (x, y, z) -> (x, z, -y)
        dirs = np.column_stack([dirs[:, 0], dirs[:, 2], -dirs[:, 1]])
    return dirs


def rsm_volumes(coeffs: np.ndarray, directions: np.ndarray, ell_max: int) -> np.ndarray:
    """``(X, Y, Z, C)`` coefficients -> ``(K, X, Y, Z)`` reciprocal-space-map values at ``K`` directions."""
    B = evaluate_real_sh(torch.tensor(directions, dtype=torch.float32), ell_max).numpy()
    return np.einsum("xyzc,kc->kxyz", coeffs, B)


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)).clip(1e-8)


def _unit(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def auto_mask(volume_coeffs: np.ndarray) -> np.ndarray:
    """Otsu threshold of the c00 volume (used when no support mask is given)."""
    from skimage.filters import threshold_otsu
    c00 = volume_coeffs[..., 0]
    return c00 > threshold_otsu(c00)


def symmetry_axis(coeffs: np.ndarray, ell_max: int):
    """Per-voxel orientation ``(theta, phi)`` of the best-fitting symmetry axis (mumott's fiber fit)."""
    from mumott.methods.utilities.fiber_fit import find_approximate_symmetry_axis
    _, theta, phi = find_approximate_symmetry_axis(coeffs.astype(np.float64), ell_max=ell_max)
    return theta, phi


# --------------------------------------------------------------------------------------------------------
# Ground-truth metrics
# --------------------------------------------------------------------------------------------------------

def ground_truth_metrics(
    coeffs: np.ndarray, gt: np.ndarray, ell_max: int, mask: Optional[np.ndarray] = None, K: int = 30,
) -> Dict[str, float]:
    """All ground-truth metrics of the paper for one reconstruction. ``mask=None`` -> Otsu on the GT c00."""
    from skimage.metrics import structural_similarity

    m = auto_mask(gt) if mask is None else np.asarray(mask, dtype=bool)
    coeffs = coeffs.astype(np.float32)
    dirs = fibonacci_hemisphere(K, "y")
    gt_rsm, pred_rsm = rsm_volumes(gt, dirs, ell_max), rsm_volumes(coeffs, dirs, ell_max)
    out: Dict[str, float] = {}

    out["rsm_corr_mean"] = float(_pearson(gt_rsm, pred_rsm)[m].mean())
    gt_ra = gt_rsm.std(0) / gt_rsm.mean(0).clip(1e-8)
    pred_ra = pred_rsm.std(0) / pred_rsm.mean(0).clip(1e-8)
    out["ra_mae"] = float(np.abs(pred_ra[m] - gt_ra[m]).mean())

    th_g, ph_g = symmetry_axis(gt, ell_max)
    th_p, ph_p = symmetry_axis(coeffs, ell_max)
    dot = np.abs((_unit(th_p, ph_p) * _unit(th_g, ph_g)).sum(-1)).clip(0, 1)
    err = np.degrees(np.arccos(dot))
    w = gt_ra * m
    out["orientation_error_mean"] = float((err * w).sum() / (w.sum() + 1e-8))

    g, p = gt[..., 0], coeffs[..., 0]
    gv, pv = g[m], p[m]
    data_range = float(gv.max() - gv.min()) + 1e-8
    mse = float(((pv - gv) ** 2).mean())
    out["psnr"] = float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))
    out["nrmse"] = float(np.sqrt(mse) / (gv.std() + 1e-8))
    try:
        _, ssim_map = structural_similarity(g, p, data_range=data_range, full=True)
        out["ssim"] = float(ssim_map[m].mean())  # SSIM of the full volume, averaged inside the mask only
    except Exception:  # pragma: no cover
        out["ssim"] = float("nan")
    return out


def orientation_agreement(a: np.ndarray, b: np.ndarray, ell_max: int, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Orientation agreement (degrees) of two reconstructions of the same sample (e.g. two mounts).

    Symmetric: the error of ``a`` against ``b`` (mask and RA weights from ``b``) and of ``b`` against ``a`` are
    averaged.
    """
    e_ab = ground_truth_metrics(a, b, ell_max, mask)["orientation_error_mean"]
    e_ba = ground_truth_metrics(b, a, ell_max, mask)["orientation_error_mean"]
    return dict(mean=0.5 * (e_ab + e_ba), a_vs_b=e_ab, b_vs_a=e_ba)


# --------------------------------------------------------------------------------------------------------
# Held-out projection metrics (no ground truth needed)
# --------------------------------------------------------------------------------------------------------

def predict_projections(coeffs: np.ndarray, dc, ell_max: int, device=None) -> np.ndarray:
    """Re-project a reconstruction through the geometry of ``dc`` -> predicted detector data ``(I, J, K, M)``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = dc.geometry
    Y_int = integration_matrix(g, ell_max, device)
    projector = build_projector(g, device)
    with torch.no_grad():
        spatial = projector(torch.tensor(np.asarray(coeffs, np.float32), device=device))
        pred = torch.einsum("nhwc,nmc->nhwm", spatial, Y_int)
    return pred.cpu().numpy()


def heldout_nrmse(coeffs: np.ndarray, held_dc, ell_max: int, device=None) -> float:
    """Plain NRMSE of the re-projection on the held-out projections: ``sqrt(mean(r^2)) / std(data)`` over valid pixels."""
    pred = predict_projections(coeffs, held_dc, ell_max, device).astype(np.float64)
    data = np.asarray(held_dc.projections.data, np.float64)
    w = np.asarray(held_dc.projections.weights) > 0
    r = pred[w] - data[w]
    return float(np.sqrt((r ** 2).mean()) / (data[w].std() + 1e-8))


def heldout_report(coeffs: np.ndarray, held_dc, ell_max: int, device=None) -> Dict[str, float]:
    """Held-out NRMSE plus the bulk-vs-bright split (where in intensity does the misfit sit?).

    ``bulk_nrmse``: NRMSE over pixels below the 99th data percentile. ``peak_rel_{0.1,1}``: relative error
    ``||r|| / ||data||`` on the brightest 0.1 % / 1 % of the data pixels. ``top{0.1,1}pct_sse_share``: share of the
    total squared error carried by the largest 0.1 % / 1 % residuals.
    """
    pred = predict_projections(coeffs, held_dc, ell_max, device).astype(np.float64)
    data = np.asarray(held_dc.projections.data, np.float64)
    w = np.asarray(held_dc.projections.weights) > 0
    d, r = data[w], pred[w] - data[w]
    sse = float((r ** 2).sum())
    out = dict(nrmse=float(np.sqrt((r ** 2).mean()) / d.std()))
    ar = np.sort(r ** 2)[::-1]
    for f, nm in ((0.001, "0.1"), (0.01, "1")):
        out[f"top{nm}pct_sse_share"] = float(ar[: max(1, int(f * len(ar)))].sum() / sse)
    order = np.argsort(-d)
    for f, nm in ((0.001, "0.1"), (0.01, "1")):
        k = order[: max(1, int(f * len(d)))]
        out[f"peak_rel_{nm}"] = float(np.sqrt((r[k] ** 2).sum() / (d[k] ** 2).sum()))
        out[f"peak_sse_share_{nm}"] = float((r[k] ** 2).sum() / sse)
    lo = d < np.quantile(d, 0.99)
    out["bulk_nrmse"] = float(np.sqrt((r[lo] ** 2).mean()) / d.std())
    return out
