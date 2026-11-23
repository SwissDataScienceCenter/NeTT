"""Real spherical harmonics and the SH -> detector-segment integration matrix.

The NeTT forward model is linear in the per-voxel SH coefficients ``c``:

    projection(i, j, k, m) = sum_c  [ X-ray transform of c ](i, j, k, c)  *  Y_int(i, m, c)

``Y_int[i, m, c]`` is the integral of the real spherical harmonic ``c`` along the
great-circle arc probed by detector segment ``m`` of projection ``i``.  It depends only
on the geometry, so it is computed once per dataset:

* SAXS: flat-plane Simpson quadrature (:func:`precompute_Y_int`).
* WAXS (non-zero ``two_theta``): mumott's own Ewald-curvature-corrected basis set
  (:func:`mumott_projection_matrix`).

Coefficient ordering: even degrees ``l = 0, 2, ..., ell_max``; within each degree
``m = -l..l``; this is mumott's ordering.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def lm_list(ell_max: int) -> List[Tuple[int, int]]:
    """(l, m) pairs, even l only, ordered as mumott orders its SH coefficients."""
    return [(l, m) for l in range(0, ell_max + 1, 2) for m in range(-l, l + 1)]


def num_coeffs(ell_max: int) -> int:
    return (ell_max // 2 + 1) * (ell_max + 1)


def associated_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Associated Legendre polynomial P_l^m(x) (Condon-Shortley phase included), stable recurrence."""
    if m < 0 or l < m:
        raise ValueError("Require 0 <= m <= l in associated_legendre")
    P_mm = torch.ones_like(x)
    if m > 0:
        root = torch.sqrt(torch.relu(1.0 - x * x))
        for k in range(1, m + 1):
            P_mm = -(2 * k - 1) * root * P_mm
    if l == m:
        return P_mm
    P_m1m = (2 * m + 1) * x * P_mm
    if l == m + 1:
        return P_m1m
    P_n2, P_n1 = P_mm, P_m1m
    for n in range(m + 2, l + 1):
        P_n = ((2 * n - 1) * x * P_n1 - (n + m - 1) * P_n2) / (n - m)
        P_n2, P_n1 = P_n1, P_n
    return P_n1


def _real_sh(l: int, m: int, cos_theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Real SH Y_l^m (4*pi-normalised as in mumott) at polar cosine / azimuth."""
    ma = abs(m)
    two_minus_delta = 1 if ma == 0 else 2
    factor = math.sqrt(two_minus_delta * (2 * l + 1) * (math.factorial(l - ma) / float(math.factorial(l + ma))))
    P = associated_legendre(l, ma, cos_theta)
    cs = (-1) ** ma  # cancel the Condon-Shortley phase carried by associated_legendre
    if m > 0:
        return cs * factor * P * torch.cos(ma * phi)
    if m < 0:
        return cs * factor * P * torch.sin(ma * phi)
    return factor * P


def evaluate_real_sh(directions: torch.Tensor, ell_max: int) -> torch.Tensor:
    """Basis matrix ``B[k, c] = Y_c(direction_k)`` for ``(K, 3)`` directions -> ``(K, C)``."""
    d = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_theta = d[:, 2].clamp(-1.0, 1.0)
    phi = torch.atan2(d[:, 1], d[:, 0])
    return torch.stack([_real_sh(l, m, cos_theta, phi) for l, m in lm_list(ell_max)], dim=-1)


def _quadrature_weights(n: int, dtype, device, mode: str = "simpson") -> torch.Tensor:
    """Normalised quadrature weights (sum to 1) for ``n`` samples along a detector segment."""
    if mode == "simpson" and n % 2 == 1 and n >= 3:
        w = [1.0 if i in (0, n - 1) else (4.0 if i % 2 == 1 else 2.0) for i in range(n)]
    elif mode == "trapezoid" and n >= 2:
        w = [1.0] + [2.0] * (n - 2) + [1.0]
    else:
        w = [1.0] * n
    s = sum(w)
    return torch.tensor([x / s for x in w], dtype=dtype, device=device)


def precompute_Y_int(probed_coordinates, ell_max: int, device, dtype=torch.float32, mode: str = "simpson") -> torch.Tensor:
    """``(N, M, C)`` SH integration matrix with the flat-plane (small-angle) quadrature."""
    _, theta_np, phi_np = probed_coordinates.to_spherical
    with torch.no_grad():
        theta = torch.as_tensor(theta_np, dtype=dtype, device=device)
        phi = torch.as_tensor(phi_np, dtype=dtype, device=device)
        N, M, I = theta.shape
        x = torch.cos(theta)
        w = _quadrature_weights(I, dtype, device, mode).view(1, 1, -1)
        out = torch.zeros(N, M, num_coeffs(ell_max), dtype=dtype, device=device)
        for idx, (l, m) in enumerate(lm_list(ell_max)):
            out[..., idx] = (_real_sh(l, m, x, phi) * w).sum(dim=2)
    return out


def mumott_projection_matrix(
    probed_coordinates, ell_max: int, device, dtype=torch.float32, cache_path: Union[str, Path, None] = None
) -> torch.Tensor:
    """``(N, M, C)`` matrix from mumott's own ``SphericalHarmonics`` basis (Ewald-curvature correct, for WAXS).

    mumott's adaptive quadrature is CPU-only and slow, so the result is cached on disk if ``cache_path`` is
    given. A small fraction of entries can come back ``nan`` (an ``arccos`` domain error inside mumott); those are
    filled with the flat-plane quadrature rather than being allowed to poison the training.
    """
    if cache_path is not None and Path(cache_path).exists():
        return torch.as_tensor(np.load(cache_path), dtype=dtype, device=device)
    from mumott.methods.basis_sets import SphericalHarmonics

    pm = np.asarray(SphericalHarmonics(ell_max=ell_max, probed_coordinates=probed_coordinates).projection_matrix)
    bad = np.isnan(pm)
    if bad.any():
        fallback = precompute_Y_int(probed_coordinates, ell_max, torch.device("cpu"), dtype=torch.float64).numpy()
        pm = np.where(bad, fallback, pm)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pm)
    return torch.as_tensor(pm, dtype=dtype, device=device)


def integration_matrix(geometry, ell_max: int, device, cache_dir: Optional[Union[str, Path]] = None) -> torch.Tensor:
    """Y_int for a whole geometry: flat-plane for SAXS, mumott's curvature-correct basis for WAXS."""
    if bool(np.any(np.asarray(geometry.two_theta) != 0)):
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f"Y_int_ell{ell_max}_{hash(geometry) & 0xFFFFFFFF:08x}.npy"
        return mumott_projection_matrix(geometry.probed_coordinates, ell_max, device, cache_path=path)
    return precompute_Y_int(geometry.probed_coordinates, ell_max, device)


def patch_nonconvergent_basis_set(basis) -> None:
    """Fill ``nan`` entries of a mumott basis set's projection matrix in place (used by the baselines).

    The ``nan`` arises from an ``arccos`` domain error in mumott's segment-length computation, which the
    ``midpoint`` integration mode never evaluates; that mode is therefore used for the affected entries.
    """
    pm = np.asarray(basis.projection_matrix)
    bad = np.isnan(pm)
    if not bad.any():
        return
    kw = dict(
        probed_coordinates=basis.probed_coordinates,
        enforce_friedel_symmetry=getattr(basis, "_enforce_friedel_symmetry", True),
        integration_mode="midpoint",
    )
    for attr in ("ell_max", "grid_scale", "kernel_scale_parameter"):
        if hasattr(basis, attr):
            kw[attr] = getattr(basis, attr)
    fallback = np.asarray(type(basis)(**kw).projection_matrix)
    basis._projection_matrix = np.where(bad, fallback, pm)
    basis._probed_coordinates_hash = hash(basis.probed_coordinates)
