"""Per-scan, self-supervised NeTT reconstruction (the single-phase recipe of the paper).

One field is trained against the measured projections of one scan, with no external data or pre-training:

    coefficients(x, y, z)  --X-ray transform-->  spatial projections  --contract with Y_int-->  predicted pixels

The loss is the masked mean squared error between predicted and measured projections (data scaled to O(1)).
Training uses AdamW with two learning rates (feature grid fast, MLP slow), linear warm-up + cosine decay,
projection mini-batches, and the two coarse-to-fine reveals of :mod:`nett.schedule`.

    from nett import NeTTConfig, fit
    result = fit(train_dc, NeTTConfig(ell_max=8))
    result.coefficients      # (X, Y, Z, C) float32 array of SH coefficients (mumott ordering)
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import NeTTConfig
from .field import build_field
from .projector import build_projector
from .schedule import Annealer
from .sh import integration_matrix


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    """LR factor: linear warm-up over ``warmup`` steps, then cosine decay to ~0."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def subset_geometry(geometry, idx: np.ndarray):
    """Deep copy of a mumott ``Geometry`` restricted to projections ``idx``."""
    g = copy.deepcopy(geometry)
    keep = set(int(i) for i in idx)
    for j in range(len(geometry) - 1, -1, -1):
        if j not in keep:
            del g[j]
    return g


def _chunk_pool(geometry, perm, batch_size, target, mask, Y_int, device) -> List[dict]:
    """Partition projections into chunks (one projector per chunk, built once and cycled)."""
    pool = []
    for start in range(0, len(perm), batch_size):
        idx = np.sort(perm[start:start + batch_size])
        pool.append(dict(
            projector=build_projector(subset_geometry(geometry, idx), device),
            Y_int=Y_int[idx], target=target[idx], mask=mask[idx],
        ))
    return pool


def predict(field_coeffs: torch.Tensor, projector, Y_int: torch.Tensor) -> torch.Tensor:
    """Forward model: coefficients ``(X, Y, Z, C)`` -> predicted detector pixels ``(I, J, K, M)``."""
    return torch.einsum("nijc,nmc->nijm", projector(field_coeffs), Y_int)


@dataclass
class FitResult:
    coefficients: np.ndarray                    # (X, Y, Z, C), physical units
    config: NeTTConfig
    losses: List[float] = field(default_factory=list)
    holdout_curve: List[Tuple[int, float]] = field(default_factory=list)   # (step, held-out NRMSE)
    seconds: float = 0.0
    target_scale: float = 1.0
    model: Optional[torch.nn.Module] = None


def fit(
    dc,
    cfg: NeTTConfig = NeTTConfig(),
    held_dc=None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 500,
    cache_dir: Optional[str] = None,
) -> FitResult:
    """Reconstruct the SH-coefficient volume of the scan ``dc`` with the NeTT recipe.

    ``held_dc`` (optional) only *monitors* held-out NRMSE every ``cfg.eval_every`` steps; it never influences
    training (no early stopping, no checkpoint selection). ``checkpoint_path`` makes the run resumable.
    ``cache_dir`` stores the (WAXS-only, slow) curvature-corrected integration matrix.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    geometry = dc.geometry
    volume_shape = tuple(int(s) for s in geometry.volume_shape)
    target = torch.tensor(dc.projections.data, device=device, dtype=torch.float32)
    mask = torch.tensor(dc.projections.weights, device=device, dtype=torch.float32).bool()
    n_proj = target.shape[0]

    # The forward model is linear in the coefficients, so training on data scaled to O(1) only rescales the
    # solution; the scale is baked back into the field's output at the end.
    target_scale = 1.0
    if cfg.normalize_target:
        target_scale = float(target[mask].abs().mean().clamp_min(1e-8))
        target = target / target_scale

    held = None
    if held_dc is not None:
        ht = torch.tensor(held_dc.projections.data, device=device, dtype=torch.float32) / target_scale
        held = dict(
            projector=build_projector(held_dc.geometry, device),
            Y_int=integration_matrix(held_dc.geometry, cfg.ell_max, device, cache_dir),
            target=ht, mask=torch.tensor(held_dc.projections.weights, device=device, dtype=torch.float32).bool(),
        )

    model = build_field(
        volume_shape, cfg.ell_max, field_type=cfg.field_type,
        n_levels=cfg.n_levels, n_features_per_level=cfg.n_features_per_level, base_resolution=cfg.base_resolution,
        max_resolution=cfg.max_resolution, table_size=cfg.table_size, hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers, head_init_std=cfg.head_init_std, per_l_scale_power=cfg.per_l_scale_power,
    ).to(device)

    start_step, ckpt = 0, None
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt["step"]) + 1
        if verbose:
            print(f"[resume] {checkpoint_path} at step {ckpt['step']}")

    Y_int = integration_matrix(geometry, cfg.ell_max, device, cache_dir)
    pool = _chunk_pool(geometry, np.random.permutation(n_proj), cfg.batch_size, target, mask, Y_int, device)

    # Start the isotropic component at the right scale: c00 is set so the mean prediction matches the mean data.
    if start_step == 0:
        with torch.no_grad():
            ch = pool[0]
            pred_mean = float(predict(model(), ch["projector"], ch["Y_int"])[ch["mask"]].mean().clamp_min(1e-8))
            model.set_c00_init(max(float(target[mask].mean()) / pred_mean, 1e-6))

    optimizer = torch.optim.AdamW(
        [dict(params=model.grid_parameters(), lr=cfg.lr * cfg.grid_lr_multiplier),
         dict(params=model.other_parameters(), lr=cfg.lr)],
        weight_decay=0.0,
    )
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_warmup(s, cfg.n_iterations, cfg.warmup_steps))
    for _ in range(start_step):
        scheduler.step()

    annealer = Annealer(
        n_levels=model.encoding.n_levels, lm_list=model.lm_list, total_steps=cfg.n_iterations,
        spatial_on=True, angular_on=True, spatial_frac=cfg.spatial_frac, angular_frac=cfg.angular_frac,
        stochastic_angular=cfg.stochastic_angular,
    )
    n_voxels = int(np.prod(volume_shape))
    if verbose:
        print(f"NeTT: volume={volume_shape} C={model.num_coeffs} projections={n_proj} chunks={len(pool)} | {model.encoding.describe()}")

    losses: List[float] = []
    curve: List[Tuple[int, float]] = []
    t0 = time.time()
    try:
        from tqdm import tqdm
        steps = tqdm(range(start_step, cfg.n_iterations), initial=start_step, total=cfg.n_iterations, disable=not verbose,
                     mininterval=0.2 if sys.stderr.isatty() else 120.0)   # quiet logs in batch jobs
    except ImportError:  # pragma: no cover
        steps = range(start_step, cfg.n_iterations)

    for step in steps:
        if step > 0 and cfg.reshuffle_every and step % cfg.reshuffle_every == 0:
            pool = _chunk_pool(geometry, np.random.permutation(n_proj), cfg.batch_size, target, mask, Y_int, device)
        ch = pool[step % len(pool)]
        level_w, ell_mask = annealer.masks(step, model.num_coeffs, n_voxels, device)
        level_w = level_w.to(device) if level_w is not None else None
        ell_mask = ell_mask.to(device) if ell_mask is not None else None

        optimizer.zero_grad(set_to_none=True)
        pred = predict(model(level_weights=level_w, ell_mask=ell_mask), ch["projector"], ch["Y_int"])
        m = ch["mask"]
        loss = torch.nn.functional.mse_loss(pred[m], ch["target"][m])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if checkpoint_path and (step % checkpoint_every == 0 or step == cfg.n_iterations - 1):
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, f"{checkpoint_path}.tmp")
            os.replace(f"{checkpoint_path}.tmp", checkpoint_path)

        if held is not None and (step % cfg.eval_every == 0 or step == cfg.n_iterations - 1):
            with torch.no_grad():
                hp = predict(model(), held["projector"], held["Y_int"])
                hm = held["mask"]
                d = hp[hm] - held["target"][hm]
                curve.append((step, float(torch.sqrt((d ** 2).mean()) / (held["target"][hm].std() + 1e-8))))
        losses.append(float(loss.detach()))
        if verbose and hasattr(steps, "set_postfix") and step % 10 == 0:
            steps.set_postfix(loss=f"{losses[-1]:.3e}", **({"held": f"{curve[-1][1]:.4f}"} if curve else {}))

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0

    if cfg.normalize_target:
        model.set_output_scale(target_scale)  # the field now natively emits physical units
    with torch.no_grad():
        coefficients = model().detach().cpu().numpy().astype(np.float32)
    return FitResult(coefficients, cfg, losses, curve, seconds, target_scale, model)
