"""Execute one :class:`~nett.experiments.plan.Run`: load data, reconstruct, evaluate, store.

Output (``$NETT_RESULTS``, default ``./results``)::

    <id>.npy     reconstruction, (X, Y, Z, C) float32 SH coefficients (physical units)
    <id>.json    run description, NeTT configuration / baseline settings, evaluation metrics, timings, versions
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .. import __version__
from ..config import NeTTConfig
from ..data import registry, splits
from .plan import HOLDOUT_FRACTION, HOLDOUT_SEED, Run, by_id


def results_root() -> Path:
    return Path(os.environ.get("NETT_RESULTS", "results")).expanduser().resolve()


def result_paths(run_id: str, root: Optional[Path] = None):
    root = root or results_root()
    return root / f"{run_id}.json", root / f"{run_id}.npy"


def is_done(run: Run, root: Optional[Path] = None) -> bool:
    j, n = result_paths(run.id, root)
    return j.exists() and n.exists()


def load_result(run_id: str, root: Optional[Path] = None):
    j, n = result_paths(run_id, root)
    return json.loads(j.read_text()), np.load(n)


def prepare_data(run: Run):
    """-> ``(dataset, train_dc, held_dc, full_dc)`` for a run (noise level, pruning and hold-out applied)."""
    ds = registry.get(run.dataset, run.poisson_rate)
    full = ds.dc()
    if run.prune:
        full = splits.pruned(full, *run.prune)
    mode = run.holdout
    train, held = splits.split(full, mode, HOLDOUT_FRACTION, HOLDOUT_SEED)
    return ds, train, held, full


def _versions() -> dict:
    import torch
    out = dict(nett=__version__, python=platform.python_version(), torch=torch.__version__, host=platform.node())
    for mod in ("mumott", "astra", "numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def evaluate(run: Run, coeffs: np.ndarray, ds, held_dc) -> dict:
    from .. import metrics as M
    ell = run.ell_max or ds.ell_max
    if run.evaluation == "ground_truth":
        return M.ground_truth_metrics(coeffs, ds.ground_truth(), ell, ds.volume_mask())
    if run.evaluation == "heldout":
        return M.heldout_report(coeffs, held_dc, ell)
    if run.evaluation == "cross_mount":
        other = registry.get("bone-remount" if run.dataset == "bone-main" else "bone-main").dc()
        return dict(cross_mount_nrmse=M.heldout_nrmse(coeffs, other, ell))
    return {}


def execute(run: Run, root: Optional[Path] = None, force: bool = False, verbose: bool = True) -> dict:
    """Run one experiment cell. Skips (and returns the stored result) if it already exists and ``force`` is false."""
    root = root or results_root()
    json_path, npy_path = result_paths(run.id, root)
    if is_done(run, root) and not force:
        return json.loads(json_path.read_text())
    json_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds, train, held, _ = prepare_data(run)
    record = dict(run=run.__dict__ | {"prune": list(run.prune) if run.prune else None, "overrides": dict(run.overrides)},
                  n_train=len(train.projections), n_held=len(held.projections), ell_max=run.ell_max or ds.ell_max)

    if run.method == "NeTT":
        from ..train import fit
        cfg = NeTTConfig.from_overrides(run.overrides_dict(), NeTTConfig(ell_max=run.ell_max or ds.ell_max, seed=run.seed))
        monitor = held if run.holdout != "none" else None   # held-out NRMSE is only *logged* during training
        res = fit(train, cfg, held_dc=monitor, verbose=verbose, cache_dir=str(root / ".cache"),
                  checkpoint_path=str(root / ".ckpt" / f"{run.id}.pt"))
        coeffs = res.coefficients
        (root / ".ckpt" / f"{run.id}.pt").unlink(missing_ok=True)   # finished: a --force rerun must start from scratch
        record.update(config=cfg.to_dict(), train_seconds=res.seconds, target_scale=res.target_scale,
                      loss_tail=res.losses[-20:], holdout_curve=res.holdout_curve)
    else:
        from ..baselines import BaselineSettings, run_baseline
        s = BaselineSettings(ell_max=run.ell_max or ds.ell_max)
        kw = {}
        if run.method == "ZH" and run.depends_on:
            dep_json, dep_npy = result_paths(run.depends_on, root)
            if dep_npy.exists():
                kw["sh_coeffs"] = np.load(dep_npy)      # start ZH from the stored SH reconstruction of the same data
        coeffs, rep = run_baseline(run.method, train, s, **kw)
        rep.pop("loss_history", None)
        record.update(settings=s.__dict__, report=rep)

    record["metrics"] = evaluate(run, coeffs, ds, held)
    record.update(wall_seconds=time.time() - t0, versions=_versions())
    np.save(npy_path, np.asarray(coeffs, np.float32))
    json_path.write_text(json.dumps(record, indent=1, default=float))
    if verbose:
        print(f"[done] {run.id}: " + ", ".join(f"{k}={v:.4g}" for k, v in record["metrics"].items() if isinstance(v, float)), flush=True)
    return record


def execute_id(run_id: str, **kw) -> dict:
    return execute(by_id(run_id), **kw)
