"""Figures of the paper from the stored results (``nett figures --results DIR --out DIR [--only NAME ...]``).

======================  ============================  =========================================================================
``--only`` name         output                        shows
======================  ============================  =========================================================================
``noise``               ``fig_noise.pdf``             orientation error vs Poisson rate (10, 31.6, 100), one panel per phantom
``sparsity``            ``fig_sparsity.pdf``          metric vs projections kept, RANDOM pruning (3 realisations: line = mean,
                                                      band = min-max), top row orientation error, bottom row RSM correlation
``sparsity_maxmin``     ``fig_sparsity_maxmin.pdf``   the same for Max-Min pruning (one order, no band)
``robustness``          ``fig_robustness.pdf``        top row: orientation error vs projections kept (random pruning), bottom
                                                      row: orientation error vs Poisson rate; one column per phantom
``robustness_corr``     ``fig_robustness_corr.pdf``   the same with the RSM correlation (SI)
``wedge``               ``fig_<name>_wedge.pdf`` (9)  qualitative comparison, see :func:`wedge_figure`; single figures can be
                                                      selected by file stem, e.g. ``--only fig_mammoth_wedge``
``agreement``           ``bone_agreement.json``       orientation agreement of the bone reconstructions (:func:`bone_agreement`)
======================  ============================  =========================================================================

Colours: SH grey, GK green, NeTT dark blue (same in every figure). Missing results are reported and skipped; a figure is
drawn with whatever has finished, and not written at all if nothing it needs exists.
"""
from __future__ import annotations

import itertools
import json
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..experiments.plan import (BASELINES, NOISE_RATES, PHANTOMS, PRUNE_EXTRA_LEVEL, PRUNE_EXTRA_LEVEL_PHANTOMS, PRUNE_LEVELS,
                                SEEDS, plan)
from .tables import Results

STYLE = {"SH": ("#8c8c8c", "o"), "GK": ("#2ca02c", "s"), "NeTT": ("#0b3d7a", "^")}
LABEL = {"SH": "SH", "GK": "GK", "NeTT": "NeTT (ours)"}
ORI = ("orientation_error_mean", r"orient. err. ($^\circ$)")
COR = ("rsm_corr_mean", "RSM corr.")
RATES = [("1.0e+01", 10.0), ("3.2e+01", 31.6), ("1.0e+02", 100.0)]      # 1.0e+02 = the clean data of table1
N_PROJ = 417                                                              # projections of every phantom (pruning levels are % of this)
RANDOM_STRATEGIES = ("random", "random1", "random2")


class Ctx:
    """The stored results plus bookkeeping of what was asked for but is not there (reported at the end of each figure)."""

    def __init__(self, results=None):
        self.R = Results(results)
        self.root = self.R.root
        self.planned = {r.id for r in plan()}
        self.missing: List[str] = []
        self.hits = 0

    def get(self, run_id: str, key: str) -> Optional[float]:
        v = self.R.metric(run_id, key) if run_id in self.R.rec else None
        if v is None:
            if run_id in self.planned:
                self.missing.append(run_id)
        else:
            self.hits += 1
        return v

    def mean(self, ids: Iterable[str], key: str, flag_missing: bool = True) -> Optional[float]:
        """Mean over the ids that exist (None if none does)."""
        v = [self.R.metric(i, key) if i in self.R.rec else None for i in ids]
        v = [x for x in v if x is not None]
        if not v:
            if flag_missing:
                self.missing += [i for i in ids if i in self.planned]
            return None
        self.hits += len(v)
        return float(np.mean(v))

    def volume(self, run_id: str) -> Optional[np.ndarray]:
        p = self.root / f"{run_id}.npy"
        if not p.exists():
            if run_id in self.planned:
                self.missing.append(run_id)
            return None
        self.hits += 1
        return np.load(p)


def _style():
    plt.rcParams.update({"font.size": 8})


def kept(level: int, n: int = N_PROJ) -> int:
    """Number of projections left at pruning level ``level`` %: same rule as :func:`nett.data.splits.pruned`."""
    return n - round(round((100 - level) / 10.0) * 10.0 / 100.0 * n)


def _tag(method: str, seed: int = 0) -> str:
    return f"NeTT_s{seed}" if method == "NeTT" else method


def _levels(ds: str, strat: str) -> List[int]:
    lv = list(PRUNE_LEVELS)
    if strat in PRUNE_EXTRA_LEVEL and ds in PRUNE_EXTRA_LEVEL_PHANTOMS:
        lv.append(PRUNE_EXTRA_LEVEL[strat])
    return lv


# --------------------------------------------------------------------------------------------------------
# Line panels
# --------------------------------------------------------------------------------------------------------

def _sparsity_series(C: Ctx, ds: str, protocol: str, method: str, key: str):
    """-> (kept projections, mean, min, max) over the pruning levels that exist; mean over the random realisations."""
    strats = RANDOM_STRATEGIES if protocol == "random" else ("maxmin",)
    levels = sorted(set(itertools.chain.from_iterable(_levels(ds, s) for s in strats)))
    xs, mean, lo, hi = [], [], [], []
    for lv in levels:
        vals = [C.get(f"sparsity/{ds}/{s}_L{lv}_{_tag(method)}", key) for s in strats if lv in _levels(ds, s)]
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(kept(lv)); mean.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
    return xs, mean, lo, hi


def sparsity_panel(ax, C: Ctx, ds: str, key: str, protocol: str = "random", lines: bool = False):
    """Metric vs projections kept. ``protocol``: ``"random"`` (mean + min-max band over the 3 realisations) | ``"maxmin"``.
    ``lines`` (orientation error only): dashed = the better baseline given ALL projections, dotted = the fewest projections
    with which NeTT is already better than it (annotated)."""
    series = {}
    for m in ("SH", "GK", "NeTT"):
        xs, mean, lo, hi = series[m] = _sparsity_series(C, ds, protocol, m, key)
        c, mk = STYLE[m]
        ax.plot(xs, mean, color=c, marker=mk, ms=3, lw=1.2, label=LABEL[m])
        if protocol == "random" and xs:
            ax.fill_between(xs, lo, hi, color=c, alpha=0.2, lw=0)
    if lines and protocol == "random":
        full = [dict(zip(*series[m][:2])).get(kept(100)) for m in ("SH", "GK")]
        full = [v for v in full if v is not None]
        if full:
            best = min(full)
            ax.axhline(best, color="0.35", ls="--", lw=0.9)
            xs, mean = series["NeTT"][:2]
            below = [x for x, v in zip(xs, mean) if v < best]
            if below:
                xc = min(below)
                c = STYLE["NeTT"][0]
                ax.axvline(xc, color=c, ls=":", lw=1.0)
                ax.annotate(str(xc), (xc, ax.get_ylim()[1]), xytext=(2, -9), textcoords="offset points", color=c, fontsize=7)
    ax.grid(alpha=0.25)


def _noise_value(C: Ctx, ds: str, method: str, rate: str, key: str) -> Optional[float]:
    if rate == "1.0e+02":            # clean data = the table1 runs (NeTT: mean over the seeds)
        ids = [f"table1/{ds}/{_tag(method, s)}" for s in (SEEDS if method == "NeTT" else (0,))]
    else:
        ids = [f"noise/{ds}/r{rate}_{_tag(method)}"]
    return C.mean(ids, key)


def noise_panel(ax, C: Ctx, ds: str, key: str):
    """Metric vs Poisson rate (higher = less noise), log x axis."""
    for m in ("SH", "GK", "NeTT"):
        pts = [(r, _noise_value(C, ds, m, rate, key)) for rate, r in RATES]
        pts = [(r, v) for r, v in pts if v is not None]
        c, mk = STYLE[m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, ms=3, lw=1.2, label=LABEL[m])
    ax.set_xscale("log"); ax.set_xticks([10, 31.6, 100]); ax.set_xticklabels(["10", "32", "100"]); ax.minorticks_off(); ax.grid(alpha=0.25)


def _legend(fig, ax0, dashed: bool, ncol: int = 4):
    if dashed:
        ax0.plot([], [], color="0.35", ls="--", lw=0.9, label="best baseline, all projections")
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=7)


def _save(fig, C: Ctx, out: Path, name: str) -> Optional[Path]:
    """Write the figure unless none of the results it needs exists (then say so)."""
    if C.hits == 0:
        plt.close(fig)
        print(f"[skip] {name}: no results" + _missing_msg(C))
        return None
    if C.missing:
        print(f"[partial] {name}" + _missing_msg(C))
    path = Path(out) / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")
    return path


def fig_noise(C: Ctx, out: Path) -> Optional[Path]:
    """Orientation error of NeTT/SH/GK on the three phantoms at Poisson rates 10, 31.6 and 100 (= noise-free table1 data)."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.3))
    for ax, ds in zip(axes, PHANTOMS):
        noise_panel(ax, C, ds, ORI[0]); ax.set_title(ds); ax.set_xlabel("Poisson rate (higher = less noise)")
    axes[0].set_ylabel(ORI[1]); axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    return _save(fig, C, out, "fig_noise.pdf")


def fig_sparsity(C: Ctx, out: Path, protocol: str = "random") -> Optional[Path]:
    """Two rows (orientation error, RSM correlation) x one column per phantom, against the number of projections kept."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.1), sharex=True)
    for j, ds in enumerate(PHANTOMS):
        for i, (key, yl) in enumerate((ORI, COR)):
            ax = axes[i, j]
            sparsity_panel(ax, C, ds, key, protocol, lines=(i == 0))
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(yl)
            if i == 1:
                ax.set_xlabel("projections kept")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, axes[0, 0], dashed=protocol == "random")
    return _save(fig, C, out, "fig_sparsity.pdf" if protocol == "random" else "fig_sparsity_maxmin.pdf")


def fig_robustness(C: Ctx, out: Path, metric=ORI, fname: str = "fig_robustness.pdf") -> Optional[Path]:
    """Top row: ``metric`` vs projections kept (random pruning, 3 realisations); bottom row: vs Poisson rate."""
    _style()
    key, yl = metric
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
    for j, ds in enumerate(PHANTOMS):
        sparsity_panel(axes[0, j], C, ds, key, "random", lines=(key == ORI[0]))
        axes[0, j].set_title(ds); axes[0, j].set_xlabel("projections kept")
        noise_panel(axes[1, j], C, ds, key); axes[1, j].set_xlabel("Poisson rate (higher = less noise)")
        if j == 0:
            axes[0, j].set_ylabel(yl); axes[1, j].set_ylabel(yl)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, axes[0, 0], dashed=key == ORI[0])
    return _save(fig, C, out, fname)


# --------------------------------------------------------------------------------------------------------
# Qualitative comparison: RSM-location slice + pole figure of a marked voxel
# --------------------------------------------------------------------------------------------------------

K_DIRECTIONS = 30
ALPHA_DEG = {"nielsen-m": 45.0, "nielsen-t": 45.0, "nielsen-mammoth": 45.0, "bone-main": 45.0, "steel-wire": 45.0,
             "aragonite": 44.0, "beaver-tooth": 40.0, "narwhal-tusk": 41.0}     # tilt limit (deg); selects the default RSM-location

_PLANES = {   # key: (slice, projection (sum over the out-of-plane axis), title, [voxel -> (fixed, column, row)])
    "yz": (lambda v, x, y, z: v[x, :, :].T, lambda v, x, y, z: v.sum(0).T, "YZ  (x={x})", "YZ  (proj. over X)", lambda a, b, c: (a, b, c)),
    "xz": (lambda v, x, y, z: v[:, y, :].T, lambda v, x, y, z: v.sum(1).T, "XZ  (y={y})", "XZ  (proj. over Y)", lambda a, b, c: (b, a, c)),
    "xy": (lambda v, x, y, z: v[:, :, z].T, lambda v, x, y, z: v.sum(2).T, "XY  (z={z})", "XY  (proj. over Z)", lambda a, b, c: (c, a, b)),
}


def missing_arc_deg(direction: np.ndarray, alpha_deg: float, axis=(0.0, 0.0, 1.0)) -> float:
    """Unmeasured arc (degrees) of the great circle orthogonal to an RSM-location, for a tilt limit ``alpha_deg``
    (closed form: the circle is measured where ``|b . axis| <= cos(alpha)``)."""
    g = np.asarray(axis, float) / np.linalg.norm(axis)
    y = np.asarray(direction, float) / np.linalg.norm(direction)
    sin_t = float(np.sqrt(max(0.0, 1.0 - float(y @ g) ** 2)))
    ca = np.cos(np.radians(alpha_deg))
    return 0.0 if sin_t <= ca else float(np.degrees(2 * np.pi - 4 * np.arcsin(min(1.0, ca / sin_t))))


def _basis(directions: np.ndarray, ell_max: int, n_coeff: int) -> np.ndarray:
    import torch
    from ..sh import evaluate_real_sh
    B = evaluate_real_sh(torch.tensor(np.asarray(directions, np.float32)), ell_max).numpy()
    return B[:, :n_coeff] if B.shape[1] >= n_coeff else np.pad(B, ((0, 0), (0, n_coeff - B.shape[1])))


def wedge_figure(volumes: "Dict[str, np.ndarray]", ell_max: int, out: Path, dataset: str = "", k: Optional[int] = None,
                 plane: str = "yz", x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None,
                 voxel=True, projection: bool = False, alpha_deg: Optional[float] = None, ref: Optional[str] = None,
                 titles: Optional[Dict[str, str]] = None, fontsize: float = 10, cmap: str = "inferno") -> Optional[Path]:
    """Qualitative comparison of reconstructions (columns) at one RSM-location.

    ``volumes``: ``{label: (X, Y, Z, C) SH-coefficient volume}``, in column order (ground truth / anchor first, then NeTT,
    SH, GK, ZH). Top row: the scalar volume ``I_q(x, y, z) = sum_c B_c(q) c_c(x, y, z)`` at the RSM-location ``q`` = the
    ``k``-th of 30 Fibonacci directions (default: the worst-covered one for ``alpha_deg``), shown as a slice through ``x/y/z``
    in ``plane`` (or summed over the out-of-plane axis with ``projection``). Bottom row (``voxel``: ``(x, y, z)`` or True for the
    slice position): the pole figure of that voxel, i.e. its RSM over the z >= 0 hemisphere. All panels share one colour scale,
    set by the percentiles (0.5 / 99.5) of the reference column's slice; the ring marks the voxel on the slice and
    ``q`` on the pole figure.
    """
    from ..metrics import fibonacci_hemisphere
    vols = OrderedDict((lab, np.asarray(v, np.float32)) for lab, v in volumes.items())
    if not vols:
        return None
    labels = list(vols)
    ref = ref if ref in vols else labels[0]
    titles = titles or {}
    shape = vols[labels[0]].shape[:3]
    cl = lambda v, n: int(min(max(int(v), 0), n - 1))
    x, y, z = (cl(shape[i] // 2 if c is None else c, shape[i]) for i, c in enumerate((x, y, z)))
    want_voxel = bool(voxel) or isinstance(voxel, (list, tuple))
    if want_voxel:
        vx, vy, vz = (cl(c, shape[i]) for i, c in enumerate(voxel)) if isinstance(voxel, (list, tuple)) else (x, y, z)

    dirs = fibonacci_hemisphere(K_DIRECTIONS, "y")
    if k is None:
        arcs = [missing_arc_deg(d, alpha_deg if alpha_deg is not None else ALPHA_DEG.get(dataset, 45.0)) for d in dirs]
        k = int(np.argmax(arcs))
    q = dirs[k]
    scalar = {lab: np.tensordot(v, _basis(q[None], ell_max, v.shape[-1])[0], axes=([3], [0])) for lab, v in vols.items()}

    sl, pr, t_sl, t_pr, vmap = _PLANES[plane]
    slicer, title = (pr, t_pr) if projection else (sl, t_sl)
    ref_img = slicer(scalar[ref], x, y, z)
    lo, hi = float(np.percentile(ref_img, 0.5)), float(np.percentile(ref_img, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    n_cols = len(labels)
    fig = plt.figure(figsize=(2.35 * n_cols, 2.5 + (2.7 if want_voxel else 0)))
    gs = fig.add_gridspec(2 if want_voxel else 1, n_cols)
    axes, poles, im = [], [], None
    for ci, lab in enumerate(labels):
        ax = fig.add_subplot(gs[0, ci]); axes.append(ax)
        im = ax.imshow(slicer(scalar[lab], x, y, z), cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="equal", rasterized=True)
        if want_voxel:
            fixed, col, row = vmap(vx, vy, vz)
            if projection or fixed == {"yz": x, "xz": y, "xy": z}[plane]:
                ax.scatter(col, row, s=90, facecolors="none", edgecolors="black", linewidths=2.4, zorder=4)
                ax.scatter(col, row, s=90, facecolors="none", edgecolors="white", linewidths=1.1, zorder=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles.get(lab, lab), fontsize=fontsize)
        if ci == 0:
            ax.set_ylabel(title.format(x=x, y=y, z=z), fontsize=max(fontsize - 2, 6))

    if want_voxel:
        n_th, n_ph = 46, 181                                    # 2 degree grid over the z >= 0 hemisphere
        PHI, THETA = np.meshgrid(np.linspace(0, 2 * np.pi, n_ph), np.linspace(0, np.pi / 2, n_th))
        pf = np.stack([np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)], -1).reshape(-1, 3)
        qd = -q if q[2] < 0 else q                              # Friedel fold: I(q) = I(-q) for even-l SH
        r_q, phi_q = np.arctan(np.arccos(np.clip(qd[2], -1, 1)) / 2), np.arctan2(qd[1], qd[0])
        ticks = np.linspace(0, 90, 4)
        for ci, lab in enumerate(labels):
            ap = fig.add_subplot(gs[1, ci], projection="polar"); poles.append(ap)
            c = vols[lab][vx, vy, vz]
            val = (_basis(pf, ell_max, len(c)) @ c).reshape(n_th, n_ph)
            im = ap.pcolormesh(PHI, np.arctan(THETA / 2), val, cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
            ap.scatter([phi_q], [r_q], s=90, facecolors="none", edgecolors="black", linewidths=2.4, zorder=6)
            ap.scatter([phi_q], [r_q], s=90, facecolors="none", edgecolors="white", linewidths=1.1, zorder=7)
            ap.set_yticks(np.arctan(ticks * np.pi / 360)); ap.set_yticklabels([f"{t:.0f}°" for t in ticks] if ci == 0 else [])
            ap.set_xticklabels([]); ap.set_ylim(0, np.arctan(0.5 * np.pi / 2)); ap.tick_params(labelsize=max(fontsize - 3, 6))
            if ci == 0:
                ap.set_ylabel(f"RSM  (x={vx}, y={vy}, z={vz})", fontsize=max(fontsize - 2, 6), labelpad=22)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    cb = fig.colorbar(im, ax=axes + poles, shrink=0.6, pad=0.015, fraction=0.025)
    cb.set_label("$I_q$ (a.u.)", fontsize=max(fontsize - 2, 6)); cb.ax.tick_params(labelsize=max(fontsize - 3, 6))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# figure name -> what to draw (positions from the paper's figure list; "kind" decides which result ids hold the volumes)
WEDGE = OrderedDict([
    ("fig_trabecular_bone_wedge", dict(dataset="bone-main", kind="bone", k=0, voxel=(36, 30, 49), x=36)),
    ("fig_trabecular_bone_covered", dict(dataset="bone-main", kind="bone", k=2, voxel=(36, 30, 49), x=36)),
    ("fig_mammoth_wedge", dict(dataset="nielsen-mammoth", kind="phantom", k=29, plane="xz")),
    ("fig_nielsenm_wedge", dict(dataset="nielsen-m", kind="phantom", k=22, voxel=(13, 25, 19), y=25, plane="xz")),
    ("fig_nielsent_wedge", dict(dataset="nielsen-t", kind="phantom", k=29, voxel=(23, 24, 9), y=24, plane="xz")),
    ("fig_steelwire_wedge", dict(dataset="steel-wire", kind="real", k=0, plane="xy", projection=True, voxel=(36, 31, 23), z=23)),
    ("fig_aragonite_wedge", dict(dataset="aragonite", kind="real")),
    ("fig_beaver_wedge", dict(dataset="beaver-tooth", kind="real", k=0, voxel=(18, 24, 19), x=18)),
    ("fig_narwhal_wedge", dict(dataset="narwhal-tusk", kind="real")),
])
_METHOD_TITLE = {"NeTT": "NeTT (ours)", "SH": "mumott SH", "GK": "mumott GK", "ZH": "mumott ZH"}


def wedge_volumes(C: Ctx, spec: dict):
    """-> ``({label: volume}, {label: column title})`` for one entry of :data:`WEDGE`, from the stored result ids:
    phantoms ``table1/<ds>/``, real datasets ``table2/<ds>/random_``, bone ``bone/bone-main/`` (anchor column: the
    combined-mount GK reconstruction, ``bone/bone-combined/GK``, which is not a ground truth)."""
    ds, kind = spec["dataset"], spec["kind"]
    prefix = {"phantom": f"table1/{ds}/", "real": f"table2/{ds}/random_", "bone": "bone/bone-main/"}[kind]
    vols, titles = OrderedDict(), {}
    if kind == "bone":
        v = C.volume("bone/bone-combined/GK")
        if v is not None:
            vols["ref"], titles["ref"] = v, "Combined (GK), not GT"
    for m in ("NeTT",) + BASELINES:
        v = C.volume(prefix + _tag(m))
        if v is not None:
            vols[m], titles[m] = v, _METHOD_TITLE[m]
    if kind == "phantom" and vols:
        try:
            from ..data import registry
            vols["ref"], titles["ref"] = np.asarray(registry.get(ds).ground_truth(), np.float32), "GT"
            vols.move_to_end("ref", last=False)
        except Exception as e:
            print(f"[wedge] no ground truth for {ds} ({type(e).__name__}); column skipped (set NETT_DATA and run `nett data prepare`)")
    # common channel count (the volumes of different basis sets can differ in the number of coefficients)
    if vols:
        n = max(v.shape[-1] for k_, v in vols.items() if k_ != "ref") if len(vols) > 1 else None
        for k_, v in list(vols.items()):
            if n and v.shape[-1] != n:
                vols[k_] = v[..., :n] if v.shape[-1] > n else np.pad(v, [(0, 0)] * 3 + [(0, n - v.shape[-1])])
    return vols, titles


def fig_wedge(C: Ctx, out: Path, names: Optional[Sequence[str]] = None) -> List[Path]:
    """The qualitative comparison figures (columns: ground truth or anchor, NeTT, SH, GK, ZH) of :data:`WEDGE`."""
    from ..data import registry
    written = []
    for name in (names or WEDGE):
        spec = dict(WEDGE[name])
        C.missing.clear(); C.hits = 0
        vols, titles = wedge_volumes(C, spec)
        if not any(k_ != "ref" for k_ in vols):
            print(f"[skip] {name}: no reconstruction found" + _missing_msg(C))
            continue
        if C.missing:
            print(f"[partial] {name}" + _missing_msg(C))
        ell = registry.DATASETS[spec["dataset"]].ell_max
        p = wedge_figure(vols, ell, Path(out) / f"{name}.pdf", dataset=spec["dataset"], k=spec.get("k"), plane=spec.get("plane", "yz"),
                         x=spec.get("x"), y=spec.get("y"), z=spec.get("z"), voxel=spec.get("voxel", True),
                         projection=spec.get("projection", False), ref="ref", titles=titles)
        if p:
            written.append(p)
    return written


# --------------------------------------------------------------------------------------------------------
# Bone: orientation agreement between reconstructions of the two mounts and of both together
# --------------------------------------------------------------------------------------------------------

PAIRS = (("main", "remount"), ("main", "combined"), ("remount", "combined"))


def bone_agreement(results=None, out: Optional[Path] = None, C: Optional[Ctx] = None) -> dict:
    """Orientation agreement (degrees, symmetric; see :func:`nett.metrics.orientation_agreement`) of the bone reconstructions of
    the mounts ``main``, ``remount`` and ``combined``, per method (NeTT: seed 0) and pair. Needs no GPU. Written to
    ``<out>/bone_agreement.json`` as ``{method: {"main-remount": {"mean", "a_vs_b", "b_vs_a"}, ...}}``."""
    from ..metrics import orientation_agreement
    C = C or Ctx(results)
    res: dict = {}
    for m in ("NeTT",) + BASELINES:
        vol = {mt: C.volume(f"bone/bone-{mt}/{_tag(m)}") for mt in ("main", "remount", "combined")}
        cell = {}
        for a, b in PAIRS:
            if vol[a] is not None and vol[b] is not None:
                cell[f"{a}-{b}"] = orientation_agreement(vol[a], vol[b], 8)
        if cell:
            res[m] = cell
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bone_agreement.json").write_text(json.dumps(res, indent=1))
        print(f"[saved] {out / 'bone_agreement.json'}")
    return res


# --------------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------------

def _missing_msg(C: Ctx, n: int = 6) -> str:
    miss = sorted(set(C.missing))
    return f": {len(miss)} result id(s) missing, e.g. {', '.join(miss[:n])}" if miss else ""


LINE_FIGURES = OrderedDict([
    ("noise", fig_noise),
    ("sparsity", fig_sparsity),
    ("sparsity_maxmin", lambda C, out: fig_sparsity(C, out, "maxmin")),
    ("robustness", fig_robustness),
    ("robustness_corr", lambda C, out: fig_robustness(C, out, COR, "fig_robustness_corr.pdf")),
])
ALL = list(LINE_FIGURES) + ["wedge", "agreement"]


def main(results: Optional[str] = None, out: str = "paper_figures", only: Optional[Sequence[str]] = None) -> None:
    """Build the figures named in ``only`` (default: all) from ``results`` into ``out``; never raises on missing results."""
    C = Ctx(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(C.R.rec)} finished runs in {C.root}")
    wanted: List[str] = []
    for n in (only or ALL):
        wanted += list(WEDGE) if n == "wedge" else [n]
    for name in wanted:
        if name not in LINE_FIGURES and name not in WEDGE and name != "agreement":
            print(f"[error] unknown figure {name!r}; choose from {', '.join(ALL + list(WEDGE))}")
            continue
        C.missing.clear(); C.hits = 0
        try:
            if name in LINE_FIGURES:
                LINE_FIGURES[name](C, out)
            elif name == "agreement":
                if bone_agreement(out=out, C=C) == {}:
                    print("[skip] agreement: no bone reconstructions found" + _missing_msg(C))
            else:
                fig_wedge(C, out, [name])
        except Exception as e:                                    # a broken figure must not stop the others
            plt.close("all")
            print(f"[error] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
