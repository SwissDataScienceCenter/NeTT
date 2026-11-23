"""The experiment grid of the paper, as data.

Every result of the paper (and of its Supplementary Information) is a set of :class:`Run`s. A run is one
reconstruction (NeTT or a mumott baseline) of one dataset under one data condition, followed by one evaluation:

==============  ===========================================================================================  ==========
experiment      what it produces                                                                             evaluation
==============  ===========================================================================================  ==========
``table1``      simulated phantoms x {NeTT x 4 seeds, SH, GK, ZH}; Table 1 (+ SI isotropic/RSM-corr tables)  ground truth
``table2``      4 experimental datasets x {random, high-tilt holdout} x {NeTT x 4 seeds, SH, GK, ZH}          held-out NRMSE
``bone``        trabecular bone, mounts main/remount: cross-mount NRMSE + orientation agreement (+combined)   cross-mount
``noise``       phantoms at Poisson rates 10, 31.6 (100 = table1) x {NeTT, SH, GK}; the noise figure         ground truth
``sparsity``    phantoms x {maxmin, random, random1, random2} pruning x levels x {NeTT, SH, GK}; figures      ground truth
``control``     SI "no network": free per-voxel coefficients, grid-lr multiplier x3/x10/x30, 3 phantoms      ground truth
==============  ===========================================================================================  ==========

NeTT seeds: the paper reports the mean +- SD of 4 independent runs; here they are ``seed = 0, 1, 2, 3``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

PHANTOMS = ("nielsen-m", "nielsen-t", "nielsen-mammoth")
REAL = ("steel-wire", "aragonite", "beaver-tooth", "narwhal-tusk")
BASELINES = ("SH", "GK", "ZH")
SEEDS = (0, 1, 2, 3)
NOISE_RATES = ("1.0e+01", "3.2e+01")           # 1.0e+02 is the clean data of table1
PRUNE_STRATEGIES = ("maxmin", "random", "random1", "random2")
PRUNE_LEVELS = (100, 90, 80, 70, 60, 50, 40, 30, 20)
PRUNE_EXTRA_LEVEL = {"random": 10, "random1": 10, "random2": 10}     # 10 % (42 projections), phantoms M and mammoth only
PRUNE_EXTRA_LEVEL_PHANTOMS = ("nielsen-m", "nielsen-mammoth")
CONTROL_LR_MULTIPLIERS = (3.0, 10.0, 30.0)
HOLDOUT_FRACTION, HOLDOUT_SEED = 0.15, 42


@dataclass(frozen=True)
class Run:
    id: str                                   # unique; also the result path: <results>/<id>.json (+ .npy)
    experiment: str
    dataset: str                              # registry key (see nett.data.registry)
    method: str                               # "NeTT" | "SH" | "GK" | "ZH"
    evaluation: str                           # "ground_truth" | "heldout" | "cross_mount" | "none"
    holdout: str = "none"                     # how train/held are made: none | random | hightilt
    seed: int = 0                             # NeTT training seed
    poisson_rate: Optional[str] = None        # phantoms: noise level (None = clean 1.0e+02)
    prune: Optional[Tuple[str, int]] = None   # (strategy, level in %)
    overrides: Tuple[Tuple[str, str], ...] = ()   # NeTTConfig overrides (ablations)
    depends_on: Optional[str] = None          # ZH starts from the SH run of the same data
    ell_max: Optional[int] = None             # SH band limit; None = the dataset's value in the paper
    est_minutes: int = 60

    def overrides_dict(self) -> Dict[str, str]:
        return dict(self.overrides)


def _run(exp, ds, method, ev, tag, **kw) -> Run:
    return Run(id=f"{exp}/{ds}/{tag}", experiment=exp, dataset=ds, method=method, evaluation=ev, **kw)


def _methods(exp, ds, ev, prefix="", methods=("NeTT",) + BASELINES, seeds=SEEDS, **kw) -> Iterator[Run]:
    """NeTT x seeds + baselines for one data condition. ``prefix`` distinguishes conditions inside a dataset."""
    for m in methods:
        if m == "NeTT":
            for s in seeds:
                yield _run(exp, ds, m, ev, f"{prefix}NeTT_s{s}", seed=s, est_minutes=60, **kw)
        elif m == "ZH":
            yield _run(exp, ds, m, ev, f"{prefix}ZH", depends_on=f"{exp}/{ds}/{prefix}SH", est_minutes=120, **kw)
        else:
            yield _run(exp, ds, m, ev, f"{prefix}{m}", est_minutes=30, **kw)


def table1() -> List[Run]:
    return [r for ds in PHANTOMS for r in _methods("table1", ds, "ground_truth")]


def table2() -> List[Run]:
    out = []
    for ds in REAL:
        for mode in ("random", "hightilt"):
            out += list(_methods("table2", ds, "heldout", prefix=f"{mode}_", holdout=mode))
    return out


def bone() -> List[Run]:
    out = []
    for mount in ("main", "remount"):
        out += list(_methods("bone", f"bone-{mount}", "cross_mount"))
    # both mounts together: the reference for the orientation-agreement metric (one NeTT run, no seeds)
    out += list(_methods("bone", "bone-combined", "none", seeds=(0,)))
    return out


def noise() -> List[Run]:
    return [r for ds in PHANTOMS for rate in NOISE_RATES
            for r in _methods("noise", ds, "ground_truth", prefix=f"r{rate}_", seeds=(0,), methods=("NeTT", "SH", "GK"), poisson_rate=rate)]


def sparsity() -> List[Run]:
    out = []
    for ds in PHANTOMS:
        for strat in PRUNE_STRATEGIES:
            levels = list(PRUNE_LEVELS)
            if strat in PRUNE_EXTRA_LEVEL and ds in PRUNE_EXTRA_LEVEL_PHANTOMS:
                levels.append(PRUNE_EXTRA_LEVEL[strat])
            for lv in levels:
                out += list(_methods("sparsity", ds, "ground_truth", prefix=f"{strat}_L{lv}_", seeds=(0,),
                                     methods=("NeTT", "SH", "GK"), prune=(strat, lv)))
    return out


def control() -> List[Run]:
    return [
        _run("control", ds, "NeTT", "ground_truth", f"direct_lr{int(m)}", seed=0,
             overrides=(("field_type", "direct"), ("grid_lr_multiplier", str(m))))
        for ds in PHANTOMS for m in CONTROL_LR_MULTIPLIERS
    ]


EXPERIMENTS = {"table1": table1, "table2": table2, "bone": bone, "noise": noise, "sparsity": sparsity, "control": control}


def plan(experiments=None) -> List[Run]:
    """All runs of the given experiments (default: everything), in dependency order (SH before ZH)."""
    runs: List[Run] = []
    for e in (experiments or EXPERIMENTS):
        runs += EXPERIMENTS[e]()
    ids = [r.id for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    return runs


def by_id(run_id: str) -> Run:
    for r in plan():
        if r.id == run_id:
            return r
    raise KeyError(run_id)
