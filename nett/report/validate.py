"""Compare freshly computed results with the numbers published in the paper (``expected/paper_numbers.json``).

    nett validate [--results DIR] [--cells nielsen-m narwhal-tusk_random ...]

Rule (cells with NeTT runs): the mean of the new NeTT runs must lie within 2 paper standard deviations of the paper
mean (paper: mean +- SD of 4 runs). Deterministic baselines: relative difference <= 2 % (GPU L-BFGS is not bit-reproducible).
Only cells for which results exist are checked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .tables import GT_KEYS, Results, _bone_cross, ms
from ..experiments.plan import PHANTOMS, REAL, SEEDS

EXPECTED = Path(__file__).resolve().parents[2] / "expected" / "paper_numbers.json"


def _new_cells(R: Results) -> dict:
    """Same structure as ``numbers.json`` of :mod:`nett.report.tables`, from whatever has finished."""
    out = {}
    for ph in PHANTOMS:
        cell = {}
        for k in GT_KEYS:
            v = R.nett("table1", ph, k)
            if v:
                cell.setdefault("NeTT", {})[k] = dict(zip(("mean", "sd", "n"), ms(v)))
        for m in ("SH", "GK", "ZH"):
            if R.metric(f"table1/{ph}/{m}", GT_KEYS[0]) is not None:
                cell[m] = {k: R.metric(f"table1/{ph}/{m}", k) for k in GT_KEYS}
        if cell:
            out[ph] = cell
    for ds in REAL:
        for mode in ("random", "hightilt"):
            cell = {}
            v = R.nett("table2", ds, "nrmse", prefix=f"{mode}_")
            if v:
                cell["NeTT"] = dict(zip(("mean", "sd", "n"), ms(v)))
            for m in ("SH", "GK", "ZH"):
                x = R.metric(f"table2/{ds}/{mode}_{m}", "nrmse")
                if x is not None:
                    cell[m] = x
            if cell:
                out[f"{ds}_{mode}"] = cell
    cell = {}
    nb = _bone_cross(R, "NeTT")
    if nb:
        cell["NeTT"] = dict(zip(("mean", "sd", "n"), ms(nb)))
    for m in ("SH", "GK", "ZH"):
        v = _bone_cross(R, m)
        if v:
            cell[m] = v[0]
    if cell:
        out["bone_crossmount"] = cell
    return out


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict) and "mean" not in v:
            yield from _flat(v, f"{prefix}{k}/")
        else:
            yield f"{prefix}{k}", v


def validate(results: Optional[str] = None, cells=None, rel_tol: float = 0.02, verbose: bool = True) -> bool:
    exp = dict(_flat(json.loads(EXPECTED.read_text())))
    new = dict(_flat(_new_cells(Results(results))))
    ok_all, n = True, 0
    for key in sorted(new):
        if key not in exp or (cells and not any(key.startswith(c) for c in cells)):
            continue
        e, v = exp[key], new[key]
        if isinstance(e, dict):  # NeTT: mean +- SD
            diff = abs(v["mean"] - e["mean"])
            lim = 2 * e["sd"] if e.get("sd") else 0.02 * abs(e["mean"])
            ok = diff <= lim
            msg = f"new {v['mean']:.4g} (n={v['n']}) vs paper {e['mean']:.4g} +- {e['sd']:.2g}: |diff| = {diff:.2g} <= {lim:.2g}"
        else:
            ok = abs(v - e) <= rel_tol * abs(e)
            msg = f"new {v:.4g} vs paper {e:.4g}: rel diff {(v - e) / e:+.2%}"
        n += 1
        ok_all &= ok
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {key:50s} {msg}")
    print(f"{n} cells compared: {'ALL PASS' if ok_all and n else 'SOME FAIL' if n else 'nothing to compare'}")
    return bool(ok_all and n)
