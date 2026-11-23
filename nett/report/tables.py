"""Tables of the paper from the stored results (``<results>/<experiment>/...json``).

Writes LaTeX snippets (``tab_gt.tex``, ``tab_gt_main.tex``, ``tab_gt_si.tex``, ``tab_real.tex``, ``tab_control.tex``),
``numbers.json`` (every table cell with mean / sd / n) and ``summary.csv`` (one row per finished run).

NeTT cells: mean +- SD over the independent training seeds (``sample`` standard deviation); baselines are deterministic.
Bold = best mean in a column (no significance test).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..experiments.execute import results_root
from ..experiments.plan import CONTROL_LR_MULTIPLIERS, PHANTOMS, REAL, SEEDS, plan

GT_KEYS = ["rsm_corr_mean", "ra_mae", "orientation_error_mean", "psnr", "ssim", "nrmse"]
HEAD = {"rsm_corr_mean": r"RSM corr.$\uparrow$", "ra_mae": r"RA MAE$\downarrow$", "orientation_error_mean": r"Orient.~err.$\downarrow$",
        "psnr": r"PSNR$\uparrow$", "ssim": r"SSIM$\uparrow$", "nrmse": r"NRMSE$\downarrow$"}
DEC = {"rsm_corr_mean": 3, "ra_mae": 3, "orientation_error_mean": 1, "psnr": 1, "ssim": 3, "nrmse": 2}
HIGHER = {"rsm_corr_mean", "psnr", "ssim"}
REAL_NAME = {"steel-wire": r"\shortstack[l]{steel wire\\(WAXS)}", "aragonite": r"\shortstack[l]{aragonite\\(WAXS)}",
             "beaver-tooth": r"\shortstack[l]{beaver tooth\\(WAXS)}", "narwhal-tusk": r"\shortstack[l]{narwhal tusk\\(SAXS)}"}
LABEL = {"NeTT": r"\textbf{NeTT (ours)}", "SH": "SH", "GK": "GK", "ZH": "ZH"}


class Results:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else results_root()
        self.rec: Dict[str, dict] = {}
        for r in plan():
            p = self.root / f"{r.id}.json"
            if p.exists():
                self.rec[r.id] = json.loads(p.read_text())

    def metric(self, run_id: str, key: str) -> Optional[float]:
        r = self.rec.get(run_id)
        return None if r is None else r["metrics"].get(key)

    def nett(self, exp: str, ds: str, key: str, prefix: str = "") -> List[float]:
        v = [self.metric(f"{exp}/{ds}/{prefix}NeTT_s{s}", key) for s in SEEDS]
        return [x for x in v if x is not None]


def ms(v):
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v))


def fmt(mean, sd, dec, bold):
    s = f"{mean:.{dec}f}" + (rf"{{\scriptsize$\pm${sd:.{dec}f}}}" if sd is not None else "")
    return rf"\textbf{{{s}}}" if bold else s


def table_gt(R: Results, num: dict, keys=None) -> str:
    keys = keys or GT_KEYS
    L = [r"\begin{tabular}{@{}ll" + "c" * len(keys) + r"@{}}", r"\toprule", "Dataset & Method & " + " & ".join(HEAD[k] for k in keys) + r" \\", r"\midrule"]
    for i, ph in enumerate(PHANTOMS):
        stats = {k: ms(R.nett("table1", ph, k)) if R.nett("table1", ph, k) else None for k in GT_KEYS}
        rows = {m: {k: R.metric(f"table1/{ph}/{m}", k) for k in GT_KEYS} for m in ("SH", "GK", "ZH")}
        num[ph] = {"NeTT": {k: dict(mean=stats[k][0], sd=stats[k][1], n=stats[k][2]) for k in GT_KEYS if stats[k]},
                   **{m: rows[m] for m in rows if rows[m][GT_KEYS[0]] is not None}}
        best = {}
        for k in GT_KEYS:
            cand = {m: rows[m][k] for m in rows if rows[m][k] is not None}
            if stats[k]:
                cand["NeTT"] = stats[k][0]
            best[k] = (max if k in HIGHER else min)(cand, key=cand.get) if cand else None
        for j, m in enumerate(("NeTT", "SH", "GK", "ZH")):
            cells = []
            for k in keys:
                if m == "NeTT":
                    cells.append(fmt(stats[k][0], stats[k][1], DEC[k], best[k] == "NeTT") if stats[k] else "--")
                else:
                    cells.append(fmt(rows[m][k], None, DEC[k], best[k] == m) if rows[m][k] is not None else "--")
            L.append((rf"\multirow{{4}}{{*}}{{{ph}}}" if j == 0 else "") + " & " + LABEL[m] + " & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if i < len(PHANTOMS) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def _bone_cross(R: Results, method: str) -> List[float]:
    """Mean over the two mounts (main -> remount, remount -> main); NeTT: one value per seed."""
    def v(mount, name):
        return R.metric(f"bone/bone-{mount}/{name}", "cross_mount_nrmse")
    if method == "NeTT":
        out = []
        for s in SEEDS:
            a, b = v("main", f"NeTT_s{s}"), v("remount", f"NeTT_s{s}")
            if a is not None and b is not None:
                out.append(0.5 * (a + b))
        return out
    a, b = v("main", method), v("remount", method)
    return [0.5 * (a + b)] if a is not None and b is not None else []


def table_real(R: Results, num: dict) -> str:
    L = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Dataset & Method & Random holdout & High-tilt holdout & Cross-mount \\", r"\midrule"]
    blocks = []
    for ds in REAL:
        cells = {}
        for mode in ("random", "hightilt"):
            runs = R.nett("table2", ds, "nrmse", prefix=f"{mode}_")
            st = ms(runs) if runs else None
            vals = {m: R.metric(f"table2/{ds}/{mode}_{m}", "nrmse") for m in ("SH", "GK", "ZH")}
            cand = {m: v for m, v in vals.items() if v is not None}
            if st:
                cand["NeTT"] = st[0]
            best = min(cand, key=cand.get) if cand else None
            cells[mode] = (st, vals, best)
            num[f"{ds}_{mode}"] = {"NeTT": dict(mean=st[0], sd=st[1], n=st[2]) if st else None, **vals}
        blocks.append((REAL_NAME[ds], cells))
    nb = _bone_cross(R, "NeTT")
    st = ms(nb) if nb else None
    bvals = {m: (_bone_cross(R, m) or [None])[0] for m in ("SH", "GK", "ZH")}
    cand = {m: v for m, v in bvals.items() if v is not None}
    if st:
        cand["NeTT"] = st[0]
    bbest = min(cand, key=cand.get) if cand else None
    num["bone_crossmount"] = {"NeTT": dict(mean=st[0], sd=st[1], n=st[2]) if st else None, **bvals}
    blocks.append((r"\shortstack[l]{trabecular bone\\(SAXS)}", None))
    for bi, (name, cells) in enumerate(blocks):
        for j, m in enumerate(("NeTT", "SH", "GK", "ZH")):
            row = []
            for mode in ("random", "hightilt"):
                if cells is None:
                    row.append("")
                    continue
                st_, vals, best = cells[mode]
                if m == "NeTT":
                    row.append(fmt(st_[0], st_[1], 3, best == "NeTT") if st_ else "--")
                else:
                    row.append(fmt(vals[m], None, 3, best == m) if vals[m] is not None else "--")
            if cells is not None:
                row.append("")
            elif m == "NeTT":
                row.append(fmt(st[0], st[1], 3, bbest == "NeTT") if st else "--")
            else:
                row.append(fmt(bvals[m], None, 3, bbest == m) if bvals[m] is not None else "--")
            L.append((rf"\multirow{{4}}{{*}}{{{name}}}" if j == 0 else "") + f" & {LABEL[m]} & " + " & ".join(row) + r" \\")
        L.append(r"\midrule" if bi < len(blocks) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def table_control(R: Results, num: dict) -> str:
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Dataset & $\times 3$ & $\times 10$ & $\times 30$ & NeTT \\", r"\midrule"]
    for ph in PHANTOMS:
        cells, row = [], {}
        for m in CONTROL_LR_MULTIPLIERS:
            o, c = R.metric(f"control/{ph}/direct_lr{int(m)}", "orientation_error_mean"), R.metric(f"control/{ph}/direct_lr{int(m)}", "rsm_corr_mean")
            cells.append(f"${o:.1f}^\\circ$ / {c:.3f}" if o is not None else "--")
            row[f"x{int(m)}"] = dict(orientation_error_mean=o, rsm_corr_mean=c)
        o, c = R.nett("table1", ph, "orientation_error_mean"), R.nett("table1", ph, "rsm_corr_mean")
        cells.append(f"${np.mean(o):.1f}^\\circ$ / {np.mean(c):.3f}" if o else "--")
        num[f"control_{ph}"] = row
        L.append(f"{ph} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def summary_csv(R: Results, path: Path) -> None:
    keys = sorted({k for r in R.rec.values() for k, v in r["metrics"].items() if isinstance(v, float)})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "experiment", "dataset", "method", "seed", "holdout", "poisson_rate", "prune", "n_train", *keys])
        for i, r in R.rec.items():
            u = r["run"]
            w.writerow([i, u["experiment"], u["dataset"], u["method"], u["seed"], u["holdout"], u["poisson_rate"], u["prune"], r["n_train"],
                        *[r["metrics"].get(k, "") for k in keys]])


def main(results: Optional[str] = None, out: str = "paper_tables") -> None:
    R = Results(results)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    num: dict = {}
    (out_dir / "tab_gt.tex").write_text(table_gt(R, num) + "\n")
    (out_dir / "tab_gt_main.tex").write_text(table_gt(R, {}, ["orientation_error_mean", "rsm_corr_mean", "ra_mae"]) + "\n")
    (out_dir / "tab_gt_si.tex").write_text(table_gt(R, {}, ["psnr", "ssim", "nrmse"]) + "\n")
    (out_dir / "tab_real.tex").write_text(table_real(R, num) + "\n")
    (out_dir / "tab_control.tex").write_text(table_control(R, num) + "\n")
    (out_dir / "numbers.json").write_text(json.dumps(num, indent=1))
    summary_csv(R, out_dir / "summary.csv")
    from . import macros
    macros.write(num, out_dir, R)
    print(f"{len(R.rec)} finished runs -> {out_dir}")
