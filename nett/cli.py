"""Command line interface: ``nett <command> ...`` (or ``python -m nett``).

    nett data download|prepare            fetch and prepare the data (see nett.data)
    nett fit DATASET [...]                one NeTT reconstruction + evaluation
    nett baseline METHOD DATASET [...]    one mumott baseline (SH | GK | ZH) + evaluation
    nett plan [--experiment E] [--list]   list the runs behind the paper's results
    nett run RUN_ID [...]                 execute one run of the plan
    nett worker [--experiment E]          work through the plan (file-based queue; start many in parallel)
    nett tables                           build the paper's tables from the stored results
    nett figures                          build the paper's figures from the stored results
    nett validate                         compare results with the numbers published in the paper
    nett check                            check the environment
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def cmd_data(a):
    from .data import download as D, prepare as P, registry
    if a.what == "download":
        D.download(a.only, Path(a.dest) if a.dest else None)
    elif a.what == "prepare":
        P.prepare(Path(a.dest) if a.dest else None)
    elif a.what == "list":
        for k, d in registry.DATASETS.items():
            print(f"{k:16s} {d.kind:8s} ell_max={d.ell_max:2d}  {d.description}")


def _single(a, method):
    from .experiments.execute import execute
    from .experiments.plan import Run
    ev = {"phantom": "ground_truth", "real": "heldout", "bone": "none"}
    from .data import registry
    ds = registry.get(a.dataset, a.poisson_rate)
    holdout = a.holdout or ("none" if ds.kind in ("phantom", "bone") else "random")
    evaluation = a.evaluation or ev[ds.kind]
    run = Run(id=f"adhoc/{a.dataset}/{method}{'_s' + str(a.seed) if method == 'NeTT' else ''}", experiment="adhoc", dataset=a.dataset,
              method=method, evaluation=evaluation, holdout=holdout, seed=a.seed, poisson_rate=a.poisson_rate,
              ell_max=a.ell_max, overrides=tuple(sorted(_kv(a.set).items())))
    execute(run, Path(a.out) if a.out else None, force=a.force)


def cmd_fit(a):
    _single(a, "NeTT")


def cmd_baseline(a):
    _single(a, a.method)


def cmd_plan(a):
    from .experiments.execute import is_done
    from .experiments.plan import plan
    runs = plan(a.experiment)
    for i, r in enumerate(runs):
        print(f"{i:4d}  {'done' if is_done(r) else '    '}  {r.id}")
    print(f"{len(runs)} runs, ~{sum(r.est_minutes for r in runs) / 60:.0f} GPU-hours; "
          f"{sum(is_done(r) for r in runs)} done")


def cmd_run(a):
    from .experiments.execute import execute
    from .experiments.plan import by_id, plan
    run = by_id(a.id) if a.id else plan(a.experiment)[a.task]
    execute(run, force=a.force)


def cmd_worker(a):
    from .experiments.worker import work
    work(a.experiment, a.max_runs, a.stale_hours, match=a.match)


def cmd_tables(a):
    from .report import tables
    tables.main(a.results, a.out)


def cmd_figures(a):
    from .report import figures
    figures.main(a.results, a.out, a.only)


def cmd_validate(a):
    from .report.validate import validate
    sys.exit(0 if validate(a.results, a.cells) else 1)


def cmd_check(a):
    import importlib
    ok = True
    for mod in ("torch", "numpy", "scipy", "h5py", "skimage", "mumott", "astra", "cupy"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:10s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            ok = False
            print(f"{mod:10s} MISSING ({e.__class__.__name__})")
    import torch
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="nett", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data"); d.add_argument("what", choices=["download", "prepare", "list"])
    d.add_argument("--only", nargs="*"); d.add_argument("--dest"); d.set_defaults(f=cmd_data)

    def common(s):
        s.add_argument("dataset"); s.add_argument("--ell-max", type=int); s.add_argument("--holdout", choices=["none", "random", "hightilt"])
        s.add_argument("--evaluation", choices=["ground_truth", "heldout", "cross_mount", "none"]); s.add_argument("--poisson-rate")
        s.add_argument("--seed", type=int, default=0); s.add_argument("--out"); s.add_argument("--force", action="store_true")
        s.add_argument("--set", action="append", metavar="KEY=VALUE", help="NeTTConfig override (ablations)")

    f = sub.add_parser("fit"); common(f); f.set_defaults(f=cmd_fit)
    b = sub.add_parser("baseline"); b.add_argument("method", choices=["SH", "GK", "ZH"]); common(b); b.set_defaults(f=cmd_baseline)
    pl = sub.add_parser("plan"); pl.add_argument("--experiment", nargs="*"); pl.set_defaults(f=cmd_plan)
    r = sub.add_parser("run"); r.add_argument("id", nargs="?"); r.add_argument("--experiment", nargs="*"); r.add_argument("--task", type=int)
    r.add_argument("--force", action="store_true"); r.set_defaults(f=cmd_run)
    w = sub.add_parser("worker"); w.add_argument("--experiment", nargs="*"); w.add_argument("--max-runs", type=int)
    w.add_argument("--stale-hours", type=float, default=6.0); w.add_argument("--match", help="regex on run ids"); w.set_defaults(f=cmd_worker)
    t = sub.add_parser("tables"); t.add_argument("--results"); t.add_argument("--out", default="paper_tables"); t.set_defaults(f=cmd_tables)
    fg = sub.add_parser("figures"); fg.add_argument("--results"); fg.add_argument("--out", default="paper_figures")
    fg.add_argument("--only", nargs="*"); fg.set_defaults(f=cmd_figures)
    v = sub.add_parser("validate"); v.add_argument("--results"); v.add_argument("--cells", nargs="*"); v.set_defaults(f=cmd_validate)
    c = sub.add_parser("check"); c.set_defaults(f=cmd_check)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
