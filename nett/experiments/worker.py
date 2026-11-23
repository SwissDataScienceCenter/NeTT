"""A minimal file-based work queue: start any number of workers (any scheduler, any machine sharing the results
directory); each repeatedly claims the next run whose result does not exist and which nobody else holds.

Claiming = atomically creating ``<results>/<id>.lock`` (``O_EXCL``). A lock older than ``stale_hours`` (a crashed or
pre-empted worker) is taken over; NeTT runs resume from their checkpoint. ZH runs wait for their SH run.
"""
from __future__ import annotations

import os
import re
import socket
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional

from .execute import execute, is_done, result_paths, results_root
from .plan import Run, plan


def _claim(run: Run, root: Path, stale_hours: float) -> bool:
    j, _ = result_paths(run.id, root)
    lock = j.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        if time.time() - lock.stat().st_mtime > stale_hours * 3600:
            lock.unlink(missing_ok=True)
            return _claim(run, root, stale_hours)
        return False


def work(experiments: Optional[Iterable[str]] = None, max_runs: Optional[int] = None, stale_hours: float = 6.0,
         wait_for_deps: bool = True, root: Optional[Path] = None, match: Optional[str] = None) -> List[str]:
    """``match``: only runs whose id matches this regular expression (dependencies of matching runs are included)."""
    root = root or results_root()
    runs = plan(experiments)
    if match:
        rx = re.compile(match)
        keep = {r.id for r in runs if rx.search(r.id)}
        keep |= {r.depends_on for r in runs if r.id in keep and r.depends_on}
        runs = [r for r in runs if r.id in keep]
    done: List[str] = []
    while True:
        progressed = False
        for r in runs:
            if is_done(r, root):
                continue
            if r.depends_on and not is_done(next(x for x in runs if x.id == r.depends_on), root):
                continue                                    # ZH: its SH run is not finished yet
            if not _claim(r, root, stale_hours):
                continue
            try:
                execute(r, root)
                done.append(r.id)
            except Exception:
                traceback.print_exc()
            finally:
                result_paths(r.id, root)[0].with_suffix(".lock").unlink(missing_ok=True)
            progressed = True
            if max_runs and len(done) >= max_runs:
                return done
        remaining = [r for r in runs if not is_done(r, root)]
        if not remaining:
            return done
        if not progressed:
            if not wait_for_deps:
                return done
            time.sleep(30)                                  # others are working on what is left; wait, then re-check
