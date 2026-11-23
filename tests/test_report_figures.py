"""``nett figures`` on a synthetic results directory (made-up metrics, tiny random volumes; no data, no GPU)."""
import json
from pathlib import Path

import numpy as np
import pytest

from nett.experiments.plan import plan
from nett.report import figures as F


def _write(root: Path, run_id: str, metrics: dict, volume=None):
    p = root / f"{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"metrics": metrics}))
    if volume is not None:
        np.save(root / f"{run_id}.npy", volume.astype(np.float32))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    for r in plan(["table1", "noise", "sparsity"]):
        base = {"SH": 30.0, "GK": 25.0, "ZH": 28.0, "NeTT": 15.0}[r.method]
        lvl = r.prune[1] if r.prune else 100
        rate = float(r.poisson_rate or 100)
        worse = 1 + (100 - lvl) / 100 + 10.0 / rate
        _write(root, r.id, {"orientation_error_mean": base * worse + rng.normal(), "rsm_corr_mean": 1.0 / worse})
    # tiny volumes: one phantom for the wedge figure (nielsen-t: ell_max 2, 6 coefficients), bone mounts for the agreement
    for m in ("NeTT_s0", "SH", "GK", "ZH"):
        _write(root, f"table1/nielsen-t/{m}", {}, rng.random((10, 10, 10, 6)))
    for mount in ("main", "remount", "combined"):
        for m in ("NeTT_s0", "SH", "GK", "ZH"):
            _write(root, f"bone/bone-{mount}/{m}", {}, rng.random((8, 8, 8, 45)) + 0.1)
    return root


def _pdf_ok(p: Path):
    assert p.exists() and p.stat().st_size > 1000 and p.read_bytes()[:4] == b"%PDF"


@pytest.mark.parametrize("name,fname", [("noise", "fig_noise.pdf"), ("sparsity", "fig_sparsity.pdf"), ("sparsity_maxmin", "fig_sparsity_maxmin.pdf"),
                                        ("robustness", "fig_robustness.pdf"), ("robustness_corr", "fig_robustness_corr.pdf")])
def test_line_figures(results, tmp_path, name, fname):
    F.main(str(results), str(tmp_path), [name])
    _pdf_ok(tmp_path / fname)


def test_kept_projections():
    assert [F.kept(l) for l in (100, 50, 20, 10)] == [417, 209, 83, 42]      # as in the paper's figures (125 = 30 %, 209 = 50 %)


def test_wedge_from_results_and_generic(results, tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_DATA", str(tmp_path / "no_data"))             # the tiny volumes must not meet a real ground truth
    F.main(str(results), str(tmp_path), ["fig_nielsent_wedge"])             # no ground truth available: 4 reconstruction columns
    _pdf_ok(tmp_path / "fig_nielsent_wedge.pdf")
    rng = np.random.default_rng(1)
    vols = {"GT": rng.random((9, 9, 9, 15)), "NeTT": rng.random((9, 9, 9, 15))}
    for plane, proj in (("yz", False), ("xz", False), ("xy", True)):
        p = F.wedge_figure(vols, 4, tmp_path / f"w_{plane}.pdf", dataset="nielsen-m", plane=plane, projection=proj, voxel=(3, 4, 5))
        _pdf_ok(p)
    _pdf_ok(F.wedge_figure(vols, 4, tmp_path / "w_novoxel.pdf", voxel=False, k=3))


def test_bone_agreement(results, tmp_path):
    out = F.bone_agreement(str(results), tmp_path)
    assert set(out) == {"NeTT", "SH", "GK", "ZH"}
    assert set(out["SH"]) == {"main-remount", "main-combined", "remount-combined"}
    cell = out["GK"]["main-remount"]
    assert 0 <= cell["mean"] <= 90 and abs(cell["mean"] - 0.5 * (cell["a_vs_b"] + cell["b_vs_a"])) < 1e-9
    assert json.loads((tmp_path / "bone_agreement.json").read_text()) == json.loads(json.dumps(out))


def test_missing_results_never_crash(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    F.main(str(empty), str(tmp_path / "out"))                                # everything requested, nothing exists
    msg = capsys.readouterr().out
    assert "[skip] fig_noise.pdf" in msg and "[skip] fig_mammoth_wedge" in msg and "[skip] agreement" in msg
    assert not list((tmp_path / "out").glob("*.pdf"))
    F.main(str(empty), str(tmp_path / "out"), ["nonsense"])
    assert "unknown figure" in capsys.readouterr().out


def test_partial_results_are_drawn_and_reported(results, tmp_path, capsys):
    part = tmp_path / "part"
    for r in plan(["noise"]):                                                # only the noise runs of nielsen-m, no clean point
        if r.dataset == "nielsen-m" and r.method != "ZH":
            _write(part, r.id, {"orientation_error_mean": 10.0})
    F.main(str(part), str(tmp_path / "out"), ["noise"])
    _pdf_ok(tmp_path / "out" / "fig_noise.pdf")
    assert "[partial] fig_noise.pdf" in capsys.readouterr().out
