"""CPU-only unit tests (no GPU, no data download needed unless noted)."""
import numpy as np
import pytest
import torch

from nett import NeTTConfig
from nett import sh
from nett.field import DirectField, NeTTField, build_field
from nett.schedule import Annealer
from nett.experiments import plan as P


def test_sh_basis_orthonormal_on_sphere():
    # Fibonacci sphere ~ uniform: mean(Y_a Y_b) ~ delta_ab (4*pi-normalised -> mean of Y^2 = 1)
    n = 40000
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    d = torch.tensor(np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], 1), dtype=torch.float64)
    B = sh.evaluate_real_sh(d, 6).numpy()
    G = B.T @ B / n
    assert np.allclose(G, np.eye(G.shape[0]), atol=2e-2)
    assert B.shape[1] == sh.num_coeffs(6) == len(sh.lm_list(6))


def test_lm_order_matches_mumott():
    from mumott.methods.basis_sets import SphericalHarmonics  # noqa: F401  (ordering check below uses documented convention)
    assert sh.lm_list(4) == [(0, 0)] + [(2, m) for m in range(-2, 3)] + [(4, m) for m in range(-4, 5)]


def test_config_overrides():
    c = NeTTConfig.from_overrides({"field_type": "direct", "grid_lr_multiplier": "30", "n_iterations": "100", "stochastic_angular": "false"})
    assert c.field_type == "direct" and c.grid_lr_multiplier == 30.0 and c.n_iterations == 100 and c.stochastic_angular is False
    with pytest.raises(KeyError):
        NeTTConfig.from_overrides({"nope": "1"})


def test_recipe_constants_are_the_paper_recipe():
    c = NeTTConfig()
    assert (c.n_iterations, c.lr, c.grid_lr_multiplier, c.batch_size, c.warmup_steps) == (10500, 3e-3, 10.0, 100, 50)
    assert (c.spatial_frac, c.angular_frac, c.stochastic_angular, c.head_init_std, c.per_l_scale_power) == (0.5, 0.7, True, 0.0, 1.0)
    assert (c.n_levels, c.n_features_per_level, c.hidden_dim, c.n_hidden_layers, c.loss_type) == (8, 8, 64, 2, "mse")


@pytest.mark.parametrize("kind", ["hash", "direct"])
def test_field_shapes_and_cold_start(kind):
    f = build_field((10, 12, 8), 4, field_type=kind)
    out = f()
    assert out.shape == (10, 12, 8, sh.num_coeffs(4))
    assert torch.all(out[..., 0] > 0)                        # softplus c00
    assert float(out[..., 1:].abs().max()) < 1e-6            # zero head / zero table: starts isotropic
    f.set_c00_init(3.0)
    assert abs(float(f()[..., 0].mean()) - 3.0) < 1e-4
    f.set_output_scale(2.0)
    assert abs(float(f()[..., 0].mean()) - 6.0) < 1e-3
    assert len(f.grid_parameters()) > 0


def test_direct_field_has_no_other_params():
    assert DirectField((4, 4, 4), 2).other_parameters() == []
    assert len(NeTTField((8, 8, 8), 2).other_parameters()) > 0


def test_annealer_reveal():
    f = NeTTField((8, 8, 8), 8)
    a = Annealer(f.encoding.n_levels, f.lm_list, 1000, True, True, 0.5, 0.7, True)
    assert torch.all(a.level_weights(0)[1:] == 0) and a.level_weights(0)[0] == 1
    assert torch.all(a.level_weights(600) == 1)
    torch.manual_seed(0)
    m0 = a.ell_mask(0, f.num_coeffs, 2000, torch.device("cpu"))
    m1 = a.ell_mask(900, f.num_coeffs, 2000, torch.device("cpu"))
    assert m0[:, 1:].mean() < 0.05 and m1.mean() == 1.0       # stochastic reveal: ~no voxel sees high degrees at step 0, all at the end


def test_plan_is_consistent():
    runs = P.plan()
    ids = [r.id for r in runs]
    assert len(ids) == len(set(ids))
    byid = {r.id: r for r in runs}
    for r in runs:
        if r.depends_on:
            assert r.depends_on in byid and byid[r.depends_on].method == "SH"
            assert ids.index(r.depends_on) < ids.index(r.id)
    assert sum(r.method == "NeTT" for r in P.table1()) == 3 * len(P.SEEDS)
    assert len(P.control()) == 9 and len(P.bone()) == 18


def test_pruning_levels_match_paper():
    assert P  # noqa
    from nett.data.splits import pruning_levels
    assert pruning_levels(417) == [(100, 0), (90, 42), (80, 83), (70, 125), (60, 167), (50, 208), (40, 250), (30, 292), (20, 334)]


def test_maxmin_order_is_a_nested_permutation():
    from nett.data.splits import maxmin_addition_order

    class G:  # minimal geometry stand-in
        inner_angles = np.random.default_rng(0).uniform(0, 2 * np.pi, 60)
        outer_angles = np.random.default_rng(1).uniform(-0.7, 0.7, 60)
    o = maxmin_addition_order(G)
    assert sorted(o.tolist()) == list(range(60)) and o[0] == 0


def test_metrics_perfect_reconstruction():
    from nett import metrics as M
    rng = np.random.default_rng(0)
    gt = np.zeros((6, 6, 6, 6), np.float32)
    gt[..., 0] = 1.0 + 0.1 * rng.random((6, 6, 6))
    gt[..., 3] = 0.5 * rng.random((6, 6, 6))          # l=2, m=0
    mask = np.ones((6, 6, 6), bool)
    r = M.ground_truth_metrics(gt.copy(), gt, 2, mask, K=20)
    assert r["rsm_corr_mean"] > 0.999 and r["orientation_error_mean"] < 1e-3 and r["ra_mae"] < 1e-6 and r["nrmse"] < 1e-6


@pytest.mark.skipif(not __import__("os").environ.get("NETT_DATA"), reason="NETT_DATA not set")
def test_split_and_prune_on_data(data_available):
    if not data_available:
        pytest.skip("nielsen data not available")
    from nett.data import registry, splits
    dc = registry.get("nielsen-t").dc()
    n = len(dc.projections)
    tr, he = splits.split(dc, "none")
    assert len(tr.projections) == n - 1 and len(he.projections) == 1
    tr, he = splits.split(dc, "random", 0.15, 42)
    assert len(he.projections) == round(0.15 * n)
    p50, p20 = splits.pruned(dc, "random", 50), splits.pruned(dc, "random", 20)
    a50 = {tuple(np.round(x, 6)) for x in zip(p50.geometry.inner_angles, p50.geometry.outer_angles)}
    a20 = {tuple(np.round(x, 6)) for x in zip(p20.geometry.inner_angles, p20.geometry.outer_angles)}
    assert a20 <= a50                                    # nested
