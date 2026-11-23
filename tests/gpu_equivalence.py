"""GPU checks of the standalone code against the original research code (``smartt``, optional dependency).

    python tests/gpu_equivalence.py --dataset nielsen-t [--legacy /myhome/smartt] [--steps 150]

1. Forward/adjoint of nett.projector vs the original ASTRA wrapper (must be identical up to float rounding).
2. A short NeTT fit vs the original ``saxs_naf_reconstruction`` with the same seed and settings: loss curves and
   final coefficients must agree (GPU atomics make bit-equality impossible; they agree to ~1e-3 relative).
"""
import argparse, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nielsen-t"); ap.add_argument("--legacy", default="/myhome/smartt"); ap.add_argument("--steps", type=int, default=150)
a = ap.parse_args()
sys.path.insert(0, a.legacy)
dev = torch.device("cuda")

from nett.data import registry, splits
from nett.projector import build_projector
from nett import NeTTConfig, fit

ds = registry.get(a.dataset); dc = ds.dc()
train, held = splits.split(dc, "none", seed=42)
g = train.geometry

# ---- 1. projector ----------------------------------------------------------------------------------------
from smartt.projectors import build_mumott_projector
new, old = build_projector(g, dev), build_mumott_projector(g, device=dev)
torch.manual_seed(0)
x = torch.rand(*g.volume_shape, 3, device=dev, requires_grad=True)
yn, yo = new(x), old(x)
print("forward  max|diff|", float((yn - yo).abs().max()), " scale", float(yo.abs().max()))
gy = torch.randn_like(yn)
gn, = torch.autograd.grad(yn, x, gy); go, = torch.autograd.grad(yo, x, gy)
print("adjoint  max|diff|", float((gn - go).abs().max()), " scale", float(go.abs().max()))

# ---- 2. short fit ----------------------------------------------------------------------------------------
from smartt.saxs_naf import saxs_naf_reconstruction
ell = ds.ell_max
cfg = NeTTConfig(ell_max=ell, n_iterations=a.steps, seed=0)
t = time.time(); rn = fit(train, cfg, verbose=False); tn = time.time() - t
t = time.time()
ro = saxs_naf_reconstruction(
    train, ell_max=ell, n_iterations=a.steps, lr=3e-3, batch_size=100, reg_weight_sh=0.0, reg_weight_tv=0.0, reg_weight_nonneg=0.0,
    loss_type="mse", normalize_target=True, head_init_std=0.0, spatial_frac=0.5, angular_frac=0.7, anneal_spatial=True, anneal_angular=True,
    stochastic_angular=True, seed=0, n_features_per_level=8, hidden_dim=64, n_hidden_layers=2, grid_lr_multiplier=10.0,
    optimizer_type="adamw", compute_rank_diagnostics=False, verbose=False)
to = time.time() - t
ln, lo = np.array(rn.losses), np.array(ro["losses"])
print(f"loss new/old first {ln[0]:.5f}/{lo[0]:.5f}  last {ln[-1]:.5f}/{lo[-1]:.5f}  max rel diff over steps {np.max(np.abs(ln - lo) / lo):.2e}")
cn, co = rn.coefficients, ro["reconstruction"].numpy()
print("coeff corr", float(np.corrcoef(cn.ravel(), co.ravel())[0, 1]), " rel L2 diff", float(np.linalg.norm(cn - co) / np.linalg.norm(co)))
print(f"time new {tn:.0f}s old {to:.0f}s")
