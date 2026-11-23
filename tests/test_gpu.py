"""GPU tests (need CUDA + the ASTRA toolbox)."""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def tiny_dc(data_available):
    if not data_available:
        pytest.skip("nielsen data not available")
    from nett.data import registry, splits
    dc = registry.get("nielsen-t").dc()
    train, _ = splits.split(dc, "none")
    return train


def test_projector_adjoint_pair(tiny_dc):
    """<A x, y> == <x, A^T y>: the backward pass really is the adjoint of the forward projection."""
    from nett.projector import build_projector
    from nett.train import subset_geometry
    g = subset_geometry(tiny_dc.geometry, np.arange(0, 40))
    dev = torch.device("cuda")
    A = build_projector(g, dev)
    x = torch.rand(*g.volume_shape, 2, device=dev, requires_grad=True)
    y = torch.randn(len(g), *g.projection_shape, 2, device=dev)
    lhs = float((A(x) * y).sum())
    gx, = torch.autograd.grad((A(x) * y).sum(), x)
    rhs = float((x * gx).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-3


def test_short_fit_reduces_loss(tiny_dc):
    from nett import NeTTConfig, fit
    r = fit(tiny_dc, NeTTConfig(ell_max=2, n_iterations=120, batch_size=100), verbose=False)
    assert np.mean(r.losses[-10:]) < 0.5 * np.mean(r.losses[:5])
    assert r.coefficients.shape[-1] == 6 and np.isfinite(r.coefficients).all()
