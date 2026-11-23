"""Differentiable X-ray transform of a multi-channel volume with the ASTRA toolbox.

This is the (linear) forward operator of NeTT: the volume ``(X, Y, Z, B)`` of B scalar channels (the SH
coefficients) is projected along the beam directions of a mumott ``Geometry`` and returns ``(I, J, K, B)``
projections (I projections, J x K detector pixels). The adjoint (back-projection) is supplied as the backward
pass, so gradients flow to the field.

The beam/detector geometry is read off mumott's own ``SAXSProjector`` (its basis vectors and detector offsets)
and handed to ASTRA's ``parallel3d_vec`` geometry, so the two toolboxes agree on conventions. CUDA tensors are
linked into ASTRA zero-copy through CuPy.

Requires: ``astra-toolbox`` with CUDA, ``cupy`` (for CUDA tensors).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:
    import astra
except ImportError as e:  # pragma: no cover
    raise ImportError("nett.projector needs the ASTRA toolbox (conda install -c astra-toolbox astra-toolbox)") from e

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


def astra_geometries(geometry):
    """ASTRA ``(vol_geom, proj_geom)`` equivalent to a mumott ``Geometry``."""
    from mumott.methods.projectors import SAXSProjector

    vx, vy, vz = geometry.volume_shape
    vol_geom = astra.create_vol_geom(vy, vx, vz)  # ASTRA order is (y, x, z)
    proj_shape = geometry.projection_shape  # (j, k)
    ref = SAXSProjector(geometry)
    ray, bj, bk = ref._basis_vector_projection, ref._basis_vector_j, ref._basis_vector_k
    shift = bj * geometry.j_offsets_as_array[:, None] + bk * geometry.k_offsets_as_array[:, None]
    vec = np.zeros((len(geometry), 12))
    vec[:, 0:3], vec[:, 3:6], vec[:, 6:9], vec[:, 9:12] = ray, shift, bj, bk
    proj_geom = astra.create_proj_geom("parallel3d_vec", proj_shape[1], proj_shape[0], vec)
    return vol_geom, proj_geom


def _cupy_view(t: torch.Tensor):
    return cp.ndarray(
        shape=t.shape, dtype=cp.float32,
        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * 4, t), 0),
    )


class AstraOperator:
    """Linear operator ``A`` (volume -> projections) and its adjoint ``A^T`` for one geometry."""

    def __init__(self, geometry, device: torch.device):
        self.device = device
        self.vol_geom, self.proj_geom = astra_geometries(geometry)
        self.n_proj = len(geometry)
        self.proj_shape = tuple(geometry.projection_shape)  # (J, K)
        self.vol_shape = tuple(int(s) for s in geometry.volume_shape)

    # --- one channel ------------------------------------------------------------------------------------
    def forward_single(self, vol: torch.Tensor) -> torch.Tensor:
        """(X, Y, Z) -> (I, J, K)."""
        J, K = self.proj_shape
        sino_shape = (K, self.n_proj, J)  # ASTRA sinogram layout: (det rows, angles, det cols)
        vol = vol.contiguous().float()
        if vol.device.type == "cuda":
            vol_a = cp.ascontiguousarray(cp.transpose(_cupy_view(vol), (2, 1, 0)))  # (z, y, x)
            sino = cp.zeros(sino_shape, dtype=cp.float32)
            vid, sid = astra.data3d.link("-vol", self.vol_geom, vol_a), astra.data3d.link("-sino", self.proj_geom, sino)
            self._run("FP3D_CUDA", dict(VolumeDataId=vid, ProjectionDataId=sid))
            astra.data3d.delete([sid, vid])
            return torch.as_tensor(sino, device=self.device).permute(1, 2, 0).clone()
        vol_a = np.transpose(vol.detach().cpu().numpy(), (2, 1, 0)).copy()
        vid = astra.data3d.create("-vol", self.vol_geom, vol_a)
        sid = astra.data3d.create("-sino", self.proj_geom, np.zeros(sino_shape, dtype=np.float32))
        self._run("FP3D_CUDA", dict(VolumeDataId=vid, ProjectionDataId=sid))
        out = astra.data3d.get(sid)
        astra.data3d.delete([sid, vid])
        return torch.from_numpy(out).float().to(self.device).permute(1, 2, 0)

    def adjoint_single(self, proj: torch.Tensor) -> torch.Tensor:
        """(I, J, K) -> (X, Y, Z)."""
        X, Y, Z = self.vol_shape
        if proj.device.type == "cuda":
            proj_a = proj.permute(2, 0, 1).contiguous().float()
            vol = cp.zeros((Z, Y, X), dtype=cp.float32)
            vid, sid = astra.data3d.link("-vol", self.vol_geom, vol), astra.data3d.link("-sino", self.proj_geom, _cupy_view(proj_a))
            self._run("BP3D_CUDA", dict(ReconstructionDataId=vid, ProjectionDataId=sid))
            astra.data3d.delete([sid, vid])
            return torch.as_tensor(vol, device=self.device).permute(2, 1, 0).clone()
        proj_a = proj.permute(2, 0, 1).detach().cpu().numpy()
        vid = astra.data3d.create("-vol", self.vol_geom, np.zeros((Z, Y, X), dtype=np.float32))
        sid = astra.data3d.create("-sino", self.proj_geom, proj_a)
        self._run("BP3D_CUDA", dict(ReconstructionDataId=vid, ProjectionDataId=sid))
        out = astra.data3d.get(vid)
        astra.data3d.delete([sid, vid])
        return torch.from_numpy(out).float().to(self.device).permute(2, 1, 0)

    @staticmethod
    def _run(name: str, ids: dict) -> None:
        cfg = astra.astra_dict(name)
        cfg.update(ids)
        alg = astra.algorithm.create(cfg)
        astra.algorithm.run(alg, 1)
        astra.algorithm.delete(alg)

    # --- channel loops ----------------------------------------------------------------------------------
    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        if vol.ndim == 3:
            return self.forward_single(vol)
        return torch.stack([self.forward_single(vol[..., b]) for b in range(vol.shape[3])], dim=3)

    def adjoint(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 3:
            return self.adjoint_single(proj)
        return torch.stack([self.adjoint_single(proj[..., b]) for b in range(proj.shape[3])], dim=3)


def build_projector(geometry, device: Optional[torch.device] = None):
    """Differentiable projector ``layer(x)``: ``(X, Y, Z[, B]) -> (I, J, K[, B])`` with autograd support."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op = AstraOperator(geometry, device)

    class _Projection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            if x.ndim not in (3, 4):
                raise ValueError(f"expected (X, Y, Z) or (X, Y, Z, B), got {tuple(x.shape)}")
            return op.forward(x)

        @staticmethod
        def backward(ctx, g):
            return op.adjoint(g)

    return _Projection.apply
