"""The NeTT field: (x, y, z) -> per-voxel SH coefficients, plus the no-network control.

``NeTTField`` = multi-resolution feature grid -> small ReLU MLP -> linear head with one output per SH
coefficient. Output conventions (these are what make the recipe work, see the paper's Methods):

* the readout is divided per degree by ``(l + 1)`` (``per_l_scale``) so all channels are commensurate;
* the isotropic coefficient ``c00`` goes through a softplus (non-negative intensity);
* the head weight is initialised to exactly zero, so training starts from a uniform isotropic object;
* ``ell_mask`` (angular reveal) and ``level_weights`` (spatial reveal) are supplied by the schedule;
* ``output_scale`` bakes the target normalisation back in so the field emits physical units.

``DirectField`` is the "no network" control of the SI: one free parameter per (voxel, coefficient) behind the
same output pipeline (per-degree scale, softplus on c00, angular reveal), no spatial hierarchy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import MultiResolutionHashEncoding
from .sh import lm_list as _lm_list


def _softplus_inv(y: float) -> float:
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y))) if y < 20 else y


def _grid_coords(volume_shape) -> torch.Tensor:
    """``(X*Y*Z, 3)`` coordinates in [0, 1], isotropic: every axis is divided by the longest extent and re-centred."""
    axes = [torch.arange(n, dtype=torch.float32) for n in volume_shape]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    idx = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    centres = torch.tensor([(n - 1) / 2.0 for n in volume_shape], dtype=torch.float32)
    extent = float(max(volume_shape) - 1) if max(volume_shape) > 1 else 1.0
    return (idx - centres) / extent + 0.5


class NeTTField(nn.Module):
    def __init__(
        self,
        volume_shape,
        ell_max: int = 8,
        n_levels: int = 8,
        n_features_per_level: int = 8,
        base_resolution: int = 8,
        max_resolution: Optional[int] = None,
        table_size: Optional[int] = None,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        c00_init: float = 1.0,
        per_l_scale_power: float = 1.0,
        head_init_std: float = 0.0,
    ):
        super().__init__()
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.ell_max = ell_max
        self.lm_list = _lm_list(ell_max)
        self.num_coeffs = len(self.lm_list)
        if max_resolution is None:
            max_resolution = max(self.volume_shape)
        self.encoding = MultiResolutionHashEncoding(
            n_dims=3, n_levels=n_levels, n_features_per_level=n_features_per_level,
            base_resolution=base_resolution, max_resolution=max_resolution, table_size=table_size, include_input=True,
        )
        layers, in_dim = [], self.encoding.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.num_coeffs)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[0] = _softplus_inv(c00_init)

        ells = torch.tensor([l for l, _ in self.lm_list], dtype=torch.float32)
        self.register_buffer("per_l_scale", 1.0 / (ells + 1.0) ** per_l_scale_power)
        self.register_buffer("grid_coords", _grid_coords(self.volume_shape))
        self.register_buffer("output_scale", torch.tensor(1.0))

    def set_c00_init(self, value: float) -> None:
        self.head.bias.data[0] = _softplus_inv(value)

    def set_output_scale(self, value: float) -> None:
        self.output_scale.fill_(float(value))

    def forward(self, level_weights: Optional[torch.Tensor] = None, ell_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample the field on the full voxel grid -> ``(X, Y, Z, C)`` coefficients."""
        feats = self.encoding(self.grid_coords, level_weights=level_weights)
        scaled = self.head(self.trunk(feats)) * self.per_l_scale
        coeffs = torch.cat([F.softplus(scaled[..., :1]), scaled[..., 1:]], dim=-1)
        if ell_mask is not None:
            coeffs = coeffs * ell_mask
        coeffs = coeffs * self.output_scale
        return coeffs.reshape(*self.volume_shape, self.num_coeffs)

    def grid_parameters(self):
        return list(self.encoding.parameters())

    def other_parameters(self):
        grid_ids = {id(p) for p in self.encoding.parameters()}
        return [p for p in self.parameters() if id(p) not in grid_ids]


class _DirectCoeffTable(nn.Module):
    """Free per-voxel coefficient table; mimics the encoding interface used by the optimiser grouping/annealer."""

    def __init__(self, n_voxels: int, num_coeffs: int):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(n_voxels, num_coeffs))
        self.n_levels = 1


class DirectField(NeTTField):
    """No-network control (SI): free per-voxel SH coefficients behind the same output pipeline."""

    def __init__(self, volume_shape, ell_max: int = 8, c00_init: float = 1.0, per_l_scale_power: float = 1.0, **_ignored):
        nn.Module.__init__(self)
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.ell_max = ell_max
        self.lm_list = _lm_list(ell_max)
        self.num_coeffs = len(self.lm_list)
        self.encoding = _DirectCoeffTable(int(np.prod(self.volume_shape)), self.num_coeffs)
        self.encoding.coeff.data[:, 0] = _softplus_inv(c00_init)
        ells = torch.tensor([l for l, _ in self.lm_list], dtype=torch.float32)
        self.register_buffer("per_l_scale", 1.0 / (ells + 1.0) ** per_l_scale_power)
        self.register_buffer("grid_coords", _grid_coords(self.volume_shape))
        self.register_buffer("output_scale", torch.tensor(1.0))

    def set_c00_init(self, value: float) -> None:
        self.encoding.coeff.data[:, 0] = _softplus_inv(value)

    def forward(self, level_weights: Optional[torch.Tensor] = None, ell_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scaled = self.encoding.coeff * self.per_l_scale
        coeffs = torch.cat([F.softplus(scaled[:, :1]), scaled[:, 1:]], dim=-1)
        if ell_mask is not None:
            coeffs = coeffs * ell_mask
        coeffs = coeffs * self.output_scale
        return coeffs.reshape(*self.volume_shape, self.num_coeffs)

    def grid_parameters(self):
        return [self.encoding.coeff]

    def other_parameters(self):
        return []


def build_field(volume_shape, ell_max: int, field_type: str = "hash", **kwargs) -> NeTTField:
    if field_type == "hash":
        return NeTTField(volume_shape, ell_max=ell_max, **kwargs)
    if field_type == "direct":
        return DirectField(volume_shape, ell_max=ell_max, **kwargs)
    raise ValueError(f"field_type must be 'hash' or 'direct', got {field_type!r}")
