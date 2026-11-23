"""The NeTT recipe as a single, frozen set of defaults.

Everything here is the recipe of the paper ("sp3"). The only quantity a user normally chooses is ``ell_max``
(the SH band limit), because the angular content of a sample is dataset dependent (2-12 in the paper). All
other fields exist so that the ablations of the Supplementary Information can be run as overrides, e.g.

    nett fit ... --ell-max 8
    nett fit ... --set field_type=direct --set grid_lr_multiplier=10      # "no network" control

and are not meant to be tuned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict


@dataclass(frozen=True)
class NeTTConfig:
    # ---- the one free parameter -------------------------------------------------------------------------
    ell_max: int = 8                 # SH band limit (even); 2-12 in the paper
    # ---- optimisation (AdamW, linear warm-up then cosine decay) -----------------------------------------
    n_iterations: int = 10_500
    lr: float = 3e-3                 # peak lr of the MLP trunk/head
    grid_lr_multiplier: float = 10.0  # the feature grid trains at lr * this
    warmup_steps: int = 50
    batch_size: int = 100            # projections per gradient step (one projector per chunk)
    reshuffle_every: int = 200       # re-partition the projections into chunks every this many steps
    loss_type: str = "mse"           # masked mean squared error on the (normalised) projections
    normalize_target: bool = True    # divide the data by its mean |value| so the loss is O(1); undone on output
    # ---- coarse-to-fine reveal ---------------------------------------------------------------------------
    spatial_frac: float = 0.5        # feature-grid levels fully revealed after this fraction of training
    angular_frac: float = 0.7        # all SH degrees fully revealed after this fraction (stochastic reveal)
    stochastic_angular: bool = True
    # ---- field -------------------------------------------------------------------------------------------
    field_type: str = "hash"         # "hash" (NeTT) | "direct" (no-network control)
    n_levels: int = 8
    n_features_per_level: int = 8
    base_resolution: int = 8
    max_resolution: Any = None       # default: longest volume side
    table_size: Any = None           # default: max_resolution ** 3 (dense)
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    head_init_std: float = 0.0       # zero readout init -> uniform isotropic start
    per_l_scale_power: float = 1.0   # readout divided by (l + 1) ** power
    # ---- bookkeeping -------------------------------------------------------------------------------------
    seed: int = 0
    eval_every: int = 25             # held-out NRMSE logging interval (only if a held-out set is given)

    def replace(self, **kw) -> "NeTTConfig":
        return replace(self, **kw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: Dict[str, str], base: "NeTTConfig" = None) -> "NeTTConfig":
        """Apply ``{"key": "value"}`` string overrides (as given on the command line), casting to the field type."""
        base = base or cls()
        types = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in overrides.items():
            if k not in types:
                raise KeyError(f"unknown config field {k!r}; known: {sorted(types)}")
            cur = getattr(base, k)
            if isinstance(cur, bool):
                kw[k] = str(v).lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                kw[k] = int(float(v))
            elif isinstance(cur, float):
                kw[k] = float(v)
            elif cur is None:
                kw[k] = None if str(v).lower() in ("none", "") else int(v)
            else:
                kw[k] = str(v)
        return replace(base, **kw)
