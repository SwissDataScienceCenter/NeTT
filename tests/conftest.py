import os
import pytest


def _has_gpu():
    try:
        import torch
        import astra  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if not _has_gpu():
        skip = pytest.mark.skip(reason="needs CUDA + ASTRA")
        for it in items:
            if "gpu" in it.keywords:
                it.add_marker(skip)


@pytest.fixture(scope="session")
def data_available():
    from nett.data import registry
    return (registry.data_root() / "nielsen" / "T_sim_1.0e+02.h5").exists()
