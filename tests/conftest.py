"""Shared pytest configuration for local source imports."""

from pathlib import Path
import sys

import pytest

from tests.support import require_modules

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def torch_module():
    """Provide torch when available, otherwise skip the test."""
    return require_modules("torch")


@pytest.fixture
def pyg_modules():
    """Provide the core graph-testing stack when available."""
    return require_modules("torch", "torch_geometric")


@pytest.fixture
def ete3_module():
    """Provide ete3 when available."""
    return require_modules("ete3")
