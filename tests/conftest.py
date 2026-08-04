"""Shared pytest configuration for local source imports."""

from pathlib import Path
import sys

import pytest
import torch

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


@pytest.fixture
def association_tree(ete3_module):
    """Provide a deterministic six-leaf tree for association tests."""
    return ete3_module.Tree("((A:1,B:2):1,(C:1,D:3):1,(E:2,F:1):2);")


@pytest.fixture
def association_leaf_names():
    """Provide the tree traversal order used by association fixtures."""
    return ("A", "B", "C", "D", "E", "F")


@pytest.fixture
def association_representations():
    """Provide padded leaf-aligned position representations of shape [6, 4, 3]."""
    return torch.tensor(
        [
            [[1.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 1.0, 0.5], [9.0, 9.0, 9.0]],
            [[0.0, 1.0, 0.2], [1.0, 0.0, 0.4], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
            [[0.3, 0.4, 1.0], [0.6, 0.1, 0.8], [0.5, 0.7, 0.2], [0.1, 0.9, 0.3]],
            [[0.9, 0.2, 0.1], [0.4, 0.8, 0.6], [0.2, 0.3, 0.7], [9.0, 9.0, 9.0]],
            [[0.7, 0.1, 0.9], [0.2, 0.6, 0.5], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
            [[0.1, 0.5, 0.8], [0.8, 0.4, 0.2], [0.6, 0.6, 0.6], [9.0, 9.0, 9.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def association_position_mask():
    """Provide the valid-position mask for ``association_representations``."""
    return torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )


@pytest.fixture
def association_targets():
    """Provide finite, non-constant leaf-aligned regression targets."""
    return torch.tensor([-1.0, -0.4, 0.2, 0.9, 1.5, 2.1], dtype=torch.float32)
