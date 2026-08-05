"""Sphinx configuration for the PhyloGNN user documentation."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "PhyloGNN"
author = "Minghao Du"
copyright = "2026, Minghao Du"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "__weakref__,__dict__,__module__",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
default_role = "literal"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "../issues",
    "../myprompt",
    "../../specs",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["phylognn.css"]
html_title = "PhyloGNN"
html_show_sourcelink = True
html_use_index = True
html_show_sphinx = False
pygments_style = "sphinx"
nitpicky = True
nitpick_ignore = [
    ("py:class", "Data"),
    ("py:class", "Path"),
    ("py:class", "Tensor"),
    ("py:class", "Tree"),
    ("py:class", "torch.Tensor"),
    ("py:class", "torch_geometric.data.Data"),
    ("py:class", "ete3.Tree"),
    ("py:class", "dendropy.Tree"),
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Iterable"),
    ("py:class", "collections.abc.Mapping"),
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "ete3.coretype.tree.TreeNode"),
    ("py:class", "torch.device"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "torch.nn.parameter.Parameter"),
    ("py:class", "torch.optim.optimizer.Optimizer"),
    ("py:class", "ScoreFunction"),
]

# Imported dependency docstrings contain legacy invalid-escape sequences.
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

doctest_global_setup = """
import os
os.environ.setdefault("PYTHONHASHSEED", "0")
"""

linkcheck_ignore = [
    r"https://github\.com/.*",
    r"https://docs\.github\.com/.*",
    r"https://pytorch\.org/.*",
    r"https://pytorch-geometric\.readthedocs\.io/.*",
    r"https://www\.sphinx-doc\.org/.*",
]
