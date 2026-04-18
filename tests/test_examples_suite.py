"""Smoke tests for the curated examples suite."""

import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

EXPECTED_FILES = {
    "README.md",
    "feature_engineering.py",
    "tree_to_graph.py",
    "tree_io.py",
    "single_task_training.py",
}

REMOVED_FILES = {
    "examples_converter.py",
    "examples_tree_io.py",
    "feature_engineer_example.py",
    "full_pipeline.py",
    "training_example.py",
}


def _python_supports_examples(python_executable: str) -> bool:
    completed = subprocess.run(
        [python_executable, "-c", "import phylognn, torch, torch_geometric, ete3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def _conda_executable() -> Optional[str]:
    conda_executable = os.environ.get("CONDA_EXE")
    if conda_executable and Path(conda_executable).is_file():
        return conda_executable

    resolved = shutil.which("conda")
    if resolved:
        return resolved

    return None


def _resolve_conda_env_python(env_name: str) -> Optional[str]:
    conda_executable = _conda_executable()
    if conda_executable is None:
        return None

    completed = subprocess.run(
        [conda_executable, "info", "--envs", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None

    try:
        envs = json.loads(completed.stdout).get("envs", [])
    except (TypeError, ValueError):
        return None

    for env_path in envs:
        if Path(env_path).name == env_name:
            python_path = Path(env_path) / "bin" / "python"
            if python_path.is_file():
                return str(python_path)

    return None


@lru_cache(maxsize=1)
def _selected_python_executable() -> str:
    if _python_supports_examples(sys.executable):
        return sys.executable

    pytorch_python = _resolve_conda_env_python("pytorch")
    if pytorch_python and _python_supports_examples(pytorch_python):
        return pytorch_python

    raise RuntimeError(
        "Could not find a Python interpreter that can import phylognn, torch, "
        "torch_geometric, and ete3. Tried the current interpreter and the "
        "Conda environment named 'pytorch'."
    )


def _python_has_module(module_name: str) -> bool:
    completed = subprocess.run(
        [
            _selected_python_executable(),
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def _run_example(
    script_name: str,
    *,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [_selected_python_executable(), str(EXAMPLES_DIR / script_name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def test_examples_inventory_contains_expected_entry_points():
    present = {path.name for path in EXAMPLES_DIR.iterdir() if path.is_file()}

    assert EXPECTED_FILES.issubset(present)
    assert REMOVED_FILES.isdisjoint(present)


def test_examples_readme_references_each_supported_script():
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")

    for filename in sorted(EXPECTED_FILES - {"README.md"}):
        assert filename in readme

    assert "documentation-first" in readme
    assert "self-contained" in readme
    assert "optional dependencies" in readme
    assert "examples_data/simulated_trees/" in readme
    assert "python examples/single_task_training.py" in readme


@pytest.mark.parametrize(
    "script_name, expected_text",
    [
        ("feature_engineering.py", "Feature engineering summary"),
        ("tree_to_graph.py", "Graph summary"),
    ],
)
def test_self_contained_examples_run(script_name: str, expected_text: str):
    completed = _run_example(script_name)

    assert completed.returncode == 0, completed.stderr
    assert expected_text in completed.stdout
    if script_name == "feature_engineering.py":
        assert (
            "Feature order: ['node_time', 'time_bin', 'branch_length', 'is_tip', 'is_internal']"
            in completed.stdout
        )
        assert "root: node_time=4.00" in completed.stdout
    else:
        assert (
            "feature_names: ('node_time', 'time_bin', 'branch_length', 'is_tip')"
            in completed.stdout
        )
        assert "num_nodes: 5" in completed.stdout


@pytest.mark.skipif(
    not _python_has_module("dendropy"),
    reason="Optional tree I/O dependency is not installed in the selected Python environment.",
)
def test_tree_io_example_runs_when_optional_dependency_is_available():
    completed = _run_example("tree_io.py")

    assert completed.returncode == 0, completed.stderr
    assert "Tree I/O summary" in completed.stdout
    assert "Loaded tree file:" in completed.stdout
    assert "examples_data/simulated_trees/" in completed.stdout


@pytest.mark.skipif(
    _python_has_module("dendropy"),
    reason="Missing-dependency path is only relevant when dendropy is unavailable.",
)
def test_tree_io_example_reports_missing_dependency_cleanly():
    completed = _run_example("tree_io.py")

    assert completed.returncode == 0
    assert "Optional dependency missing: dendropy" in completed.stdout
    assert 'pip install -e ".[beast]"' in completed.stdout
    assert "Traceback" not in completed.stderr


def test_single_task_training_example_runs():
    completed = _run_example("single_task_training.py")

    assert completed.returncode == 0, completed.stderr
    assert "Training summary" in completed.stdout
    assert "dataset sizes:" in completed.stdout
    assert "prediction sample:" in completed.stdout

