"""Smoke tests for the curated examples suite."""

import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from typing import List

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
        [
            python_executable,
            "-c",
            "import phylognn, torch, torch_geometric, ete3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def _candidate_python_executables() -> List[str]:
    candidates = [sys.executable]

    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")
    seen_roots = set()
    root_candidates = []

    for raw_root in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        sys.base_prefix,
    ):
        if not raw_root:
            continue
        root = Path(raw_root)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        root_candidates.append(root)

    for conda_prefix_path in root_candidates:
        prefix_python = conda_prefix_path / "bin" / "python"
        if str(prefix_python) not in candidates:
            candidates.append(str(prefix_python))

        envs_dirs = [conda_prefix_path / "envs", conda_prefix_path.parent / "envs"]
        for envs_dir in envs_dirs:
            if conda_default_env and conda_default_env != "base":
                named_env_python = envs_dir / conda_default_env / "bin" / "python"
                if str(named_env_python) not in candidates:
                    candidates.append(str(named_env_python))

            if envs_dir.is_dir():
                for env_dir in sorted(envs_dir.iterdir()):
                    python_path = env_dir / "bin" / "python"
                    if python_path.is_file() and str(python_path) not in candidates:
                        candidates.append(str(python_path))

    return candidates


@lru_cache(maxsize=1)
def _selected_python_executable() -> str:
    for python_executable in _candidate_python_executables():
        if _python_supports_examples(python_executable):
            return python_executable
    return sys.executable


def _run_example(script_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_selected_python_executable(), str(EXAMPLES_DIR / script_name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_examples_inventory_contains_expected_entry_points():
    present = {path.name for path in EXAMPLES_DIR.iterdir() if path.is_file()}

    assert EXPECTED_FILES.issubset(present)
    assert REMOVED_FILES.isdisjoint(present)


def test_examples_readme_references_each_supported_script():
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")

    for filename in sorted(EXPECTED_FILES - {"README.md"}):
        assert filename in readme


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
