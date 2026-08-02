"""Smoke tests for the curated examples suite."""

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
EXAMPLE_OUTPUT_DIR = ROOT / "example_outputs" / "toml_training_config"
TOML_CHECKPOINT = EXAMPLE_OUTPUT_DIR / "final_model.pt"
TOML_HISTORY = EXAMPLE_OUTPUT_DIR / "history.json"

EXPECTED_FILES = {
    "README.md",
    "feature_engineering.py",
    "quickstart_training.py",
    "tree_to_graph.py",
    "tree_io.py",
    "single_task_training.py",
    "toml_training_config.py",
    "toml_training_config.toml",
    "complete_pipeline.py",
    "extant_trait_regression.py",
}

REMOVED_FILES = {
    "examples_converter.py",
    "examples_tree_io.py",
    "feature_engineer_example.py",
    "full_pipeline.py",
    "training_example.py",
}


def _python_supports_examples(python_executable: str) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [python_executable, "-c", "import phylognn, torch, torch_geometric, ete3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
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

    phylognn_python = _resolve_conda_env_python("phylognn")
    if phylognn_python and _python_supports_examples(phylognn_python):
        return phylognn_python

    raise RuntimeError(
        "Could not find a Python interpreter that can import phylognn, torch, "
        "torch_geometric, and ete3. Tried the current interpreter and the "
        "Conda environment named 'phylognn'."
    )


def _python_has_module(module_name: str) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
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
        env=env,
    )
    return completed.returncode == 0


def _run_example(
    script_name: str,
    *,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
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
        ("quickstart_training.py", "Quickstart training summary"),
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
    elif script_name == "tree_to_graph.py":
        assert (
            "feature_names: ('node_time', 'time_bin', 'branch_length', 'is_tip')"
            in completed.stdout
        )
        assert "num_nodes: 5" in completed.stdout
        assert "virtual node count: 6" in completed.stdout
    else:
        assert "x shape: (5, 4)" in completed.stdout
        assert "edge_index shape: (2, 8)" in completed.stdout
        assert "target shape: (1,)" in completed.stdout
        assert "batch ready: true" in completed.stdout
        assert "prediction:" in completed.stdout


def test_tree_io_example_handles_available_or_missing_optional_dependency():
    completed = _run_example("tree_io.py")

    assert completed.returncode == 0, completed.stderr
    if _python_has_module("dendropy"):
        assert "Tree I/O summary" in completed.stdout
        assert "Loaded tree file:" in completed.stdout
        assert "examples_data/simulated_trees/" in completed.stdout
    else:
        assert "Optional dependency missing: dendropy" in completed.stdout
        assert 'pip install -e ".[beast]"' in completed.stdout
        assert "Traceback" not in completed.stderr


def test_single_task_training_example_runs():
    legacy_output_dir = ROOT / "example_outputs" / "single_task_training"
    shutil.rmtree(legacy_output_dir, ignore_errors=True)

    completed = _run_example("single_task_training.py")

    assert completed.returncode == 0, completed.stderr
    assert "Training summary" in completed.stdout
    assert "dataset sizes:" in completed.stdout
    assert "prediction sample:" in completed.stdout
    assert not legacy_output_dir.exists()


def test_toml_training_config_example_runs():
    completed = _run_example("toml_training_config.py")

    assert completed.returncode == 0, completed.stderr
    assert "TOML training run summary" in completed.stdout
    assert "configured model: GATBiLSTMNet" in completed.stdout
    assert "metrics: mse, rmse" in completed.stdout
    assert "checkpoint: example_outputs/toml_training_config/final_model.pt" in completed.stdout
    assert "history: example_outputs/toml_training_config/history.json" in completed.stdout
    assert TOML_CHECKPOINT.is_file()
    assert TOML_HISTORY.is_file()


def test_toml_training_config_uses_single_config_creation_path():
    script = (EXAMPLES_DIR / "toml_training_config.py").read_text(encoding="utf-8")

    assert script.count("create_trainer_from_config(") == 1
    assert "load_training_config" not in script


def test_complete_pipeline_example_runs_without_existing_checkpoint():
    if TOML_CHECKPOINT.exists():
        TOML_CHECKPOINT.unlink()

    completed = _run_example("complete_pipeline.py")

    assert completed.returncode == 0, completed.stderr
    assert "Complete pipeline summary" in completed.stdout
    assert "checkpoint:" in completed.stdout
    assert "graph x shape:" in completed.stdout
    assert "prediction:" in completed.stdout


def test_extant_trait_regression_example_runs():
    completed = _run_example("extant_trait_regression.py")

    assert completed.returncode == 0, completed.stderr
    assert "Extant trait regression summary" in completed.stdout
    assert "test MSE:" in completed.stdout
    assert "test R2:" in completed.stdout
    for filename in (
        "extant_trait_regression_best.pt",
        "extant_trait_regression_loss.png",
        "extant_trait_regression_scatter.png",
    ):
        assert (ROOT / "example_outputs" / filename).is_file()
