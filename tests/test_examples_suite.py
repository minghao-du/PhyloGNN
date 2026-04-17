"""Smoke tests for the curated examples suite."""

from pathlib import Path
import subprocess
import sys

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


def _run_example(script_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script_name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_examples_inventory_contains_only_the_new_entry_points():
    present = {path.name for path in EXAMPLES_DIR.iterdir() if path.is_file()}

    assert present == EXPECTED_FILES
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
