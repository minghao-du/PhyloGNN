"""Smoke tests for the curated examples suite."""

from pathlib import Path


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


def test_examples_inventory_contains_only_the_new_entry_points():
    present = {path.name for path in EXAMPLES_DIR.iterdir() if path.is_file()}

    assert EXPECTED_FILES.issubset(present)
    assert REMOVED_FILES.isdisjoint(present)


def test_examples_readme_references_each_supported_script():
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")

    for filename in sorted(EXPECTED_FILES - {"README.md"}):
        assert filename in readme
