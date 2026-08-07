"""Regression tests for documentation and runnable-example contracts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCS_SOURCE = ROOT / "docs" / "source"
DOCS_BUILD_HTML = ROOT / "docs" / "_build" / "html"
DOCS_EXAMPLES = DOCS_SOURCE / "examples"
USER_GUIDE_INDEX = DOCS_SOURCE / "user_guide" / "index.rst"
TOP_LEVEL_INDEX = DOCS_SOURCE / "index.rst"
EXAMPLES_DIR = ROOT / "examples"
TOML_CHECKPOINT = ROOT / "example_outputs" / "toml_training_config" / "final_model.pt"
TOML_HISTORY = ROOT / "example_outputs" / "toml_training_config" / "history.json"

USER_GUIDE_ENTRIES = [
    "tree_input",
    "leaf_regression",
    "feature_engineering",
    "graph_conversion",
    "datasets_and_splits",
    "training",
    "training_config",
    "metrics_tracking",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _toctree_entries(path: Path) -> list[str]:
    lines = _read(path).splitlines()
    entries: list[str] = []
    in_toctree = False

    for line in lines:
        stripped = line.strip()
        if stripped == ".. toctree::":
            in_toctree = True
            continue
        if not in_toctree:
            continue
        if not line.startswith((" ", "\t")):
            if stripped:
                break
            continue
        if not stripped or stripped.startswith(":"):
            continue
        entries.append(stripped)

    return entries


def _text_files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    suffixes = {".html", ".js", ".css", ".rst", ".py", ".toml", ".txt"}
    return [item for item in path.rglob("*") if item.is_file() and item.suffix in suffixes]


def _contains_case_insensitive(path: Path, needle: str) -> bool:
    return needle.lower() in _read(path).lower()


def test_docs_optional_dependency_uses_read_the_docs_theme_only():
    with PYPROJECT.open("rb") as file:
        optional = tomllib.load(file)["project"]["optional-dependencies"]

    docs_dependencies = " ".join(optional["docs"]).lower()

    assert "sphinx-rtd-theme" in docs_dependencies
    assert "furo" not in docs_dependencies


def test_sphinx_config_uses_read_the_docs_theme():
    conf = _read(DOCS_SOURCE / "conf.py")

    assert 'html_theme = "sphinx_rtd_theme"' in conf
    assert "html_theme_options" not in conf


def test_documentation_source_has_no_furo_references():
    source_files = [PYPROJECT, *_text_files_under(DOCS_SOURCE)]

    offenders = [path for path in source_files if _contains_case_insensitive(path, "furo")]

    assert offenders == []


def test_generated_html_has_no_furo_references_after_build():
    html_files = _text_files_under(DOCS_BUILD_HTML)
    assert html_files, "Run the Sphinx HTML build before checking generated output."

    offenders = [path for path in html_files if _contains_case_insensitive(path, "furo")]

    assert offenders == []


def test_user_guide_toctree_order_and_reachability():
    entries = _toctree_entries(USER_GUIDE_INDEX)

    assert entries == USER_GUIDE_ENTRIES
    assert entries.index("feature_engineering") < entries.index("graph_conversion")
    for entry in USER_GUIDE_ENTRIES:
        assert (USER_GUIDE_INDEX.parent / f"{entry}.rst").is_file()


def test_generated_user_guide_navigation_order_and_reachability():
    index_html = DOCS_BUILD_HTML / "user_guide" / "index.html"
    html = _read(index_html)

    hrefs = [f"{entry}.html" for entry in USER_GUIDE_ENTRIES]
    positions = [html.index(href) for href in hrefs]

    assert positions == sorted(positions)


def test_leaf_regression_documentation_contracts():
    guide = _read(DOCS_SOURCE / "user_guide" / "leaf_regression.rst")
    reference = _read(DOCS_SOURCE / "reference" / "leaf_regression.rst")
    example = _read(DOCS_EXAMPLES / "single_tree_leaf_regression.rst")

    for text in (guide, reference, example):
        assert "leaf" in text.lower()
        assert "mask" in text.lower()
        assert "transductive" in text.lower()

    for symbol in (
        "LeafRegressionData",
        "LeafRegressionConfig",
        "LeafFitResult",
        "LeafCrossValidationResult",
        "LeafRegressionResult",
        "prepare_leaf_regression",
        "fit_leaf_regression",
        "cross_validate_leaf_regression",
        "run_leaf_regression",
    ):
        assert symbol in reference

    for term in (
        "Inputs",
        "Run command",
        "Expected output",
        "Files written",
        "Optional dependencies",
        "Failure modes",
        "Source",
    ):
        assert term in example
    assert "examples/single_tree_leaf_regression.py" in example
    assert ".. literalinclude::" in example
    assert ":language: python" in example


def test_generated_leaf_regression_navigation_and_pages_are_reachable():
    for index_path, page_name in (
        (DOCS_BUILD_HTML / "user_guide" / "index.html", "leaf_regression.html"),
        (DOCS_BUILD_HTML / "reference" / "index.html", "leaf_regression.html"),
        (DOCS_BUILD_HTML / "examples" / "index.html", "single_tree_leaf_regression.html"),
    ):
        html = _read(index_path)
        assert page_name in html

    generated = _read(DOCS_BUILD_HTML / "examples" / "single_tree_leaf_regression.html")
    assert "leaf count:" in generated


def test_examples_docs_are_discoverable_from_toctrees():
    assert "examples/index" in _toctree_entries(TOP_LEVEL_INDEX)
    assert _toctree_entries(DOCS_EXAMPLES / "index.rst") == [
        "feature_engineering",
        "tree_to_graph",
        "tree_io",
        "single_task_training",
        "toml_training_config",
        "complete_pipeline",
        "extant_trait_regression",
        "single_tree_leaf_regression",
    ]
    for name in _toctree_entries(DOCS_EXAMPLES / "index.rst"):
        assert (DOCS_EXAMPLES / f"{name}.rst").is_file()


def test_examples_docs_map_to_runnable_files():
    pages = {
        path.stem: _read(path) for path in DOCS_EXAMPLES.glob("*.rst") if path.name != "index.rst"
    }

    for name in (
        "feature_engineering",
        "tree_to_graph",
        "tree_io",
        "single_task_training",
        "toml_training_config",
        "complete_pipeline",
        "extant_trait_regression",
    ):
        assert f"examples/{name}.py" in pages[name]
        assert ".. literalinclude::" in pages[name]
        assert ":language: python" in pages[name]

    assert "examples/toml_training_config.toml" in pages["toml_training_config"]
    assert str(TOML_CHECKPOINT.relative_to(ROOT)) in pages["toml_training_config"]
    assert str(TOML_HISTORY.relative_to(ROOT)) in pages["toml_training_config"]
    assert str(TOML_CHECKPOINT.relative_to(ROOT)) in pages["complete_pipeline"]


def test_example_docs_cover_inputs_actions_outputs_failure_modes_and_options():
    required_terms = (
        "Inputs",
        "Run command",
        "Expected output",
        "Files written",
        "Optional dependencies",
        "Failure modes",
        "Source",
    )
    pages = [path for path in DOCS_EXAMPLES.glob("*.rst") if path.name != "index.rst"]

    covered = 0
    total = len(required_terms) * len(pages)
    for page in pages:
        text = _read(page)
        covered += sum(term in text for term in required_terms)

    assert covered / total >= 0.9


def test_generated_examples_pages_are_reachable():
    examples_index = _read(DOCS_BUILD_HTML / "examples" / "index.html")

    for name in (
        "feature_engineering",
        "tree_to_graph",
        "tree_io",
        "single_task_training",
        "toml_training_config",
        "complete_pipeline",
        "extant_trait_regression",
    ):
        assert f"{name}.html" in examples_index


def test_generated_examples_pages_include_runtime_markers():
    text = "\n".join(
        _read(DOCS_BUILD_HTML / "examples" / name)
        for name in (
            "feature_engineering.html",
            "tree_to_graph.html",
            "tree_io.html",
            "single_task_training.html",
            "toml_training_config.html",
            "complete_pipeline.html",
            "extant_trait_regression.html",
        )
    )

    assert re.search(r"Feature engineering summary", text)
    assert re.search(r"Graph summary", text)
    assert re.search(r"Tree I/O summary", text)
    assert re.search(r"Training summary", text)
    assert re.search(r"TOML training run summary", text)
    assert re.search(r"Complete pipeline summary", text)
    assert "checkpoint: example_outputs/toml_training_config/final_model.pt" in text


def test_extant_trait_regression_docs_match_trainer_workflow():
    """Build the page and verify its source and generated contract details."""
    completed = subprocess.run(
        [
            "sphinx-build",
            "-b",
            "html",
            str(DOCS_SOURCE),
            str(DOCS_BUILD_HTML),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    source = _read(DOCS_EXAMPLES / "extant_trait_regression.rst")
    generated = _read(DOCS_BUILD_HTML / "examples" / "extant_trait_regression.html")
    for text in (source, generated):
        lowered = text.lower()
        assert "trainingconfig" in lowered
        assert "trainer" in lowered
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            assert mask_name in lowered
        assert "best_model.pt" in lowered
        assert "torch.expm1" in lowered
        assert "mse" in lowered
        assert "r2" in lowered
        for marker in (
            "Extant trait regression summary",
            "train/val/test nodes:",
            "test MSE:",
            "test R2:",
            "checkpoint:",
            "loss plot:",
            "scatter plot:",
        ):
            assert marker.lower() in lowered
        for output_path in (
            "example_outputs/extant_trait_regression_best.pt",
            "example_outputs/extant_trait_regression_loss.png",
            "example_outputs/extant_trait_regression_scatter.png",
        ):
            assert output_path in text


def test_quickstart_references_runnable_training_smoke_test():
    quickstart = _read(DOCS_SOURCE / "quickstart.rst")

    assert "python examples/quickstart_training.py" in quickstart
    assert "../../examples/quickstart_training.py" in quickstart
    assert "Quickstart training summary" in quickstart
    assert "prediction:" in quickstart


def test_metric_tracking_documentation_contracts():
    tracking = _read(DOCS_SOURCE / "user_guide" / "metrics_tracking.rst")
    config = _read(DOCS_SOURCE / "user_guide" / "training_config.rst")
    toml_example = _read(DOCS_EXAMPLES / "toml_training_config.rst")
    leaf = _read(DOCS_SOURCE / "user_guide" / "leaf_regression.rst")
    training_reference = _read(DOCS_SOURCE / "reference" / "training.rst")
    leaf_reference = _read(DOCS_SOURCE / "reference" / "leaf_regression.rst")
    leaf_example = _read(DOCS_EXAMPLES / "single_tree_leaf_regression.rst")
    leaf_example_source = _read(EXAMPLES_DIR / "single_tree_leaf_regression.py")
    troubleshooting = _read(DOCS_SOURCE / "troubleshooting.rst")

    for metric_name in (
        "train/loss",
        "train/lr",
        "train/epoch_time_sec",
        "val/loss",
        "final/best_val_loss",
        "final/best_epoch",
        "cv/fold_score",
        "cv/validation_leaf_count",
        "cv/mean_score",
        "cv/weighted_score",
        "cv/std_score",
        "cv/min_score",
        "cv/max_score",
        "cv/mae",
        "cv/pearson_r",
        "train/score",
        "val/score",
        "train/mae",
        "val/mae",
        "train/pearson_r",
        "val/pearson_r",
    ):
        assert metric_name in tracking

    assert "TrackingConfig(metrics=None)" in tracking
    assert "TrackingConfig(metrics=())" in tracking
    assert (
        'TrackingConfig(metrics=("train/loss", "train/score", "val/loss", "val/score"))' in tracking
    )
    assert "Operational fields" in tracking
    assert "status/state" in tracking
    assert 'metrics = ["train/loss", "val/loss"]' in config
    assert "metrics = []" in config
    assert "tracking.metrics" in config
    assert 'metrics = ["train/loss", "val/loss"]' in toml_example

    normalized_leaf = " ".join(leaf.split())
    for term in (
        "population standard deviation",
        "out-of-fold",
        "cv/mae",
        "cv/pearson_r",
        "RuntimeWarning",
        "omitted",
    ):
        assert term in normalized_leaf

    for term in (
        "whole epoch",
        "train/score",
        "val/score",
        "train/mae",
        "val/mae",
        "train/pearson_r",
        "val/pearson_r",
        "no validation loader",
        "all ``val/*``",
        "fewer than two paired observations",
        "zero variance in either input",
        "once per run and partition",
        "RuntimeWarning",
    ):
        assert term in normalized_leaf

    for term in (
        "epoch prediction and target inputs",
        "train/loss",
        "val/loss",
        "train/score",
        "val/score",
        "train/mae",
        "val/mae",
        "train/pearson_r",
        "val/pearson_r",
        "TrackingError",
        "scalar-only",
    ):
        assert term in leaf_reference

    for term in (
        "TrackingConfig",
        "train/score",
        "val/score",
        "train/mae",
        "val/mae",
        "train/pearson_r",
        "val/pearson_r",
        "train-only",
        "scalar-only",
    ):
        assert term in leaf_example
        assert term in leaf_example_source

    assert "metrics" in training_reference
    assert "quantitative" in training_reference
    assert "tracking_config" in leaf_reference
    assert "metric selection" in leaf_reference

    for term in (
        "before tracker start",
        "duplicate",
        "secret",
        "lazy",
        "not uploaded",
    ):
        assert term in troubleshooting.lower()
