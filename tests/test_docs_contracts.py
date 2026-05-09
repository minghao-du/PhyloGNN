"""Regression tests for documentation and runnable-example contracts."""

from __future__ import annotations

from pathlib import Path
import re
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
    "feature_engineering",
    "graph_conversion",
    "training",
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


def test_examples_docs_are_discoverable_from_toctrees():
    assert "examples/index" in _toctree_entries(TOP_LEVEL_INDEX)
    assert _toctree_entries(DOCS_EXAMPLES / "index.rst") == [
        "toml_training_config",
        "complete_pipeline",
    ]
    assert (DOCS_EXAMPLES / "toml_training_config.rst").is_file()
    assert (DOCS_EXAMPLES / "complete_pipeline.rst").is_file()


def test_examples_docs_map_to_runnable_files():
    toml_page = _read(DOCS_EXAMPLES / "toml_training_config.rst")
    pipeline_page = _read(DOCS_EXAMPLES / "complete_pipeline.rst")

    assert "examples/toml_training_config.py" in toml_page
    assert "examples/toml_training_config.toml" in toml_page
    assert "examples/complete_pipeline.py" in pipeline_page
    assert str(TOML_CHECKPOINT.relative_to(ROOT)) in toml_page
    assert str(TOML_HISTORY.relative_to(ROOT)) in toml_page
    assert str(TOML_CHECKPOINT.relative_to(ROOT)) in pipeline_page


def test_example_docs_cover_inputs_actions_outputs_failure_modes_and_options():
    required_terms = ("Inputs", "Actions", "Expected outputs", "Failure modes", "Optional settings")
    pages = [
        DOCS_EXAMPLES / "toml_training_config.rst",
        DOCS_EXAMPLES / "complete_pipeline.rst",
    ]

    covered = 0
    total = len(required_terms) * len(pages)
    for page in pages:
        text = _read(page)
        covered += sum(term in text for term in required_terms)

    assert covered / total >= 0.9


def test_generated_examples_pages_are_reachable():
    examples_index = _read(DOCS_BUILD_HTML / "examples" / "index.html")

    assert "toml_training_config.html" in examples_index
    assert "complete_pipeline.html" in examples_index


def test_generated_examples_pages_include_runtime_markers():
    text = "\n".join(
        _read(DOCS_BUILD_HTML / "examples" / name)
        for name in ("toml_training_config.html", "complete_pipeline.html")
    )

    assert re.search(r"TOML training run summary", text)
    assert re.search(r"Complete pipeline summary", text)
    assert "checkpoint: example_outputs/toml_training_config/final_model.pt" in text
