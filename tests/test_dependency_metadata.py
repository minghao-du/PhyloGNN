"""Audit package dependency metadata against repository import evidence."""

from __future__ import annotations

import ast
import re
import sys
import sysconfig
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
EVIDENCE_ROOTS = (
    REPO_ROOT / "src" / "phylognn",
    REPO_ROOT / "tests",
    REPO_ROOT / "examples",
    REPO_ROOT / "docs" / "source",
)
README = REPO_ROOT / "README.md"

IMPORT_TO_DISTRIBUTION = {
    "black": "black",
    "dendropy": "dendropy",
    "ete3": "ete3",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pytest": "pytest",
    "ruff": "ruff",
    "sphinx": "sphinx",
    "sphinx_rtd_theme": "sphinx-rtd-theme",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torchmetrics": "torchmetrics",
    "torch_scatter": "torch-scatter",
    "tqdm": "tqdm",
    "wandb": "wandb",
}

LOCAL_IMPORT_ROOTS = {"examples", "phylognn", "tests"}
OUT_OF_SCOPE_IMPORTS = {"setuptools"}


@dataclass(frozen=True)
class AuditClassification:
    distribution_name: str
    classification: str
    profile: str | None
    rationale: str = ""


AUDIT_CLASSIFICATIONS = {
    "black": AuditClassification(
        "black", "development", "dev", "Formatting tool advertised by the dev extra."
    ),
    "dendropy": AuditClassification(
        "dendropy",
        "optional",
        "beast",
        "Lazy tree I/O dependency for BEAST/NEXUS file workflows.",
    ),
    "ete3": AuditClassification("ete3", "core", "default"),
    "entmax": AuditClassification(
        "entmax",
        "core",
        "default",
        "Maintained sparse-attention normalization dependency for the default model.",
    ),
    "matplotlib": AuditClassification(
        "matplotlib",
        "optional",
        "examples",
        "Plotting dependency for the extant trait regression example.",
    ),
    "numpy": AuditClassification(
        "numpy",
        "core",
        "default",
        "Retained runtime dependency from repository guidance and existing public contract.",
    ),
    "pandas": AuditClassification(
        "pandas",
        "aggregate-only",
        "all",
        "Retained as an optional workflow helper in the aggregate user workflow extra.",
    ),
    "pytest": AuditClassification(
        "pytest", "development", "dev", "Test runner advertised by the dev extra."
    ),
    "ruff": AuditClassification(
        "ruff", "development", "dev", "Linting tool advertised by the dev extra."
    ),
    "sphinx": AuditClassification(
        "sphinx", "documentation", "docs", "Documentation builder advertised by the docs extra."
    ),
    "sphinx-rtd-theme": AuditClassification(
        "sphinx-rtd-theme",
        "documentation",
        "docs",
        "Documentation theme advertised by the docs extra.",
    ),
    "torch": AuditClassification("torch", "core", "default"),
    "torch-geometric": AuditClassification("torch-geometric", "core", "default"),
    "torchmetrics": AuditClassification(
        "torchmetrics",
        "core",
        "default",
        "Trainer metric lifecycle depends on TorchMetrics Metric objects.",
    ),
    "torch-scatter": AuditClassification(
        "torch-scatter",
        "core",
        "default",
        "Public GATBiLSTMNet import path uses torch_scatter.scatter.",
    ),
    "tqdm": AuditClassification(
        "tqdm", "core", "default", "Public Trainer progress bar dependency."
    ),
    "wandb": AuditClassification(
        "wandb", "optional", "wandb", "Lazy experiment tracking dependency."
    ),
}

EXPECTED_PROFILES = {
    "default": {
        "entmax",
        "ete3",
        "numpy",
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torchmetrics",
        "tqdm",
    },
    "beast": {"dendropy"},
    "wandb": {"wandb"},
    "docs": {"sphinx", "sphinx-rtd-theme"},
    "dev": {"black", "pytest", "ruff"},
    "examples": {"matplotlib"},
    "all": {"dendropy", "matplotlib", "pandas", "wandb"},
}

NEW_DEPENDENCY_VERSION_BOUNDS = {
    "entmax": ">=1.3",
    "torch-scatter": ">=2.1.0",
    "torchmetrics": ">=1.0.0",
    "tqdm": ">=4.65.0",
}


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)


def _requirement_distribution(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _profile_requirements(profile: str) -> list[str]:
    project = _load_pyproject()["project"]
    if profile == "default":
        return list(project["dependencies"])
    return list(project["optional-dependencies"][profile])


def _profile_distributions(profile: str) -> set[str]:
    return {
        _requirement_distribution(requirement) for requirement in _profile_requirements(profile)
    }


def _declared_distributions() -> set[str]:
    declared = set(_profile_distributions("default"))
    optional_dependencies = _load_pyproject()["project"]["optional-dependencies"]
    for profile in optional_dependencies:
        declared.update(_profile_distributions(profile))
    return declared


def _stdlib_roots() -> set[str]:
    roots = set(sys.builtin_module_names)
    if sys.stdlib_module_names:
        roots.update(sys.stdlib_module_names)
    roots.add(Path(sysconfig.get_path("stdlib")).name)
    return roots


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in EVIDENCE_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _import_roots_from_ast(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.partition(".")[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            roots.add(node.args[0].value.partition(".")[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_modules"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    roots.add(arg.value.partition(".")[0])
        elif (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "import_module")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            roots.add(node.args[0].value.partition(".")[0])
    return roots


def discovered_import_roots() -> set[str]:
    roots: set[str] = set()
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        roots.update(_import_roots_from_ast(tree))
    return roots


def discovered_third_party_imports() -> set[str]:
    stdlib_roots = _stdlib_roots()
    discovered = set()
    for root in discovered_import_roots():
        if root in stdlib_roots or root in LOCAL_IMPORT_ROOTS or root in OUT_OF_SCOPE_IMPORTS:
            continue
        discovered.add(root)
    return discovered


def _text_workflow_evidence() -> set[str]:
    evidence = set()
    for path in [
        README,
        *(REPO_ROOT / "docs" / "source").rglob("*.rst"),
        *(REPO_ROOT / "examples").rglob("*.md"),
    ]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in ("beast", "dendropy", "wandb", "sphinx", "docs", "dev"):
            if token in text:
                evidence.add(token)
    return evidence


def test_default_dependencies_cover_public_runtime_import_contract():
    assert _profile_distributions("default") == EXPECTED_PROFILES["default"]
    assert {
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torchmetrics",
        "ete3",
        "tqdm",
        "numpy",
    }.issubset(_profile_distributions("default"))

    for import_name, distribution_name in IMPORT_TO_DISTRIBUTION.items():
        classification = AUDIT_CLASSIFICATIONS.get(distribution_name)
        if classification and classification.profile == "default":
            assert distribution_name in _profile_distributions("default"), import_name


def test_import_name_mappings_are_explicit_for_non_matching_distributions():
    assert IMPORT_TO_DISTRIBUTION["torch_geometric"] == "torch-geometric"
    assert IMPORT_TO_DISTRIBUTION["torch_scatter"] == "torch-scatter"


def test_discovered_third_party_imports_have_audit_classifications():
    discovered_distributions = {
        IMPORT_TO_DISTRIBUTION.get(import_name, import_name.replace("_", "-"))
        for import_name in discovered_third_party_imports()
    }

    assert discovered_distributions <= set(AUDIT_CLASSIFICATIONS)


def test_declared_dependencies_have_profile_declarations_and_retention_rationales():
    discovered_distributions = {
        IMPORT_TO_DISTRIBUTION.get(import_name, import_name.replace("_", "-"))
        for import_name in discovered_third_party_imports()
    }
    for distribution_name in sorted(_declared_distributions()):
        classification = AUDIT_CLASSIFICATIONS[distribution_name]
        assert distribution_name in EXPECTED_PROFILES[classification.profile]
        if distribution_name not in discovered_distributions:
            assert classification.rationale


def test_optional_profiles_match_audited_workflow_boundaries():
    for profile, expected_distributions in EXPECTED_PROFILES.items():
        assert _profile_distributions(profile) == expected_distributions

    assert _profile_distributions("dev").isdisjoint(_profile_distributions("all"))
    assert _profile_distributions("docs").isdisjoint(_profile_distributions("all"))
    assert _profile_distributions("default").isdisjoint(_profile_distributions("all"))
    assert {"beast", "dendropy", "wandb", "sphinx", "docs", "dev"} <= _text_workflow_evidence()


def test_new_default_runtime_dependencies_keep_expected_lower_bounds():
    requirements = {
        _requirement_distribution(item): item for item in _profile_requirements("default")
    }
    for distribution_name, version_bound in NEW_DEPENDENCY_VERSION_BOUNDS.items():
        assert requirements[distribution_name] == f"{distribution_name}{version_bound}"


def test_package_metadata_avoids_platform_specific_installation_assumptions():
    metadata = PYPROJECT.read_text(encoding="utf-8").lower()
    forbidden_patterns = [
        "--extra-index-url",
        "--find-links",
        "pip install",
        "python -m pip",
        "cu118",
        "cu121",
        "cu124",
        "cuda",
        "http://",
        "https://",
        ".whl",
    ]

    assert not [pattern for pattern in forbidden_patterns if pattern in metadata]
