"""Release-facing smoke tests for public contracts."""

import importlib
import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        ("phylognn", {"TreeFeatureEngineer", "TreeToGraphConverter", "TrainingConfig", "Trainer"}),
        ("phylognn.data", {"TreeFeatureEngineer", "TreeToGraphConverter"}),
        ("phylognn.models", {"BasePhyloGNN", "GATBiLSTMNet"}),
        (
            "phylognn.training",
            {
                "ConfiguredTrainingSetup",
                "Trainer",
                "TrainingConfig",
                "TrainingConfigError",
                "create_trainer_from_config",
                "load_training_config",
            },
        ),
        ("phylognn.io", {"TreeReadConfig", "read_tree_as_ete3"}),
        ("phylognn.utils", {"get_max_meta_time"}),
    ],
)
def test_curated_facades_keep_expected_names(module_name, expected_names):
    """Release validation should confirm curated facades still expose supported names."""
    module = importlib.import_module(module_name)

    assert expected_names.issubset(set(module.__all__))


@pytest.mark.parametrize(
    ("module_name", "hidden_name"),
    [
        ("phylognn", "read_tree_as_ete3"),
        ("phylognn.data", "read_tree_as_ete3"),
        ("phylognn.models", "GATBlock"),
        ("phylognn.training", "all"),
    ],
)
def test_release_contracts_reject_hidden_names(module_name, hidden_name):
    """Release validation should detect leaked internal or optional names."""
    module = importlib.import_module(module_name)

    with pytest.raises(AttributeError):
        getattr(module, hidden_name)


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def test_dependency_profiles_match_release_metadata_contract():
    """Release metadata should keep runtime, workflow, docs, and dev profiles separate."""
    with PYPROJECT.open("rb") as file:
        project = tomllib.load(file)["project"]

    dependencies = {_distribution_name(item) for item in project["dependencies"]}
    optional = {
        profile: {_distribution_name(item) for item in requirements}
        for profile, requirements in project["optional-dependencies"].items()
    }

    assert dependencies == {
        "ete3",
        "numpy",
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torchmetrics",
        "tqdm",
    }
    assert optional["beast"] == {"dendropy"}
    assert optional["wandb"] == {"wandb"}
    assert optional["docs"] == {"sphinx", "sphinx-rtd-theme"}
    assert "furo" not in optional["docs"]
    assert optional["dev"] == {"black", "pytest", "ruff"}
    assert optional["all"] == {"dendropy", "matplotlib", "pandas", "wandb"}


def test_release_metadata_rejects_installer_specific_dependency_assumptions():
    """Package metadata must stay portable across platforms and accelerators."""
    metadata = PYPROJECT.read_text(encoding="utf-8").lower()

    assert "--extra-index-url" not in metadata
    assert "--find-links" not in metadata
    assert "pip install" not in metadata
    assert "python -m pip" not in metadata
    assert ".whl" not in metadata
    assert "cuda" not in metadata
    assert not re.search(r"\bcu\d{3}\b", metadata)
