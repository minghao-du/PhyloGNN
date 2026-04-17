# Coverage Mapping: Critical Test Coverage

## Tier Definitions

| Tier | Scope | Minimum Expectation | Validation Commands |
|------|-------|---------------------|---------------------|
| Tier 1 | `phylognn.data.*`, `phylognn.models.*`, `phylognn.training.*` | Success-path, failure-path, and regression coverage for scientific/data contracts | `pytest tests/test_feature_engineer.py tests/test_data_conversion.py tests/test_model_contracts.py tests/test_training_components.py tests/test_data_contracts.py` |
| Tier 2 | `phylognn`, `phylognn.data`, `phylognn.models`, `phylognn.training`, `phylognn.io`, `phylognn.utils` | Curated export, lazy import, optional boundary, and facade ownership coverage | `pytest tests/test_public_api.py tests/test_training_api.py tests/test_models_api.py tests/test_io_api.py tests/test_utils_api.py tests/test_release_contracts.py` |

## Coverage Targets

| Target | Tier | Public Contract | Test Locations | Scenario Coverage | Exception |
|--------|------|-----------------|----------------|-------------------|-----------|
| `phylognn` | Tier 2 | Root package exports only the curated public surface and stable version metadata | `tests/test_public_api.py`, `tests/test_release_contracts.py` | `__all__`, lazy exports, hidden-name exclusion, invalid attr access | |
| `phylognn.data` | Tier 2 | Data facade exposes only `TreeFeatureEngineer` and `TreeToGraphConverter` | `tests/test_public_api.py`, `tests/test_release_contracts.py` | Curated export surface and lazy resolution | |
| `phylognn.data.feature_engineer` | Tier 1 | Feature metadata is stable, input validation fails fast, feature attachment is deterministic | `tests/test_feature_engineer.py` | Metadata views, selective feature addition, invalid origin/features | |
| `phylognn.data.converter` | Tier 1 | Converter preserves feature ordering, validates inputs, and produces stable graph metadata | `tests/test_data_conversion.py`, `tests/test_data_contracts.py` | Virtual-node feature naming, empty-tree rejection, deterministic node names/metadata | |
| `phylognn.data.tree_io` | Tier 1 | Optional tree I/O validates config and fails clearly when files or extras are missing | `tests/test_io_api.py`, `tests/test_release_contracts.py` | Optional dependency boundary, config validation, lazy import contract | |
| `phylognn.io` | Tier 2 | Optional tree I/O stays behind a dedicated facade and is lazily resolved | `tests/test_public_api.py`, `tests/test_io_api.py`, `tests/test_release_contracts.py` | `__all__`, `__dir__`, invalid attr access, no leakage to root/data facades | |
| `phylognn.models` | Tier 2 | Model facade exports supported model classes and hides low-level layers | `tests/test_public_api.py`, `tests/test_models_api.py`, `tests/test_release_contracts.py` | Curated export surface and hidden helper exclusion | |
| `phylognn.models.base` | Tier 1 | Base model validation rejects malformed graph inputs and preserves encoder/head utilities | `tests/test_model_contracts.py` | `validate_data`, freezing/unfreezing, embedding dimension contract | |
| `phylognn.models.layers` | Tier 1 | Layer builders validate constructor arguments and expose stable output-dimension behavior | `tests/test_model_contracts.py` | Constructor validation and repr/output-dim assertions | |
| `phylognn.models.gat_lstm` | Tier 1 | Temporal GAT model validates constructor args and temporal requirements explicitly | `tests/test_model_contracts.py` | Invalid temporal mode, missing `num_time_bins`, graph-pool validation | |
| `phylognn.models.multitask` | Tier 1 | Multi-task model returns named outputs and preserves time-bin pooling contract | `tests/test_model_contracts.py` | Task-name handling, output-key contract, pooling shape expectations | |
| `phylognn.training` | Tier 2 | Training facade exports dataset splits, trainer utilities, metrics, and factory helpers | `tests/test_training_api.py`, `tests/test_release_contracts.py` | Curated `__all__`, hidden-name exclusion, export presence | |
| `phylognn.training.dataset` | Tier 1 | Dataset helpers validate split structure and preserve deterministic indexing rules | `tests/test_training_components.py` | Split validation helpers and sample-id contracts | |
| `phylognn.training.metrics` | Tier 1 | Public metrics remain available and compute stable scalar outputs | `tests/test_training_components.py` | Metric output semantics and exported function availability | |
| `phylognn.training.trainer` | Tier 1 | Training config and trainer helpers fail fast on invalid config/runtime assumptions | `tests/test_training_components.py` | `TrainingConfig.validate`, task-name sanitization, scalar-detach helpers | |
| `phylognn.utils` | Tier 2 | Utilities facade exposes only supported helper names | `tests/test_utils_api.py`, `tests/test_release_contracts.py` | `__all__` contract and invalid attr access | |
| `phylognn.utils.tree_utils` | Tier 1 | Utility helper returns deterministic max metadata value and handles missing metadata safely | `tests/test_utils_api.py` | Positive path and empty-metadata path | |

## Current Gap Notes

- Existing repository coverage already protects parts of the public facade, feature metadata, converter virtual-node naming, and training facade exports.
- Missing or incomplete ownership before this feature included dedicated tests for `phylognn.io`, `phylognn.utils`, core model modules, training component internals, and release-level scientific/data contracts.
- `tests/support.py` and enriched `tests/conftest.py` provide shared fixtures so new coverage can stay focused on contracts rather than duplicated setup.
