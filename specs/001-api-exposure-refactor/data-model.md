# Data Model: API Exposure Refactor

## StableApiLayer

- Purpose: Defines the small set of top-level `phylognn` entry points intended
  for ordinary users.
- Fields:
  - `export_names`: ordered collection of supported root-package names
  - `version_name`: public package version attribute name
  - `workflow_scope`: description of which user workflows this layer covers
- Validation rules:
  - Every exported name MUST resolve to a real symbol.
  - The order of exported names MUST remain deterministic.
  - The layer MUST exclude low-level implementation helpers.
- Relationships:
  - References symbols implemented in `DataApiLayer`, `ModelsApiLayer`, and
    `TrainingApiLayer`.

## DataApiLayer

- Purpose: Curated public API for feature engineering, graph conversion, and
  optional tree I/O.
- Fields:
  - `primary_exports`: stable data-pipeline names intended for routine use
  - `optional_io_exports`: explicitly separated tree-loading names in
    `phylognn.io`
  - `feature_metadata_contract`: read-only exposure of feature name metadata
  - `virtual_node_flag_name`: canonical public name for the virtual-node marker
- Validation rules:
  - Primary data-pipeline imports MUST work without optional tree I/O coupling
    in the root package.
  - Public feature metadata exposed to users MUST be deterministic and
    read-only where specified.
  - The virtual-node marker name MUST be consistent across docs and outputs.
- Relationships:
  - Supplies symbols to `StableApiLayer`.
  - Shares graph-field contracts with the converter and downstream models.

## ModelsApiLayer

- Purpose: Curated model-facing API for end-user and advanced modeling imports.
- Fields:
  - `base_exports`: intentionally supported abstract or reusable model bases
  - `primary_model_exports`: end-user model classes
  - `internal_layer_exports`: low-level layer and head classes kept at explicit
    module paths only
- Validation rules:
  - Package-level model exports MUST contain only supported user-facing or
    intentionally public advanced symbols.
  - Internal layer classes MUST NOT be advertised as stable package-level API
    unless deliberately promoted.
- Relationships:
  - Supplies primary models to `StableApiLayer`.
  - Depends on internal modules such as `layers.py` and `multitask.py`.

## TrainingApiLayer

- Purpose: Curated training-facing API for datasets, trainer configuration,
  trainer orchestration, and public metric helpers.
- Fields:
  - `dataset_exports`: split-aware dataset types intended for users
  - `trainer_exports`: trainer classes and factory helpers
  - `metric_exports`: public evaluation helpers
  - `export_list_name`: canonical package-level export declaration
- Validation rules:
  - All listed exports MUST exist.
  - Nonexistent names MUST NOT appear in the public training contract.
  - Public metric and factory helpers intended for user workflows MUST be
    surfaced intentionally.
- Relationships:
  - Supplies trainer objects to `StableApiLayer`.
  - Shares validation evidence with package-level import tests.

## PublicMetadataContract

- Purpose: Represents user-visible metadata and naming conventions exposed by
  the package.
- Fields:
  - `package_version_attr`
  - `feature_names_view`
  - `available_features_view`
  - `canonical_public_names`
- Validation rules:
  - Public metadata MUST use conventional, stable naming.
  - Metadata views exposed to users MUST not allow accidental mutation when the
    public contract is intended to be read-only.
  - Canonical names MUST match documentation, examples, and tests.
- Relationships:
  - Applies across `StableApiLayer` and `DataApiLayer`.
