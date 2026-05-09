# Contributing

## Example and Configuration Validation Notes

The curated examples are intended to stay aligned with the public package API.
When changing TOML training configuration behavior, keep regression coverage
for missing files, malformed TOML, missing required sections and model
dimensions, unknown keys, unsupported model, loss, and metric names, duplicate
metric names, wrong value types, and invalid numeric ranges.
