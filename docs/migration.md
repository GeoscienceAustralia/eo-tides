# Migrating from `dea_tools`

The `eo-tides` package contains functions that were previously available in the [`Digital Earth Australia Notebooks and Tools` repository](https://github.com/GeoscienceAustralia/dea-notebooks/).
To migrate your code from `dea-tools` to `eo-tides`, please be aware of the following breaking changes:

## Breaking changes

### Tide model directory environment variable updated

The `DEA_TOOLS_TIDE_MODELS` environmental variable has been renamed to `EO_TIDES_TIDE_MODELS`.
