# Changelog

## 0.8.2 - 2025-08-18

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Breaking changes

- Refactor `tide_aliasing` function to use `constituent` param name instead of `c`, use a list of default major tide constituents from `pyTMD`, remove "type" column, and set a 10 year default max on period values by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/118

#### Bug fixes

- Fix bug with specifying custom list of constituents in #117 by upgrading `pyTMD`

#### Other changes

- Add spell check to pre-commit, minor formatting updates by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/116
- Minor updates and upgrades to tests and `ruff`, `uv` versions by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/118

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.8.1...0.8.2

## 0.8.1 - 2025-07-17

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Other changes

* Add new EO satellite tide aliasing function for evaluating potential temporal biases in EO analyses in https://github.com/GeoscienceAustralia/eo-tides/pull/113
* Add `jupyter` to notebook optional dependencies to make Jupyter Notebooks easier to run in https://github.com/GeoscienceAustralia/eo-tides/pull/112

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.8.0...0.8.1

## 0.8.0 - 2025-06-24

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

This release provides new functionality to customise tide modelling:

1. A new `extra_databases` parameter to model tides using models that are not natively supported by `pyTMD`, accepting custom tide model databases in either Python dictionary or JSON file format
2. A new `constituents` parameter to restrict tide modelling to a custom subset of harmonic constituents

For example, to model tides using a custom `EOT20_custom` tide model:

```
import pandas as pd
from eo_tides.model import model_tides

custom_db_dict = {
  "elevation": {
    "EOT20_custom": {
      "format": "FES-netcdf",
      "model_file": [
        "EOT20/ocean_tides/2N2_ocean_eot20.nc",
        "EOT20/ocean_tides/J1_ocean_eot20.nc",
        "EOT20/ocean_tides/K1_ocean_eot20.nc",
        "EOT20/ocean_tides/K2_ocean_eot20.nc",
        "EOT20/ocean_tides/M2_ocean_eot20.nc",
        "EOT20/ocean_tides/M4_ocean_eot20.nc",
        "EOT20/ocean_tides/MF_ocean_eot20.nc",
        "EOT20/ocean_tides/MM_ocean_eot20.nc",
        "EOT20/ocean_tides/N2_ocean_eot20.nc",
        "EOT20/ocean_tides/O1_ocean_eot20.nc",
        "EOT20/ocean_tides/P1_ocean_eot20.nc",
        "EOT20/ocean_tides/Q1_ocean_eot20.nc",
        "EOT20/ocean_tides/S1_ocean_eot20.nc",
        "EOT20/ocean_tides/S2_ocean_eot20.nc",
        "EOT20/ocean_tides/SA_ocean_eot20.nc",
        "EOT20/ocean_tides/SSA_ocean_eot20.nc",
        "EOT20/ocean_tides/T2_ocean_eot20.nc"
      ],
      "name": "EOT20_custom",
      "reference": "https://doi.org/10.17882/79489",
      "scale": 0.01,
      "type": "z",
      "variable": "tide_ocean",
      "version": "EOT20"
    }
  }
}

model_tides(
    x=148,
    y=-16,
    time=pd.date_range("2022-01-01", "2023-12-31", freq="1h"),
    model=["EOT20_custom", "EOT20"],
    directory="/var/share/tide_models/",
    extra_databases=custom_db_dict,
    output_format="wide",
)



```
#### New features

* Support custom tide models by passing in extra tide model databases by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/105
* Support customising constituents during tide modelling by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/108

#### Other changes

* Major Ruff refactor by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/108

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.7.5...0.8.0

## 0.7.5 - 2025-06-23

### What's Changed

Minor update to remove Dask pin now that Dask compatability issue has been solved in `odc-stac`

#### Documentation updates

* Fix unlinked URLs in changelog by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/103

#### Other changes

* Remove dask pin to close #76 by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/106

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.7.4...0.7.5

## 0.7.4 - 2025-05-30

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New features

* Add version attribute to init file by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/98

#### Documentation updates

* Fix capitalisation in JOSS paper bibliography by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/97
* Update suggested citation to use new JOSS paper citation by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/100

#### Other changes

* Update PR labelling and release template by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/99
* Reformat code with additional `ruff` linting rules by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/101

### New Contributors

* @github-actions made their first contribution in https://github.com/GeoscienceAustralia/eo-tides/pull/96

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.7.3...0.7.4

## 0.7.3 - 2025-05-22

### Changes

- Use dynamic version handling via `hatch-vcs`, add automatic changelog update action by @robbibt in https://github.com/GeoscienceAustralia/eo-tides/pull/95
- Bump the python-deps group with 2 updates by @dependabot in https://github.com/GeoscienceAustralia/eo-tides/pull/94

**Full Changelog**: https://github.com/GeoscienceAustralia/eo-tides/compare/0.7.2...0.7.3

## 0.7.2 - 2025-05-19

### New features

- Minor updates to improve documentation around accessing GESLA 3.0 tide gauge data,
- Made validation functions more re-usable by removing hard-coded paths and adding helpful error messages.

## 0.7.1 - 2025-05-19

Minor update to package dependencies

## 0.7.0 - 2025-05-14

### New features

- This version corresponds to the code archived as part of [Journal of Open Source Software publication](https://joss.theoj.org/papers/b5680c39bf831c1159c41a2eb7ec9c5e).
  No new features introduced.

## 0.6.5 - 2025-05-08

### New features

- Support for `pyTMD` versions 2.2.3 and 2.2.4

## 0.6.4 - 2025-02-31

### New features

- Updated installation documentation to improve reproducibility

### Bug fixes

- Temporarily pinned Dask to avoid `odc-geo` and `odc-stac` errors

## 0.6.3 - 2025-03-20

### New features

- Major updates to installation and setup documentation, to provide additional context about tide modelling data and installing `eo-tides` using both `pip` and `uv`

### Bug fixes

- Fixed bug where ensemble tide modelling used excessive memory, by ensuring dtype of ensemble modelled tides matches dtype of input modelled tides ([#70](https://github.com/GeoscienceAustralia/eo-tides/pull/70))
- Added missing `dask` dependency to requirements ([#68](https://github.com/GeoscienceAustralia/eo-tides/pull/68))

## 0.6.2 - 2025-02-25

### New features

- Added `apply_node` parameteter to `model_tides` to apply `pyTMD`'s adjustments to harmonic constituents to allow for periodic modulations over the 18.6-year nodal period (lunar nodal tide). Default is False.

### Bug fixes

- Further fixes for bug causing tide model clipping with `clip_tides` to fail for bounding boxes completely west of the prime meridian ([#50](https://github.com/GeoscienceAustralia/eo-tides/issues/50)); default value for `crop` param is now `"auto"` instead of `True`.

## 0.6.1 - 2025-02-20

### Bug fixes

- Fixed bug causing tide model clipping with `clip_tides` to fail for bounding boxes completely west of the prime meridian ([#50](https://github.com/GeoscienceAustralia/eo-tides/issues/50))

## 0.6.0 - 2025-02-11

### New features

- Added `return_phases` parameter to `eo_tides.eo.tag_tides`, which will return a dataframe containing tide phase information for each satellite observation
- Added support for [Technical University of Denmark tide models](https://doi.org/10.11583/DTU.23828874) (DTU23)
- Minor docs improvements, updates for new FES2022 data format

## 0.5.0 - 2025-01-17

### New features

- Added draft version of a Journal of Open Source Software paper
- Added benchmarking notebook that compares performance with parallelisation on and off

### Bug fixes

- Fix documentation to point to correct location of `list_models` function (e.g. `eo_tides.utils.list_models`)

### Breaking changes

- Removed Python 3.9 support
- Added Python 3.13 support

## 0.4.0 - 2025-12-21

### New features

- Publishes ensemble tide modelling code for combining multiple global ocean tide models into a single locally optimised ensemble tide model using external model ranking data (e.g. satellite altimetry or NDWI-tide correlations along the coastline).
  
  - Update ensemble code to latest version that includes FES2022, GOT5.6 and TPXO10 tide models
  - Make ensemble model calculation function a top level function (i.e. rename from `_ensemble_model` to `ensemble_tides`)
  - Load tide model ranking points from external `flatgeobuf` format file for faster cloud access
  
- Major refactor to statistics functions to standardise code across both `pixel_stats` and `tide_stats` and add support for multiple models
  
  - `tide_stats` will now return a `pandas.Series` if one model is requested, and a `pandas.DataFrame` if multiple are requested
  - Added a new `point_col` parameter to `tide_stats` to control the colour of plotted points. If `plot_var` is also provided, points will now be coloured differently by default.
  
- Added a new `crop_buffer` parameter to configure buffer distance when cropping model files with `crop=True` (defaults to 5 degrees)
  
- Reorder `model_tides` parameters to provide more logical flow and move more common params like `mode`, `output_format` and `output_units` higher
  

### Bug fixes

- Fix warnings from `load_gauge_gesla` function

### Breaking changes

- The `plot_col` parameter from `tide_stats` has been renamed to `plot_var`

## 0.3.1 - 2024-11-15

### New features

- Add new "all" option to `model` param in `model_tides`, `pixel_tides` etc, which will model tides using all available tide models in your provided `directory`.

### Bug fixes

- Fix bug where GOT5.6 was not detected as a valid model because it contains files in multiple directories (e.g. both "GOT5.6" and "GOT5.5"). This also affected clipping GOT5.6 data using the `eo_tides.utils.clip_models` function.

## 0.3.0 - 2024-11-11

### New features

- Added new `eo_tides.utils.clip_models` function for clipping tide models to a smaller spatial extent. This can have a major positive impact on performance, sometimes producing more than a 10 x speedup. This function identifies all NetCDF-format tide models in a given input directory, including "ATLAS-netcdf" (e.g. `TPXO9-atlas-nc`), "FES-netcdf" (e.g. `FES2022`, `EOT20`), and "GOT-netcdf" (e.g. `GOT5.5`) format files. Files for each model are then clipped to the extent of the provided bounding box, handling model-specific file structures. After each model is clipped, the result is exported to the output directory and verified with `pyTMD` to ensure the clipped data is suitable for tide modelling.

![image](https://github.com/user-attachments/assets/7b9ffab7-2614-4d04-9799-e56500ab810c)

### Major changes

- The `parallel_splits` parameter that controls the number of chunks data is broken into for parallel analysis has been refactored to use a new default of "auto". This now attempts to automatically determine a sensible value based on available CPU, number of points, and number of models being run. All CPUs will be used where possible, unless this will produce splits with less than 1000 points in each (which would increase overhead). Parallel splits will be reduced if multiple models are requested, as these are run in parallel too and will compete for the same resources.
- Changed the default interpolation `method` from "spline" to "linear". This appears to produce the same results, but works considerably faster.
- Updates to enable correct cropping, recently resolved in PyTMD 2.1.8

### Breaking changes

- The `list_models` function has been relocated to `eo_tides.utils` (from `eo_tides.model`)

## 0.2.0 - 2024-10-30

### New features

- New `model_phases` function for calculating tidal phases ("low-flow", high-flow", "high-ebb", "low-ebb") for each tide height in a timeseries. Ebb and low phases are calculated by running the `eo_tides.model.model_tides` function twice, once for the requested timesteps, and again after subtracting a small time offset (by default, 15 minutes). If tides increased over this period, they are assigned as "flow"; if they decreased, they are assigned as "ebb". Tides are considered "high" if equal or greater than 0 metres tide height, otherwise "low".
- Major refactor to use consistent input parameters across all EO focused functions: input can now be either `xr.DataArray` or `xr.Dataset` or `odc.geo.geobox.GeoBox`; if an xarray object is passed, it must have a `"time"` dimension; if GeoBox is passed, time must be provided by the `time` parameter.
- `time` parameters now accept any format that can be converted by `pandas.to_datetime()`; e.g. np.ndarray[datetime64], pd.DatetimeIndex, pd.Timestamp, datetime.datetime and strings (e.g. "2020-01-01 23:00").
- `model_tides` now uses default cropping approach from `pyTMD`, rather than applying a bespoke 1 degree buffer around the selected analysis area
- `model_tides` refactored to use simpler approach to loading tide consistuents enabled in `pyTMD==2.1.7`

### Breaking changes

- The `ds` param in all satellite data functions (`tag_tides`, `pixel_tides`, `tide_stats`, `pixel_tides`) has been renamed to a more generic name `data` (to account for now accepting either `xarray.Dataset`, `xarray.DataArray` or a `odc.geo.geobox.GeoBox` inputs).

## 0.1.0 - 2024-10-18

### New features

- Initial creation of `eo-tides` repo

### Breaking changes

See [Migrating from DEA Tools](migration.md) for a guide to updating your code from the original [`Digital Earth Australia Notebooks and Tools` repository](https://github.com/GeoscienceAustralia/dea-notebooks/).
