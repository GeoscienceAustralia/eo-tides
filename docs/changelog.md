# Changelog

## v0.3.1 (2024-11-15)

### New features

- Add new "all" option to `model` param in `model_tides`, `pixel_tides` etc, which will model tides using all available tide models in your provided `directory`.

### Bug fixes

- Fix bug where GOT5.6 was not detected as a valid model because it contains files in multiple directories (e.g. both "GOT5.6" and "GOT5.5"). This also affected clipping GOT5.6 data using the `eo_tides.utils.clip_models` function.

## v0.3.0 (2024-11-11)

### New features

- Added new `eo_tides.utils.clip_models` function for clipping tide models to a smaller spatial extent. This can have a major positive impact on performance, sometimes producing more than a 10 x speedup. This function identifies all NetCDF-format tide models in a given input directory, including "ATLAS-netcdf" (e.g. `TPXO9-atlas-nc`), "FES-netcdf" (e.g. `FES2022`, `EOT20`), and "GOT-netcdf" (e.g. `GOT5.5`) format files. Files for each model are then clipped to the extent of the provided bounding box, handling model-specific file structures. After each model is clipped, the result is exported to the output directory and verified with `pyTMD` to ensure the clipped data is suitable for tide modelling.

![image](https://github.com/user-attachments/assets/7b9ffab7-2614-4d04-9799-e56500ab810c)

### Major changes

- The `parallel_splits` parameter that controls the number of chunks data is broken into for parallel analysis has been refactored to use a new default of "auto". This now attempts to automatically determine a sensible value based on available CPU, number of points, and number of models being run. All CPUs will be used where possible, unless this will produce splits with less than 1000 points in each (which would increase overhead). Parallel splits will be reduced if multiple models are requested, as these are run in parallel too and will compete for the same resources.
- Changed the default interpolation `method` from "spline" to "linear". This appears to produce the same results, but works considerably faster.
- Updates to enable correct cropping, recently resolved in PyTMD 2.1.8

### Breaking changes

- The `list_models` function has been relocated to `eo_tides.utils` (from `eo_tides.model`)

## v0.2.0 (2024-10-30)

### New features

- New `model_phases` function for calculating tidal phases ("low-flow", high-flow", "high-ebb", "low-ebb") for each tide height in a timeseries. Ebb and low phases are calculated by running the `eo_tides.model.model_tides` function twice, once for the requested timesteps, and again after subtracting a small time offset (by default, 15 minutes). If tides increased over this period, they are assigned as "flow"; if they decreased, they are assigned as "ebb". Tides are considered "high" if equal or greater than 0 metres tide height, otherwise "low".
- Major refactor to use consistent input parameters across all EO focused functions: input can now be either `xr.DataArray` or `xr.Dataset` or `odc.geo.geobox.GeoBox`; if an xarray object is passed, it must have a `"time"` dimension; if GeoBox is passed, time must be provided by the `time` parameter.
- `time` parameters now accept any format that can be converted by `pandas.to_datetime()`; e.g. np.ndarray[datetime64], pd.DatetimeIndex, pd.Timestamp, datetime.datetime and strings (e.g. "2020-01-01 23:00").
- `model_tides` now uses default cropping approach from `pyTMD`, rather than applying a bespoke 1 degree buffer around the selected analysis area
- `model_tides` refactored to use simpler approach to loading tide consistuents enabled in `pyTMD==2.1.7`

### Breaking changes

- The `ds` param in all satellite data functions (`tag_tides`, `pixel_tides`, `tide_stats`, `pixel_tides`) has been renamed to a more generic name `data` (to account for now accepting either `xarray.Dataset`, `xarray.DataArray` or a `odc.geo.geobox.GeoBox` inputs).

## v0.1.0 (2024-10-18)

### New features

- Initial creation of `eo-tides` repo

### Breaking changes

See [Migrating from DEA Tools](migration.md) for a guide to updating your code from the original [`Digital Earth Australia Notebooks and Tools` repository](https://github.com/GeoscienceAustralia/dea-notebooks/).

<!-- ### Bug fixes -->
