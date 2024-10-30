# Changelog

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
