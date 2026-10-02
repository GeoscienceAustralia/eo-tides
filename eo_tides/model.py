"""Core tide modelling functionality.

This module provides tools for modelling ocean tide heights and phases
for any location or time period using one or more global tide models.
"""

# Used to postpone evaluation of type annotations
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Only import if running type checking
if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Sequence


import geopandas as gpd
import numpy as np
import pandas as pd
import pyTMD
import timescale.time
import xarray as xr
from odc.geo.geom import multipoint
from tqdm import tqdm

from .utils import (
    DEFAULT_ENSEMBLE_FUNCS,
    DEFAULT_ENSEMBLE_MODELS,
    DatetimeLike,
    _set_directory,
    _standardise_models,
    _standardise_time,
    idw,
)


def _load_model_datatree(
    models,
    directory,
    time,
    bounds,
    crop_buffer=5,
    append_node=False,
    output_units="m",
    constituents=None,
    extra_databases=None,
):
    out = {}

    for m in models:
        # Load model file and reduce constituents
        extra_databases = [] if extra_databases is None else extra_databases
        pytmd_model = pyTMD.io.model(
            directory=directory,
            extra_databases=extra_databases,
        ).from_database(m, group="z")

        # Lazy load tide model data and append correction data
        ds = pytmd_model.open_dataset(
            append_node=append_node,
            constituents=constituents,
            chunks="auto",
        )

        # Convert units
        ds = ds.tmd.to_units(output_units)

        # Append with corrections info
        ds.attrs["corrections"] = pytmd_model.corrections
        out[m] = ds

    # Create data tree with time as a dimension
    time_coords = xr.Dataset(coords={"time": time})
    dt = xr.DataTree.from_dict({"/": time_coords, **out})

    # Crop to area of interest and then load into memory
    dt = dt.tmd.crop(bounds=bounds, buffer=crop_buffer)

    # Load into memory and return
    return dt.compute()


def _interp_const(ds, x, y, mode, crs=4326, **interp_kwargs):
    """Interpolate tidal constituents for a dataset or datatree node.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing tidal constituents.
    x : array_like
        X coordinates for interpolation.
    y : array_like
        Y coordinates for interpolation.
    mode : str
        Method to use to prepare coordinates in interpolation.
        Accepts "one-to-many", "one-to-one" or "grid" (default).
    crs : int, optional
        Coordinate reference system EPSG code. Default is 4326.
    **interp_kwargs
        Additional keyword arguments passed to `ds.tmd.interp`
        to control the constituent interpolation. For example:
        "extrapolate", "cutoff", "k", "power", "workers".

    Returns
    -------
    xarray.Dataset
        Dataset containing the interpolated constituents.

    """
    # Skip structural parent nodes that contain no data variables
    if not ds.data_vars:
        return ds

    # Prepare coordinates differently based on analysis method;
    # use "grid" by default
    type_mapping = {"one-to-many": "time series", "one-to-one": "trajectory"}
    coord_type = type_mapping.get(mode, "grid")

    # Prepare coordinates for interpolation
    x_coords, y_coords = ds.tmd.coords_as(x, y, type=coord_type, crs=crs)

    # Interpolate constituents at coordinates
    return ds.tmd.interp(x_coords, y_coords, **interp_kwargs)


def _predict_tides_dataset(ds, time):
    """Apply map_blocks to predict tides for each interpolated tidal constituent dataset.

    For example, this function can be applied to each model
    node in an xarray.DataTree.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing interpolated tidal constituents
        and time coordinates.
    time : DatetimeLike
        One or more UTC times at which to model tide heights, as
        an array of datatimes.

    Returns
    -------
    tides : xarray.Dataset
        Predicted tide values.

    """
    # Skip structural parent nodes that contain no data variables
    if not ds.data_vars:
        return ds

    # Add required tide times, connections and deltat
    ts = timescale.from_datetime(time)
    deltat = np.zeros_like(ts.tide) if ds.corrections in ("OTIS", "ATLAS", "TMD3") else ts.tt_ut1
    ds = ds.assign_coords(
        tide_time=("time", ts.tide),
        deltat=("time", deltat),
    )

    # Predict tides for each chunk in the dataset
    return xr.map_blocks(
        _predict_tides_chunk,
        ds,
    ).to_dataset(name="tide_height")


def _predict_tides_chunk(ds):
    """Model tides for each chunk in an interpolated tidal constituent dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing interpolated tidal
        constituents and time coordinates (including
        "tide_time", "corrections" and "deltat"
        coordinates and attributes).

    Returns
    -------
    tides : xarray.Dataset
        Predicted tide values.

    """
    # Calculate tides using the perfectly sized float times
    tides = (
        ds.drop_vars("time")
        .tmd.predict(
            ds.tide_time,
            corrections=ds.corrections,
            deltat=ds.deltat,
        )
        .astype("float32")
    )
    tides += ds.drop_vars("time").tmd.infer(
        ds.tide_time,
        corrections=ds.corrections,
        deltat=ds.deltat,
    )

    # Re-apply datetimes and drop tide time, deltat
    return tides.assign_coords(time=ds.time).drop_vars(["tide_time", "deltat"])


def _optimise_chunks(time_length, mode="grid", target_mb=32, output_dtype=np.float32):
    """Calculate optimal chunk sizes for xarray objects based on analysis mode.

    Parameters
    ----------
    time_length : int
        Number of expected observations along the time dimension.
    mode : str, optional
        Analysis mode determining the chunking strategy.
        Options are "grid", "one-to-many", or "one-to-one". Default is "grid".
    target_mb : int, optional
        Target chunk size in megabytes. Default is 32.
    output_dtype : type, optional
        Data type of the output array. Default is np.float32.

    Returns
    -------
    dict
        Dictionary mapping dimension names to their optimal chunk sizes.

    """
    # Convert target memory size to bytes
    target_bytes = target_mb * (1024**2)

    # Determine memory footprint of a single array element
    element_bytes = np.dtype(output_dtype).itemsize

    # Handle one-to-one mode where chunking occurs along time
    if mode == "one-to-one":
        max_time = target_bytes / element_bytes
        chunk_size = max(int(np.floor(max_time)), 1)
        return {"time": chunk_size}

    # Determine memory footprint per spatial pixel or station across all timesteps
    bytes_per_unit = time_length * element_bytes

    # Handle one-to-many mode where chunking occurs along stations
    if mode == "one-to-many":
        max_stations = target_bytes / bytes_per_unit
        chunk_size = max(int(np.floor(max_stations)), 1)
        return {"station": chunk_size}

    # Handle grid mode where chunking occurs across x and y
    if mode == "grid":
        max_pixels = target_bytes / bytes_per_unit
        chunk_size = max(int(np.floor(np.sqrt(max_pixels))), 1)
        return {"x": chunk_size, "y": chunk_size}

    err_msg = f"Unsupported method: {mode}"
    raise Exception(err_msg)


# TODO: Fix nodata gaps when run with extrapolation=False
def ensemble_tides(
    ds,
    crs="EPSG:4326",
    ensemble="ensemble",
    ensemble_models=None,
    ensemble_func=None,
    ensemble_top_n=3,
    ensemble_stat="median",
    ranking_points="https://dea-public-data-dev.s3-ap-southeast-2.amazonaws.com/derivative/dea_intertidal/supplementary/rankings_ensemble_2017-2019.fgb",
    ranking_valid_perc=0.02,
    **idw_kwargs,
):
    """Combine multiple tide models into a single locally optimised ensemble.

    Uses external model ranking data (e.g. satellite altimetry or
    NDWI-tide correlations along the coastline) to inform the
    selection of the best local tide models.

    This function performs the following steps:

    1. Takes a dataset of tide heights from multiple tide models (with
       models specified along the "tide_model" dimension).
    2. Loads model ranking points from an external file, filters them
       based on the valid data percentage, and retains relevant columns.
    3. Interpolates the model rankings into the coordinates of the
       original dataframe using Inverse Weighted Interpolation (IDW).
    4. Uses rankings to combine multiple tide models into a single
       optimised ensemble model (by default, by taking the median of the
       top 3 ranked models).
    5. Returns a new dataset containing the combined ensemble model
       predictions.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset containing a "tide_height" variable and multiple
        models specified along the "tide_model" dimension.
    crs : str, optional
        Coordinate reference system used to re-project ranking
        points data. This should match the CRS of `ds`.
        Defaults to "EPSG:4326" (degrees latitude, longitude).
    ensemble : list or str, optional
        The ensemble(s) to generate. Several options are supported
        by default (for custom ensembles, provide a dictionary
        of functions to `ensemble_func`):

        - `"ensemble"`: Default method, combining the top N models at
        each location (`ensemble_top_n`) using the statistic
        `ensemble_stat` (e.g. median, mean etc). Equivalent to
        "ensemble-median-top3" with default settings.
        - `"ensemble-mean"`: Mean of all input models
        - `"ensemble-median"`: Median of all input models
        - `"ensemble-top"`: Tides from the locally top-ranked model
        - `"ensemble-bottom"`: Tides from the locally bottom-ranked model
        - `"ensemble-mean-top3"`: Mean of top 3 locally ranked models
        - `"ensemble-median-top3"`: Median of top 3 locally ranked models

    ensemble_models : list
        A list of input models to include in ensemble modelling
        (e.g. models to be combined to create multi-model ensembles).
        All values must exist as columns with the prefix "rank_" in
        `ranking_points`.
    ensemble_func : dict, optional
        Can be used to provide custom ensemble generation functions.
        Functions should take `ds` as an input, and combine multiple models
        using an xarray.DataArray of interpolated `ranks`, e.g.:
        `{
        "ensemble-median": lambda x, ranks=None, **kw: x.median(dim="tide_model"),
        "ensemble-top": lambda x, ranks, **kw: x.where(ranks == 1).mean(dim="tide_model"),
        "ensemble-mean-top5": lambda x, ranks, **kw: x.where(ranks <= 5).mean(dim="tide_model")
        }`
        Dictionary keys are used to name output ensembles.
    ensemble_top_n : int, optional
        For the default `ensemble="ensemble"`, this sets the number of top
        models to include in the aggregation calculation. Defaults to 3.
    ensemble_stat : str, optional
        For the default `ensemble="ensemble"`, this sets the method used
        to aggregate multiple models. Defaults to "median", supports "mean".
    ranking_points : str, optional
        Path to the file containing model ranking points. This dataset
        should include columns containing rankings for each tide
        model, named with the prefix "rank_". e.g. "rank_EOT20".
        Low values should represent high rankings (e.g. 1 = top ranked).
        The default value points to an example file covering Australia.
    ranking_valid_perc : float, optional
        Minimum percentage of valid data required to include a model
        rank point in the analysis, as defined in a column named
        "valid_perc". Defaults to 0.02.
    **idw_kwargs
        Optional keyword arguments to pass to the `idw` function used
        for interpolation. Useful values include `k` (number of nearest
        neighbours to use in interpolation), `max_dist` (maximum
        distance to nearest neighbours), and `k_min` (minimum number of
        neighbours required after `max_dist` is applied).

    Returns
    -------
    xarray.Dataset
        An dataset with a "tide_height" variable and a "tide_model"
        dimension that contains each requested ensemble model prediction.
        The default ensemble will be named "ensemble" (if custom
        ensemble functions are provided via `ensemble_func`, each
        ensemble will be named using the provided dictionary keys).

    """
    # Cast to list of strings for consistent handling
    ensemble = [str(m) for m in np.atleast_1d(ensemble)]

    # If no ensemble inputs are defined, use default list
    if ensemble_models is None:
        ensemble_models = DEFAULT_ENSEMBLE_MODELS

    # If custom functions provided, join them with default list
    ensemble_func = {} if ensemble_func is None else ensemble_func
    ensemble_func = ensemble_func | DEFAULT_ENSEMBLE_FUNCS

    # Verify all ensembles are members of combined function dict
    missing_ensemble = set(ensemble) - set(ensemble_func)
    if missing_ensemble:
        err_msg = (
            f"The following requested `ensemble` is not defined: {sorted(missing_ensemble)}\n"
            "Please provide a custom `ensemble_func` dictionary."
        )
        raise Exception(err_msg)

    # Broadcast x and y so they share spatial dimensions across all modes
    x_spatial, y_spatial = xr.broadcast(ds.x, ds.y)

    # Extract spatial dimensions and shape from spatial coordinates
    spatial_dims = list(x_spatial.dims)
    spatial_shape = [ds.sizes[d] for d in spatial_dims]

    # Flatten spatial coordinates for IDW interpolation
    x_interp = x_spatial.to_numpy().ravel()
    y_interp = y_spatial.to_numpy().ravel()

    # Filter coordinates to those linked to spatial dims
    spatial_coords = {k: v for k, v in ds.coords.items() if set(v.dims).issubset(spatial_dims)}

    # Load model ranks and filter to required data
    model_ranking_cols = [f"rank_{m}" for m in ensemble_models]
    model_ranks_gdf = (
        gpd.read_file(ranking_points, engine="pyogrio")
        .to_crs(crs)
        .query(f"valid_perc > {ranking_valid_perc}")
        .dropna(how="all")
        .filter(model_ranking_cols + ["geometry"])  # noqa: RUF005
    )

    # Calculate model rankings
    ranks = (
        # Interpolate model rankings and re-shape to match input
        # data and total number of models
        xr.DataArray(
            idw(
                input_z=model_ranks_gdf[model_ranking_cols],
                input_x=model_ranks_gdf.geometry.x,
                input_y=model_ranks_gdf.geometry.y,
                output_x=x_interp,
                output_y=y_interp,
                **idw_kwargs,
            ).reshape(*spatial_shape, len(ensemble_models)),
            dims=[*spatial_dims, "tide_model"],
            coords={
                **spatial_coords,
                "tide_model": ensemble_models,
            },
        )
        # Re-rank to account for missing models
        .rank(dim="tide_model")
        .astype("float32")
    )

    # Create output list to hold computed ensemble model outputs
    ensemble_list = []

    # Loop through all provided ensemble generation functions
    for ensemble_i in ensemble:
        # Get ensemble function from function dictionary
        ensemble_f = ensemble_func[ensemble_i]

        # Apply function with runtime variables
        ensemble_model_ds = ensemble_f(
            ds,
            ranks=ranks,
            top_n=ensemble_top_n,
            stat=ensemble_stat,
        ).expand_dims({"tide_model": [ensemble_i]})

        ensemble_list.append(ensemble_model_ds)

    # Concatenate into a single ensemble output
    return xr.concat(ensemble_list, dim="tide_model")


# TODO: Sort out "crop" param functionality
# TODO: Restore parallel flag for dask processing?
def model_tides(
    x: float | Sequence[float] | xr.DataArray,
    y: float | Sequence[float] | xr.DataArray,
    time: DatetimeLike,
    model: str | Iterable[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    crs: str = "EPSG:4326",
    mode: str = "one-to-many",
    tide_stage: bool = False,
    output_format: str = "long",
    output_units: str = "m",
    output_nodata: float = -32768,
    method: str = "linear",
    extrapolate: bool = True,
    extrapolate_k: int = 1,
    cutoff: float | None = None,
    crop: bool | str = "auto",  # noqa: ARG001
    crop_buffer: float | None = 5,
    append_node: bool = False,
    constituents: list[str] | None = None,
    dask_chunks: str | dict | None = "auto",
    extra_databases: str | os.PathLike | list | None = None,
    **ensemble_kwargs,
) -> pd.DataFrame | xr.Dataset:
    """Model tide heights and tide stages across coordinates and timesteps using multiple ocean tide models.

    Calculates modelled tide heights (and high/low/ebb/flow tide stages)
    for input spatial coordinates and timesteps using one or more global
    ocean tide models. Calculations are parallelised for performance and
    support point-in-time (`"one-to-one"`), time-series (`"one-to-many"`),
    and spatial grid (`"grid"`) analysis modes, returning results as
    either a pandas DataFrame or xarray Dataset.

    Supports all ocean tide models supported by `pyTMD`, including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg Assimilation Methods for Tides (HAMTIDE11)
    - Technical University of Denmark models (DTU23)

    Additional custom "ensemble" models are also supported; see:
    https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.model.ensemble_tides

    Requires local tide model data. For setup instructions, see:
    https://geoscienceaustralia.github.io/eo-tides/setup/

    Based on the `pyTMD` package's `pyTMD.compute.tide_elevations` function:
    https://pytmd.readthedocs.io/en/latest/api_reference/compute.html#pyTMD.compute.tide_elevations

    Parameters
    ----------
    x : float or sequence of floats
        One or more x coordinates at which to model tides. Assumes
        degrees longitude (EPSG:4326) by default; use `crs` to specify
        a different coordinate reference system.
    y : float or sequence of floats
        One or more y coordinates at which to model tides. Assumes
        degrees latitude (EPSG:4326) by default; use `crs` to specify
        a different coordinate reference system.
    time : DatetimeLike
        One or more UTC times at which to model tide heights. Accepts
        any time format compatible with `pandas.to_datetime()`, e.g.
        datetime.datetime, pd.Timestamp, pd.DatetimeIndex, numpy.datetime64,
        or date/time strings (e.g. "2020-01-01 23:00"). For example:
        `time = pd.date_range(start="2000", end="2001", freq="5h")`.
    model : str or iterable of str, optional
        The tide model (or list of models) to use to model tides.
        Defaults to "EOT20"; specify "all" to use all models available
        in `directory`. For a full list of available and supported models,
        run `from eo_tides.utils import list_models; list_models()`.
        Ensemble tide modelling can also be requested by passing either
        "ensemble", or any of the ensemble options supported by
        `from eo_tides.model import ensemble_tides`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    crs : str, optional
        Input coordinate reference system for x/y coordinates.
        Defaults to "EPSG:4326" (degrees latitude, longitude).
    mode : str, optional
        Tide modelling analysis mode, which determines how spatial
        coordinates and timesteps are combined. Supports three options:

        - `"one-to-one"`: Models tides using one timestep per x/y coordinate.
          Output length is `len(time)`; requires `len(x) == len(y) == len(time)`.
        - `"one-to-many"`: Models tides for every x/y coordinate across
          every timestep in `time`. Output length will be `len(x) * len(time)`;
          requires `len(x) == len(y)`.
        - `"grid"`: Models tides across a 2D grid defined by 1D `x` and `y`
          coordinate axes, for every timestep in `time`. Output size is
          `len(x) * len(y) * len(time)`.

    tide_stage : bool, optional
            Whether to calculate an additional "tide_stage" variable/column
            classifying each modelled tide observation into four categories:

            - `"high-ebb"`: High (>= 0 m) receding tide
            - `"low-ebb"`: Low (< 0 m) receding tide
            - `"low-flow"`: Low (< 0 m) rising tide
            - `"high-flow"`: High (>= 0 m) rising tide

    output_format : str, optional
        Whether to return an output pandas.Dataframe in "long" format
        (a single "tide_height" column stacked vertically by tide model,
        time, x and y), a pandas.Dataframe in "wide" format (with a
        column for each tide model), or as an xarray.Dataset ("xarray")
        with a "tide_height" variable. Defaults to "long".
    output_units : str, optional
        Units for the returned tide heights. Options are:

        - `"m"`: Metres as "float32" floating-point values (default).
        - `"cm"`: Centimetres as "int16" integers (scaled by 100).
        - `"mm"`: Millimetres as "int16" integers (scaled by 1000).

        Using integer units can help reduce memory usage.
    output_nodata : int, optional
        Nodata fill value used when integer units ("cm" or "mm") are
        selected. Because integer data types cannot store `NaN`, `NaN`
        values are replaced with this value prior to casting to `int16`.
        Ignored when `output_units="m"`.
    method : str, optional
        Method used to interpolate tide model constituents to the requested
        x/y coordinates. Supports "linear" (default) and "nearest".
    extrapolate : bool, optional
        Whether to extrapolate tides inland of the valid tide model
        extent. This can ensure tide are returned everywhere, but
        accuracy will degrade with distance (e.g. inland or along complex
        estuaries or rivers). Set `cutoff` to define the maximum
        extrapolation distance, and `extrapolate_k` to set how many
        nearest neighbours will be used for extrapolation.
    extrapolate_k: int, optional
        Number of nearest neighbours to use for extrapolation. The
        default of 1 will use Nearest Neighbour interpolation, and higher
        values will use Inverse Distance Weighted (IDW) interpolation.
    cutoff : float, optional
        Maximum distance in kilometres to extrapolate tides inland of the
        valid tide model extent. The default of None allows extrapolation
        at any (i.e. infinite) distance.
    crop : bool or str, optional
        Whether to crop tide model files on-the-fly to improve performance.
        Defaults to "auto", which enables cropping when supported (some
        clipped model files limited to the western hemisphere may not support
        on-the-fly cropping). Use `crop_buffer` to adjust the buffer
        distance used for cropping.
    crop_buffer : int or float, optional
        The buffer distance in degrees to crop tide model files around the
        requested x/y coordinates. Defaults to 5, which will crop model
        files using a five degree buffer.
    append_node : bool, optional
        Apply adjustments to harmonic constituents to allow for periodic
        modulations over the 18.6-year nodal period (lunar nodal tide).
        Default is False.
    constituents : list, optional
        Optional list of tide constituents to use for tide prediction,
        e.g. `constituents=["m2", "s2"]`. Default is None, which will use
        all available constituents.
    dask_chunks: str or dict, optional
        Dask chunking used to model tides in parallel. The default
        "auto" will attempt to automatically determine optimal
        chunking based on `mode` and the number of timesteps in `time`.
        Supports custom Dask chunks provided as a dictionary, e.g.
        `dask_chunks={"x": 20, "y": 20}`.
    extra_databases : str or path or list, optional
        Additional custom tide model definitions to load, provided as
        dictionaries or paths to JSON database files. Use this to
        enable custom tide models not included with `pyTMD`.
        See: https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#model-database
    **ensemble_kwargs :
        Keyword arguments used to customise the generation of optional
        ensemble tide models if ensemble tide modelling is requested.
        These are passed to the underlying `_ensemble_model` function.
        Useful parameters include `ensemble_models` (what input tide
        models to use for ensemble generation), `ensemble_func` (custom
        ensemble functions), `ranking_points` (path to model
        rankings data), `k` (for controlling how model rankings are
        interpolated), and `ensemble_top_n` (how many top models to use
        in the ensemble calculation).

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        A pandas.Dataframe containing modelled tide heights
        ("tide_height") and optionally tide stages ("tide_stage")
        if `output_format=="long"` or `output_format=="wide"`, or
        an xarray.Dataset if `output_format=="xarray"`.

    """
    # Turn inputs into arrays for consistent handling
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = _standardise_time(time)

    # Warn and remove deprecated execution arguments
    deprecated_args = ["parallel", "parallel_splits", "parallel_max"]
    for arg in deprecated_args:
        if arg in ensemble_kwargs:
            ensemble_kwargs.pop(arg)
            warnings.warn(
                f"The '{arg}' argument is deprecated and will be removed in a future release. Please use `dask_chunks` instead.",
                category=FutureWarning,
                stacklevel=2,
            )

    # Validate input arguments
    if time is None:
        err_msg = "Times for modelling tides must be provided via `time`."
        raise ValueError(err_msg)

    if method not in ("linear", "nearest"):
        err_msg = f"Invalid interpolation method '{method}'. Must be 'linear', or 'nearest'."
        raise ValueError(err_msg)

    if output_units not in ("m", "cm", "mm"):
        err_msg = "Output units must be either 'm', 'cm', or 'mm'."
        raise ValueError(err_msg)

    if output_format not in ("long", "wide", "xarray"):
        err_msg = "Output format must be either 'long', 'wide', or 'xarray'."
        raise ValueError(err_msg)

    if not np.issubdtype(x.dtype, np.number):
        err_msg = "`x` must contain only valid numeric values, and must not be None."
        raise TypeError(err_msg)

    if not np.issubdtype(y.dtype, np.number):
        err_msg = "`y` must contain only valid numeric values, and must not be None."
        raise TypeError(err_msg)

    if mode in ("one-to-one", "one-to-many") and len(x) != len(y):
        err_msg = (
            "In 'one-to-one' or 'one-to-many' mode, `x` and `y` represent point "
            "coordinate pairs and must be the same length."
        )
        raise ValueError(err_msg)

    if mode == "one-to-one" and len(x) != len(time):
        err_msg = (
            "The number of supplied `x` and `y` points and `time` values must be "
            "identical in 'one-to-one' mode. Use 'one-to-many' mode if you intended "
            "to model multiple timesteps at each point."
        )
        raise ValueError(err_msg)

    # Set tide modelling files directory. If no custom path is
    # provided, try global environment variable.
    directory = _set_directory(directory)

    # Standardise model list, handling "all" and ensemble
    # functionality and any custom tide model definitions
    ensemble_models = ensemble_kwargs.get("ensemble_models")
    ensemble_func = ensemble_kwargs.get("ensemble_func")
    models_requested, models_to_process, ensemble_requested, ensemble_models = _standardise_models(
        model=model,
        directory=directory,
        ensemble_models=ensemble_models,
        ensemble_func=ensemble_func,
        extra_databases=extra_databases,
    )

    # Calculate cropping bounds from x and y coords
    mpt = multipoint(list(zip(x, y, strict=False)), crs=crs)
    bbox = mpt.boundingbox.to_crs("EPSG:4326")
    bounds = bbox.range_x + bbox.range_y
    bounds_str = f"({', '.join(f'{v:.2f}' for v in bounds)})"

    # Restructure x and y coords into 2D if in grid mode
    if mode == "grid":
        x, y = np.meshgrid(x, y)

    # Optionally calculate tide_stage by adding additional timesteps
    # after subtracting a 15 min time offset. These timesteps are used
    # to determine if tides are rising or falling at each observation;
    # they are discarded before any data is returned.
    if tide_stage:
        time_n = len(time)
        time = np.concatenate([time, time - pd.Timedelta("15min")])

        # Also repeat x and y coords in one-to-one mode
        if mode == "one-to-one":
            x = np.tile(x, 2)
            y = np.tile(y, 2)

    # Load model data as a datatree containing a time index
    print(f"Loading model files after cropping to {bounds_str} with a {crop_buffer}° buffer")
    dt = _load_model_datatree(
        models=models_to_process,
        directory=directory,
        time=time,
        bounds=bounds,
        crop_buffer=crop_buffer,
        append_node=append_node,
        output_units=output_units,
        constituents=constituents,
        extra_databases=extra_databases,
    )

    # Interpolate data with progress bar, ignoring parent node
    with tqdm(total=len(dt.children), desc="Interpolating model constituents") as pbar:

        def _interp_with_progress(ds, **kwargs):
            """Run interpolation with progress bar, skipping parent node."""
            result = _interp_const(ds, **kwargs)
            if ds.data_vars:
                pbar.update(1)
            return result

        interpolated = dt.map_over_datasets(
            _interp_with_progress,
            kwargs={
                "x": x,
                "y": y,
                "mode": mode,
                "crs": crs,
                "method": method,
                "extrapolate": extrapolate,
                "cutoff": np.inf if cutoff is None else cutoff,
                "k": extrapolate_k,
                "power": 2,
                "workers": -1,
            },
        )

    # Determine optimal chunking
    if dask_chunks == "auto":
        dask_chunks = _optimise_chunks(
            time_length=len(time),
            mode=mode,
        )
        print(f"Automatically determining Dask chunking: {dask_chunks}")
    interp_chunked = interpolated.chunk(dask_chunks)

    # Predict tides for each model and each chunk
    print("Modelling tides")
    tides = interp_chunked.map_over_datasets(
        _predict_tides_dataset,
        kwargs={"time": time},
    )
    tides.load()

    # Combine data tree nodes into a dataset
    models_combined = xr.concat(
        [child.ds for child in tides.children.values()],
        dim=pd.Index(tides.children.keys(), name="tide_model"),
        coords="minimal",
        compat="override",
    )

    # Optionally compute ensemble model and add to dataset
    if ensemble_requested:
        print("Combining modelled tides into multi-model ensemble")
        ensemble_ds = ensemble_tides(
            ds=models_combined,
            crs=crs,
            ensemble=ensemble_requested,
            ensemble_models=ensemble_models,
            ensemble_func=ensemble_func,
            **ensemble_kwargs,
        )

        # Combine ensemble with existing models
        models_combined = xr.concat([models_combined, ensemble_ds], dim="tide_model")

        # Return only requested models
        models_combined = models_combined.sel(tide_model=models_requested)

    # Add "tide_stage" variable to data if requested
    if tide_stage:
        # Extract preceding timesteps and drop coordinates to enable
        # direct comparison between timesteps
        pre = models_combined.isel(time=slice(time_n, None)).drop_vars("time")

        # Redefine data to include only target timesteps
        models_combined = models_combined.isel(time=slice(0, time_n))

        # Determine ebb/flow and high/low state; tides are ebbing
        # if preceding tides were higher than current tides.
        is_ebb = models_combined.tide_height < pre.tide_height
        is_high = models_combined.tide_height >= 0
        ebb_flow = xr.where(is_ebb, "ebb", "flow")
        high_low = xr.where(is_high, "high", "low")

        # Combine into tide_stage string and mask missing data
        tide_phase = high_low + "-" + ebb_flow
        valid_mask = models_combined.tide_height.notnull() & pre.tide_height.notnull()
        models_combined["tide_stage"] = xr.where(valid_mask, tide_phase, "")

    # Re-assign original coords due to tide modelling functions
    # returning coords in the native CRS of the model
    models_combined = models_combined.assign_coords(
        x=(models_combined.x.dims, x, models_combined.x.attrs),
        y=(models_combined.y.dims, y, models_combined.y.attrs),
    )

    # Simplify x and y dimensions if they are two-dimensional
    if mode == "grid" and models_combined.x.ndim == 2:
        models_combined = models_combined.assign_coords(
            y=models_combined.y.isel(x=0),
            x=models_combined.x.isel(y=0),
        )

    # Units and dtype conversion
    if output_units in ("cm", "mm"):
        models_combined = models_combined.fillna(output_nodata).astype("int16")
        models_combined["tide_height"].attrs["_FillValue"] = output_nodata

    # Rename station to site if it exists, then return xarray with CRS info
    if output_format == "xarray":
        print("Returning data in xarray format")
        if "station" in models_combined.dims:
            return models_combined.rename_dims({"station": "site"})
        return models_combined.odc.assign_crs(crs)

    # Convert to DataFrame, raising performance warning if large size
    size = models_combined.tide_height.size

    if size > 1_000_000:
        print("Converting to dataframe (this can be slow; for faster performance use `output_format='xarray'`).")
    else:
        print("Converting to dataframe")

    tide_df = models_combined.to_dataframe()

    # Move x and y to the index if they are data columns (one-to-one and one-to-many modes)
    index_cols = [col for col in ["x", "y"] if col in tide_df.columns]
    if index_cols:
        tide_df = tide_df.set_index(index_cols, append=True)

    # Drop internal spatial dimension from index as x and y uniquely identify locations
    if "station" in tide_df.index.names:
        tide_df = tide_df.droplevel("station")

    # Return long format pandas dataframe directly
    if output_format == "long":
        print("Returning data in 'long' format")
        return tide_df

    # Or convert to wide format by unstacking tide model index
    print("Returning data in 'wide' format")
    return tide_df.squeeze("columns").unstack("tide_model")


def model_phases(
    x: float | list[float] | xr.DataArray,
    y: float | list[float] | xr.DataArray,
    time: DatetimeLike,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    time_offset: str = "15 min",  # noqa: ARG001
    return_tides: bool = False,  # noqa: ARG001
    **model_tides_kwargs,
) -> pd.DataFrame:
    """Model tide phases at multiple coordinates or timesteps using ocean tide models.

    .. deprecated::
        `model_phases` is deprecated and will be removed in a future release.
        Please use `model_tides` with `tide_stage=True` directly instead.

    Parameters
    ----------
    x : float or list of floats
        One or more x coordinates at which to model tides. Assumes
        degrees longitude (EPSG:4326) by default; use `crs` to specify
        a different coordinate reference system.
    y : float or list of floats
        One or more y coordinates at which to model tides. Assumes
        degrees latitude (EPSG:4326) by default; use `crs` to specify
        a different coordinate reference system.
    time : DatetimeLike
        One or more UTC times at which to model tide heights.
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        Defaults to "EOT20".
    directory : str or PathLike, optional
        The directory containing tide model data files.
    time_offset : str, optional
        Unused parameter kept for backwards compatibility.
    return_tides : bool, optional
        Unused parameter kept for backwards compatibility.
    **model_tides_kwargs :
        Optional parameters passed directly to `model_tides`.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing modelled tides and tide stages.

    """
    warnings.warn(
        "`model_phases` is deprecated and will be removed in a future release. "
        "Please use the `tide_stage` functionality in `model_tides` directly.",
        category=FutureWarning,
        stacklevel=2,
    )

    # Delegate directly to model_tides using tide_stage
    return model_tides(
        x=x,
        y=y,
        time=time,
        model=model,
        directory=directory,
        tide_stage=True,
        **model_tides_kwargs,
    )
