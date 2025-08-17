"""Core tide modelling functionality.

This module provides tools for modelling ocean tide heights and phases
for any location or time period using one or more global tide models.
"""

# Used to postpone evaluation of type annotations
from __future__ import annotations

import os
import textwrap
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import TYPE_CHECKING

import psutil

# Only import if running type checking
if TYPE_CHECKING:
    import xarray as xr

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pyTMD
import timescale.time
from tqdm import tqdm

from .utils import (
    DatetimeLike,
    _set_directory,
    _standardise_models,
    _standardise_time,
    idw,
)


def _parallel_splits(
    total_points: int,
    model_count: int,
    parallel_max: int | None = None,
    min_points_per_split: int = 1000,
) -> int:
    """Calculate the optimal number of parallel splits for data processing.

    Optimal parallelisation is estimated based on system resources
    and processing constraints.

    Parameters
    ----------
    total_points : int
        Total number of data points to process
    model_count : int
        Number of models that will be run in parallel
    parallel_max : int, optional
        Maximum number of parallel processes to use. If None, uses CPU core count
    min_points_per_split : int, default=1000
        Minimum number of points that should be processed in each split

    """
    # Get available CPUs. First see if `CPU_GUARANTEE` exists in
    # environment (if running in JupyterHub); if not use psutil
    # followed by standard CPU count
    if parallel_max is None:
        # Take the first valid output
        raw_value = os.environ.get("CPU_GUARANTEE") or psutil.cpu_count(logical=False) or os.cpu_count() or 1

        # Convert to integer
        parallel_max = int(float(raw_value)) if isinstance(raw_value, str) else int(raw_value)

    # Calculate optimal number of splits based on constraints
    splits_by_size = total_points / min_points_per_split
    splits_by_cpu = parallel_max / model_count
    optimal_splits = min(splits_by_size, splits_by_cpu)

    # Convert to integer and ensure at least 1 split
    return int(max(1, optimal_splits))


def _model_tides(
    model,
    x,
    y,
    time,
    directory,
    crs,
    mode,
    output_units,
    method,
    extrapolate,
    cutoff,
    crop,
    crop_buffer,
    append_node,
    constituents,
    extra_databases,
):
    """Worker function applied in parallel by `model_tides`.

    Handles the extraction of tide modelling constituents and tide
    modelling using `pyTMD`.
    """
    # Load models from pyTMD database
    extra_databases = [] if extra_databases is None else extra_databases
    pytmd_model = pyTMD.io.model(
        directory=directory,
        extra_databases=extra_databases,
    ).elevation(model)

    # Reproject x, y to latitude/longitude
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())

    # Convert datetime
    ts = timescale.time.Timescale().from_datetime(time.flatten())

    try:
        # Read tidal constants and interpolate to grid points
        amp, ph, c = pytmd_model.extract_constants(
            lon,
            lat,
            type=pytmd_model.type,
            crop=True if crop == "auto" else crop,
            buffer=crop_buffer,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
            append_node=append_node,
            constituents=constituents,
            extra_databases=extra_databases,
        )

        # TODO: Return constituents
        # print(model, amp.shape)
        # print(amp.shape, ph.shape, c)
        # print(pd.DataFrame({"amplitude": amp}))

    except ValueError:
        # If on-the-fly cropping is auto, try again with crop turned off
        if crop == "auto":
            warnings.warn(
                "On-the-fly cropping is not compatible with the provided "
                "model files; running with `crop=False`. This will not "
                "affect your results but may lead to a minor slowdown. "
                "This can occur when analysing clipped model files restricted "
                "to the western hemisphere. To suppress this warning, manually "
                "set `crop=False`.",
                stacklevel=2,
            )

            # Read tidal constants and interpolate to grid points
            amp, ph, c = pytmd_model.extract_constants(
                lon,
                lat,
                type=pytmd_model.type,
                crop=False,
                buffer=crop_buffer,
                method=method,
                extrapolate=extrapolate,
                cutoff=cutoff,
                append_node=append_node,
                constituents=constituents,
                extra_databases=extra_databases,
            )

        # Otherwise, raise error if cropping if set to True
        else:
            error_msg = (
                "On-the-fly cropping (e.g. `crop=True`) is not compatible with your "
                "provided clipped model files. Please set `crop=False` or `crop='auto'`, "
                "or run your analysis on unclipped global model files to avoid this error."
            )
            raise Exception(error_msg) from None

    # Raise error if constituent files do not cover analysis extent
    except IndexError:
        error_msg = (
            f"The {model} tide model constituent files do not cover the analysis extent "
            f"({min(lon):.2f}, {max(lon):.2f}, {min(lat):.2f}, {max(lat):.2f}). "
            "This can occur if you are using clipped model files to improve run times. "
            "Consider using model files that cover your entire analysis area."
        )
        raise Exception(error_msg) from None

    # Calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # Calculate constituent oscillation
    hc = amp * np.exp(cph)

    # Compute delta times based on model
    deltat = np.zeros_like(ts.tt_ut1) if pytmd_model.corrections in ("OTIS", "ATLAS", "TMD3", "netcdf") else ts.tt_ut1

    # In "one-to-many" mode, extracted tidal constituents and timesteps
    # are repeated/multiplied out to match the number of input points and
    # timesteps, enabling the modeling of tides across all combinations
    # of input times and points. In "one-to-one" mode, no repetition is
    # needed, so each repeat count is set to 1.
    points_repeat = len(x) if mode == "one-to-many" else 1
    time_repeat = len(time) if mode == "one-to-many" else 1
    t, hc, deltat = (
        np.tile(ts.tide, points_repeat),
        hc.repeat(time_repeat, axis=0),
        np.tile(deltat, points_repeat),
    )

    # Create arrays to hold outputs
    tide = np.ma.zeros((len(t)), fill_value=np.nan)
    tide.mask = np.any(hc.mask, axis=1)

    # Predict tidal elevations at time
    tide.data[:] = pyTMD.predict.drift(
        t,
        hc,
        c,
        deltat=deltat,
        corrections=pytmd_model.corrections,
    )

    # Infer minor corrections
    minor = pyTMD.predict.infer_minor(
        t,
        hc,
        c,
        deltat=deltat,
        corrections=pytmd_model.corrections,
        minor=pytmd_model.minor,
    )
    tide.data[:] += minor.data[:]

    # Replace invalid values with fill value
    tide.data[tide.mask] = tide.fill_value

    # Convert data to pandas.DataFrame, and set index to our input
    # time/x/y values
    tide_df = pd.DataFrame(
        {
            "time": np.tile(time, points_repeat),
            "x": np.repeat(x, time_repeat),
            "y": np.repeat(y, time_repeat),
            "tide_model": model,
            "tide_height": tide,
        },
    ).set_index(["time", "x", "y"])

    # Optionally convert outputs to integer units (can save memory)
    if output_units == "m":
        tide_df["tide_height"] = tide_df.tide_height.astype(np.float32)
    elif output_units == "cm":
        tide_df["tide_height"] = (tide_df.tide_height * 100).astype(np.int16)
    elif output_units == "mm":
        tide_df["tide_height"] = (tide_df.tide_height * 1000).astype(np.int16)

    return tide_df


def ensemble_tides(
    tide_df,
    crs,
    ensemble_models,
    ensemble_func=None,
    ensemble_top_n=3,
    ranking_points="https://dea-public-data-dev.s3-ap-southeast-2.amazonaws.com/derivative/dea_intertidal/supplementary/rankings_ensemble_2017-2019.fgb",
    ranking_valid_perc=0.02,
    **idw_kwargs,
):
    """Combine multiple tide models into a single locally optimised ensemble tide model.

    Uses external model ranking data (e.g. satellite altimetry or
    NDWI-tide correlations along the coastline) to inform the
    selection of the best local models.

    This function performs the following steps:

    1. Takes a dataframe of tide heights from multiple tide models, as
       produced by `eo_tides.model.model_tides`
    2. Loads model ranking points from an external file, filters them
       based on the valid data percentage, and retains relevant columns
    3. Interpolates the model rankings into the coordinates of the
       original dataframe using Inverse Weighted Interpolation (IDW)
    4. Uses rankings to combine multiple tide models into a single
       optimised ensemble model (by default, by taking the mean of the
       top 3 ranked models)
    5. Returns a new dataframe with the combined ensemble model predictions

    Parameters
    ----------
    tide_df : pandas.DataFrame
        DataFrame produced by `eo_tides.model.model_tides`, containing
        tide model predictions in long format with columns:
        `["time", "x", "y", "tide_height", "tide_model"]`.
    crs : string
        Coordinate reference system for the "x" and "y" coordinates in
        `tide_df`. Used to ensure that interpolations are performed
        in the correct CRS.
    ensemble_models : list
        A list of models to include in the ensemble modelling process.
        All values must exist as columns with the prefix "rank_" in
        `ranking_points`.
    ensemble_func : dict, optional
        By default, a simple ensemble model will be calculated by taking
        the mean of the `ensemble_top_n` tide models at each location.
        However, a dictionary containing more complex ensemble
        calculations can also be provided. Dictionary keys are used
        to name output ensemble models; functions should take a column
        named "rank" and convert it to a weighting, e.g.:
        `ensemble_func = {"ensemble-custom": lambda x: x["rank"] <= 3}`
    ensemble_top_n : int, optional
        If `ensemble_func` is None, this sets the number of top models
        to include in the mean ensemble calculation. Defaults to 3.
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
    pandas.DataFrame
        DataFrame containing the ensemble model predictions, matching
        the format of the input `tide_df` (e.g. columns `["time", "x",
        "y", "tide_height", "tide_model"]`. By default the 'tide_model'
        column will be labeled "ensemble" for the combined model
        predictions (but if a custom dictionary of ensemble functions is
        provided via `ensemble_func`, each ensemble will be named using
        the provided dictionary keys).

    """
    # Raise data if `tide_df` provided in wide format
    if "tide_model" not in tide_df:
        err_msg = (
            "`tide_df` does not contain the expected 'tide_model' and "
            "'tide_height' columns. Ensure that tides were modelled in "
            "long format (i.e. `output_format='long'` in `model_tides`).",
        )
        raise Exception(err_msg)

    # Extract x and y coords from dataframe
    x = tide_df.index.get_level_values(level="x")
    y = tide_df.index.get_level_values(level="y")

    # Identify input datatype
    input_dtype = tide_df.tide_height.dtype

    # Load model ranks points and reproject to same CRS as x and y
    model_ranking_cols = [f"rank_{m}" for m in ensemble_models]
    try:
        model_ranks_gdf = (
            gpd.read_file(ranking_points, engine="pyogrio")
            .to_crs(crs)
            .query(f"valid_perc > {ranking_valid_perc}")
            .dropna(how="all")
            .filter(model_ranking_cols + ["geometry"])  # noqa: RUF005
        )
    except KeyError:
        error_msg = f"""
        Not all of the expected "rank_" columns {model_ranking_cols} were
        found in the columns of the ranking points file ({ranking_points}).
        Consider passing a custom list of models using `ensemble_models`.
        """
        raise Exception(textwrap.dedent(error_msg).strip()) from None

    # Use points to interpolate model rankings into requested x and y
    id_kwargs_str = "" if idw_kwargs == {} else idw_kwargs
    print(f"Interpolating model rankings using IDW interpolation {id_kwargs_str}")
    ensemble_ranks_df = (
        # Run IDW interpolation on subset of ranking columns
        pd.DataFrame(
            idw(
                input_z=model_ranks_gdf[model_ranking_cols],
                input_x=model_ranks_gdf.geometry.x,
                input_y=model_ranks_gdf.geometry.y,
                output_x=x,
                output_y=y,
                **idw_kwargs,
            ),
            columns=model_ranking_cols,
        )
        .assign(x=x, y=y)
        # Drop any duplicates then melt columns into long format
        .drop_duplicates()
        .melt(id_vars=["x", "y"], var_name="tide_model", value_name="rank")
        # Remove "rank_" prefix to get plain model names
        .replace({"^rank_": ""}, regex=True)
        # Set index columns and rank across groups
        .set_index(["tide_model", "x", "y"])
        .groupby(["x", "y"])
        .rank()
        .astype("float32")  # use smaller dtype for rankings to save memory
    )

    # If no custom ensemble funcs are provided, use a default ensemble
    # calculation that takes the mean of the top N tide models
    if ensemble_func is None:
        ensemble_func = {"ensemble": lambda x: x["rank"] <= ensemble_top_n}

    # Create output list to hold computed ensemble model outputs
    ensemble_list = []

    # Loop through all provided ensemble generation functions
    for ensemble_n, ensemble_f in ensemble_func.items():
        print(f"Combining models into single {ensemble_n} model")

        # Join ranks to input tide data, compute weightings and group
        grouped = (
            # Add tide model as an index so we can join with model ranks
            tide_df.set_index("tide_model", append=True)
            .join(ensemble_ranks_df)
            # Add temp columns containing weightings and weighted values
            .assign(
                weights=ensemble_f,  # use custom func to compute weights
                weighted=lambda i: i.tide_height * i.weights,
            )
            # Groupby is specified in a weird order here as this seems
            # to be the easiest way to preserve correct index sorting
            .groupby(["x", "y", "time"])
        )

        # Use weightings to combine multiple models into single ensemble
        ensemble_df = (
            # Calculate weighted mean
            grouped.weighted.sum()
            .div(grouped.weights.sum())
            # Make sure datatype is the same as the input
            .astype(input_dtype)
            # Convert to dataframe
            .to_frame("tide_height")
            # Label ensemble model and ensure indexes are in expected order
            .assign(tide_model=ensemble_n)
            .reorder_levels(["time", "x", "y"], axis=0)
        )

        ensemble_list.append(ensemble_df)

    # Combine all ensemble models and return as a single dataframe
    return pd.concat(ensemble_list)


def model_tides(
    x: float | list[float] | xr.DataArray,
    y: float | list[float] | xr.DataArray,
    time: DatetimeLike,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    crs: str = "EPSG:4326",
    mode: str = "one-to-many",
    output_format: str = "long",
    output_units: str = "m",
    method: str = "linear",
    extrapolate: bool = True,
    cutoff: float | None = None,
    crop: bool | str = "auto",
    crop_buffer: float | None = 5,
    append_node: bool = False,
    constituents: list[str] | None = None,
    parallel: bool = True,
    parallel_splits: int | str = "auto",
    parallel_max: int | None = None,
    ensemble_models: list[str] | None = None,
    extra_databases: str | os.PathLike | list | None = None,
    **ensemble_kwargs,
) -> pd.DataFrame:
    """Model tide heights at multiple coordinates or timesteps using using multiple ocean tide models.

    This function is parallelised to improve performance, and
    supports all tidal models supported by `pyTMD`, including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution tide models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global tide models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg direct data Assimilation Methods for Tides models (HAMTIDE11)
    - Technical University of Denmark tide models (DTU23)

    This function requires access to tide model data files.
    For tide model setup instructions, refer to the guide:
    https://geoscienceaustralia.github.io/eo-tides/setup/

    This function is a modification of the `pyTMD` package's
    `pyTMD.compute.tide_elevations` function. For more info:
    https://pytmd.readthedocs.io/en/latest/api_reference/compute.html#pyTMD.compute.tide_elevations
    https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories

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
        One or more UTC times at which to model tide heights. Accepts
        any time format compatible with `pandas.to_datetime()`, e.g.
        datetime.datetime, pd.Timestamp, pd.DatetimeIndex, numpy.datetime64,
        or date/time strings (e.g. "2020-01-01 23:00"). For example:
        `time = pd.date_range(start="2000", end="2001", freq="5h")`.
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        Defaults to "EOT20"; specify "all" to use all models available
        in `directory`. For a full list of available and supported models,
        run `from eo_tides.utils import list_models; list_models()`.
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
        Tide modelling analysis mode. Supports two options:

        - `"one-to-many"`: Models tides at every x/y coordinate for
        every timestep in `time`. This is useful for Earth observation
        workflows where you want to model tides at many spatial points
        for a common set of acquisition times (e.g. satellite overpasses).

        - `"one-to-one"`: Model tides using one timestep for each x/y
        coordinate. In this mode, the number of x/y coordinates must
        match the number of timesteps in `time`.
    output_format : str, optional
        Whether to return the output dataframe in long format (with
        results stacked vertically along "tide_model" and "tide_height"
        columns), or wide format (with a column for each tide model).
        Defaults to "long".
    output_units : str, optional
        Units for the returned tide heights. Options are:

        - `"m"` (default): floating point values in metres
        - `"cm"`: integer values in centimetres (x100)
        - `"mm"`: integer values in millimetres (x1000)

        Using integer units can help reduce memory usage.
    method : str, optional
        Method used to interpolate tide model constituent files.
        Defaults to "linear"; options include:

        - `"linear"`, `"nearest"`: scipy regular grid interpolations
        - `"spline"`: scipy bivariate spline interpolation
        - `"bilinear"`: quick bilinear interpolation
    extrapolate : bool, optional
        If True (default), extrapolate tides inland of the valid tide
        model extent using nearest-neighbor interpolation. This can
        ensure tide are returned everywhere, but accuracy may degrade
        with distance from the valid model extent (e.g. inland or along
        complex estuaries or rivers). Set `cutoff` to define the
        maximum extrapolation distance.
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
        Optional list of tide constituents to use for tide prediction.
        Default is None, which will use all available constituents.
    parallel : bool, optional
        Whether to parallelise tide modelling. If multiple tide models
        are requested, these will be run in parallel. If enough workers
        are available, the analysis will also be split into spatial
        chunks for additional parallelisation (see "parallel_splits"
        below). Default is True.
    parallel_splits : str or int, optional
        Whether to split the input x and y coordinates into smaller,
        evenly-sized chunks that are processed in parallel. This can
        provide a large performance boost when processing large numbers
        of coordinates. The default is "auto", which will automatically
        attempt to determine optimal splits based on available CPUs,
        the number of input points, and the number of models.
    parallel_max : int, optional
        Maximum number of processes to run in parallel. The default of
        None will automatically determine this from your available CPUs.
    ensemble_models : list of str, optional
        An optional list of models used to generate the ensemble tide
        model if "ensemble" tide modelling is requested. Defaults to
        `["EOT20", "FES2012", "FES2014_extrapolated", "FES2022_extrapolated",
        "GOT4.10", "GOT5.5_extrapolated", "GOT5.6_extrapolated",
        "TPXO10-atlas-v2-nc", "TPXO8-atlas-nc", "TPXO9-atlas-v5-nc"]`.
    extra_databases : str or path or list, optional
        Additional custom tide model definitions to load, provided as
        dictionaries or paths to JSON database files. Use this to
        enable custom tide models not included with `pyTMD`.
        See: https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#model-database
    **ensemble_kwargs :
        Keyword arguments used to customise the generation of optional
        ensemble tide models if "ensemble" tide modelling is requested.
        These are passed to the underlying `_ensemble_model` function.
        Useful parameters include `ranking_points` (path to model
        rankings data), `k` (for controlling how model rankings are
        interpolated), and `ensemble_top_n` (how many top models to use
        in the ensemble calculation).

    Returns
    -------
    pandas.DataFrame
        A dataframe containing modelled tide heights.

    """
    # Turn inputs into arrays for consistent handling
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = _standardise_time(time)

    # Validate input arguments
    if time is None:
        err_msg = "Times for modelling tides must be provided via `time`."
        raise ValueError(err_msg)

    if method not in ("bilinear", "spline", "linear", "nearest"):
        err_msg = (
            f"Invalid interpolation method '{method}'. Must be one of 'bilinear', 'spline', 'linear', or 'nearest'."
        )
        raise ValueError(err_msg)

    if output_units not in ("m", "cm", "mm"):
        err_msg = "Output units must be either 'm', 'cm', or 'mm'."
        raise ValueError(err_msg)

    if output_format not in ("long", "wide"):
        err_msg = "Output format must be either 'long' or 'wide'."
        raise ValueError(err_msg)

    if not np.issubdtype(x.dtype, np.number):
        err_msg = "`x` must contain only valid numeric values, and must not be None."
        raise TypeError(err_msg)

    if not np.issubdtype(y.dtype, np.number):
        err_msg = "`y` must contain only valid numeric values, and must not be None."
        raise TypeError(err_msg)

    if len(x) != len(y):
        err_msg = "`x` and `y` must be the same length."
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

    # Standardise model list, handling "all" and "ensemble" functionality,
    # and any custom tide model definitions
    models_to_process, models_requested, ensemble_models = _standardise_models(
        model=model,
        directory=directory,
        ensemble_models=ensemble_models,
        extra_databases=extra_databases,
    )

    # Update tide modelling func to add default keyword arguments that
    # are used for every iteration during parallel processing
    iter_func = partial(
        _model_tides,
        directory=directory,
        crs=crs,
        mode=mode,
        output_units=output_units,
        method=method,
        extrapolate=extrapolate,
        cutoff=np.inf if cutoff is None else cutoff,
        crop=crop,
        crop_buffer=crop_buffer,
        append_node=append_node,
        constituents=constituents,
        extra_databases=extra_databases,
    )

    # If automatic parallel splits, calculate optimal value
    # based on available parallelisation, number of points
    # and number of models
    if parallel_splits == "auto":
        parallel_splits = _parallel_splits(
            total_points=len(x),
            model_count=len(models_to_process),
            parallel_max=parallel_max,
        )

    # Verify that parallel splits are not larger than number of points
    assert isinstance(parallel_splits, int)  # noqa: S101
    if parallel_splits > len(x):
        err_msg = f"Parallel splits ({parallel_splits}) cannot be larger than the number of points ({len(x)})."
        raise ValueError(err_msg)

    # Parallelise if either multiple models or multiple splits requested
    if parallel & ((len(models_to_process) > 1) | (parallel_splits > 1)):
        with ProcessPoolExecutor(max_workers=parallel_max) as executor:
            print(
                f"Modelling tides with {', '.join(models_to_process)} in parallel (models: {len(models_to_process)}, splits: {parallel_splits})",
            )

            # Optionally split lon/lat points into `splits_n` chunks
            # that will be applied in parallel
            x_split = np.array_split(x, parallel_splits)
            y_split = np.array_split(y, parallel_splits)

            # Get every combination of models and lat/lon points, and
            # extract as iterables that can be passed to `executor.map()`
            # In "one-to-many" mode, pass entire set of timesteps to each
            # parallel iteration by repeating timesteps by number of total
            # parallel iterations. In "one-to-one" mode, split up
            # timesteps into smaller parallel chunks too.
            if mode == "one-to-many":
                model_iters, x_iters, y_iters = zip(
                    *[(m, x_split[i], y_split[i]) for m in models_to_process for i in range(parallel_splits)],
                    strict=False,
                )
                time_iters = [time] * len(model_iters)
            elif mode == "one-to-one":
                time_split = np.array_split(time, parallel_splits)
                model_iters, x_iters, y_iters, time_iters = zip(
                    *[
                        (m, x_split[i], y_split[i], time_split[i])
                        for m in models_to_process
                        for i in range(parallel_splits)
                    ],
                    strict=False,
                )

            # Apply func in parallel, iterating through each input param
            try:
                model_outputs = list(
                    tqdm(
                        executor.map(
                            iter_func,
                            model_iters,
                            x_iters,
                            y_iters,
                            time_iters,
                        ),
                        total=len(model_iters),
                    ),
                )
            except BrokenProcessPool:
                err_msg = (
                    "Parallelised tide modelling failed, likely to to an out-of-memory error. "
                    "Try reducing the size of your analysis, or set `parallel=False`."
                )
                raise RuntimeError(err_msg) from None

    # Model tides in series if parallelisation is off
    else:
        model_outputs = []

        for model_i in models_to_process:
            print(f"Modelling tides with {model_i}")
            tide_df = iter_func(model_i, x, y, time)
            model_outputs.append(tide_df)

    # Combine outputs into a single dataframe
    tide_df = pd.concat(model_outputs, axis=0)

    # Optionally compute ensemble model and add to dataframe
    if "ensemble" in models_requested:
        ensemble_df = ensemble_tides(tide_df, crs, ensemble_models, **ensemble_kwargs)

        # Update requested models with any custom ensemble models, then
        # filter the dataframe to keep only models originally requested
        models_requested = list(
            np.union1d(models_requested, ensemble_df.tide_model.unique()),
        )
        tide_df = pd.concat([tide_df, ensemble_df]).query(
            "tide_model in @models_requested",
        )

    # Optionally convert to a wide format dataframe with a tide model in
    # each dataframe column
    if output_format == "wide":
        # Pivot into wide format with each time model as a column
        print("Converting to a wide format dataframe")
        tide_df = tide_df.pivot(columns="tide_model", values="tide_height")  # noqa: PD010

        # If in 'one-to-one' mode, reindex using our input time/x/y
        # values to ensure the output is sorted the same as our inputs
        if mode == "one-to-one":
            output_indices = pd.MultiIndex.from_arrays(
                [time, x, y],
                names=["time", "x", "y"],
            )
            tide_df = tide_df.reindex(output_indices)

    return tide_df


def model_phases(
    x: float | list[float] | xr.DataArray,
    y: float | list[float] | xr.DataArray,
    time: DatetimeLike,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    time_offset: str = "15 min",
    return_tides: bool = False,
    **model_tides_kwargs,
) -> pd.DataFrame:
    """Model tide phases at multiple coordinates or timesteps using multiple ocean tide models.

    Ebb and low phases (low-flow, high-flow, high-ebb, low-ebb)
    are calculated by running the `eo_tides.model.model_tides`
    function twice, once for the requested timesteps, and again
    after subtracting a small time offset (15 mins by default).
    If tides increased over this period, they are assigned as
    "flow"; if they decreased, they are assigned as "ebb".
    Tides are considered "high" if equal or greater than 0
    metres tide height, otherwise "low".

    This function supports all parameters that are supported
    by `model_tides`.

    For tide model setup instructions, refer to the guide:
    https://geoscienceaustralia.github.io/eo-tides/setup/

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
        One or more UTC times at which to model tide heights. Accepts
        any time format compatible with `pandas.to_datetime()`, e.g.
        datetime.datetime, pd.Timestamp, pd.DatetimeIndex, numpy.datetime64,
        or date/time strings (e.g. "2020-01-01 23:00"). For example:
        `time = pd.date_range(start="2000", end="2001", freq="5h")`.
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        Defaults to "EOT20"; specify "all" to use all models available
        in `directory`. For a full list of available and supported models,
        run `from eo_tides.utils import list_models; list_models()`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    time_offset: str, optional
        The time offset/delta used to generate a time series of
        offset tide heights required for phase calculation. Defaults
        to "15 min"; can be any string passed to `pandas.Timedelta`.
    return_tides: bool, optional
        Whether to return intermediate modelled tide heights as a
        "tide_height" column in the output dataframe. Defaults to False.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `output_format` (e.g.
        whether to return results in wide or long format), `crop`
        (whether to crop tide model constituent files on-the-fly to
        improve performance) etc.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing modelled tide phases.

    """
    # Pop output format and mode for special handling
    output_format = model_tides_kwargs.pop("output_format", "long")
    mode = model_tides_kwargs.pop("mode", "one-to-many")

    # Model tides
    tide_df = model_tides(
        x=x,
        y=y,
        time=time,
        model=model,
        directory=directory,
        **model_tides_kwargs,
    )

    # Model tides for a time 15 minutes prior to each previously
    # modelled satellite acquisition time. This allows us to compare
    # tide heights to see if they are rising or falling.
    pre_df = model_tides(
        x=x,
        y=y,
        time=time - pd.Timedelta(time_offset),
        model=model,
        directory=directory,
        **model_tides_kwargs,
    )

    # Compare tides computed for each timestep. If the previous tide
    # was higher than the current tide, the tide is 'ebbing'. If the
    # previous tide was lower, the tide is 'flowing'
    ebb_flow = (tide_df.tide_height < pre_df.tide_height.to_numpy()).replace(
        {True: "ebb", False: "flow"},
    )

    # If tides are greater than 0, then "high", otherwise "low"
    high_low = (tide_df.tide_height >= 0).replace({True: "high", False: "low"})

    # Combine into one string and add to data
    tide_df["tide_phase"] = high_low.astype(str) + "-" + ebb_flow.astype(str)

    # Optionally convert to a wide format dataframe with a tide model in
    # each dataframe column
    if output_format == "wide":
        # Pivot into wide format with each time model as a column
        print("Converting to a wide format dataframe")
        tide_df = tide_df.pivot(columns="tide_model")  # noqa: PD010

        # If in 'one-to-one' mode, reindex using our input time/x/y
        # values to ensure the output is sorted the same as our inputs
        if mode == "one-to-one":
            output_indices = pd.MultiIndex.from_arrays(
                [time, x, y],
                names=["time", "x", "y"],
            )
            tide_df = tide_df.reindex(output_indices)

        # Optionally drop tides
        if not return_tides:
            return tide_df.drop("tide_height", axis=1)["tide_phase"]

    # Optionally drop tide heights
    if not return_tides:
        return tide_df.drop("tide_height", axis=1)

    return tide_df
