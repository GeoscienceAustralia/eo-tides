# Used to postpone evaluation of type annotations
from __future__ import annotations

import os
import pathlib
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

# Only import if running type checking
if TYPE_CHECKING:
    import xarray as xr

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pyTMD
from colorama import Style, init
from pyTMD.io.model import load_database, model
from tqdm import tqdm

from .utils import idw


def _set_directory(directory):
    """
    Set tide modelling files directory. If no custom
    path is provided, try global environmental variable
    instead.
    """
    if directory is None:
        if "EO_TIDES_TIDE_MODELS" in os.environ:
            directory = os.environ["EO_TIDES_TIDE_MODELS"]
        else:
            raise Exception(
                "No tide model directory provided via `directory`, and/or no "
                "`EO_TIDES_TIDE_MODELS` environment variable found. "
                "Please provide a valid path to your tide model directory."
            )

    # Verify path exists
    directory = pathlib.Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"No valid tide model directory found at path `{directory}`")
    else:
        return directory


def list_models(
    directory: str | os.PathLike | None = None,
    show_available: bool = True,
    show_supported: bool = True,
    raise_error: bool = False,
) -> tuple[list[str], list[str]]:
    """
    List all tide models available for tide modelling, and
    all models supported by `eo-tides` and `pyTMD`.

    This function scans the specified tide model directory
    and returns a list of models that are available in the
    directory as well as the full list of all supported models.

    For instructions on setting up tide models, see:
    <https://geoscienceaustralia.github.io/eo-tides/setup/>

    Parameters
    ----------
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    show_available : bool, optional
        Whether to print a list of locally available models.
    show_supported : bool, optional
        Whether to print a list of all supported models, in
        addition to models available locally.
    raise_error : bool, optional
        If True, raise an error if no available models are found.
        If False, raise a warning.

    Returns
    -------
    available_models : list of str
        A list of all tide models available within `directory`.
    supported_models : list of str
        A list of all tide models supported by `eo-tides`.
    """
    init()  # Initialize colorama

    # Set tide modelling files directory. If no custom path is
    # provided, try global environment variable.
    directory = _set_directory(directory)

    # Get full list of supported models from pyTMD database
    model_database = load_database()["elevation"]
    supported_models = list(model_database.keys())

    # Extract expected model paths
    expected_paths = {}
    for m in supported_models:
        model_file = model_database[m]["model_file"]
        model_file = model_file[0] if isinstance(model_file, list) else model_file
        expected_paths[m] = str(directory / pathlib.Path(model_file).expanduser().parent)

    # Define column widths
    status_width = 4  # Width for emoji
    name_width = max(len(name) for name in supported_models)
    path_width = max(len(path) for path in expected_paths.values())

    # Print list of supported models, marking available and
    # unavailable models and appending available to list
    if show_available or show_supported:
        total_width = min(status_width + name_width + path_width + 6, 80)
        print("─" * total_width)
        print(f"{'󠀠🌊':^{status_width}} | {'Model':<{name_width}} | {'Expected path':<{path_width}}")
        print("─" * total_width)

    available_models = []
    for m in supported_models:
        try:
            model_file = model(directory=directory).elevation(m=m)
            available_models.append(m)

            if show_available:
                # Mark available models with a green tick
                status = "✅"
                print(f"{status:^{status_width}}│ {m:<{name_width}} │ {expected_paths[m]:<{path_width}}")
        except:
            if show_supported:
                # Mark unavailable models with a red cross
                status = "❌"
                print(
                    f"{status:^{status_width}}│ {Style.DIM}{m:<{name_width}} │ {expected_paths[m]:<{path_width}}{Style.RESET_ALL}"
                )

    if show_available or show_supported:
        print("─" * total_width)

        # Print summary
        print(f"\n{Style.BRIGHT}Summary:{Style.RESET_ALL}")
        print(f"Available models: {len(available_models)}/{len(supported_models)}")

    # Raise error or warning if no models are available
    if not available_models:
        warning_text = (
            f"No valid tide models are available in `{directory}`. "
            "Are you sure you have provided the correct `directory` path, "
            "or set the `EO_TIDES_TIDE_MODELS` environment variable "
            "to point to the location of your tide model directory?"
        )
        if raise_error:
            raise Exception(warning_text)
        else:
            warnings.warn(warning_text, UserWarning)

    # Return list of available and supported models
    return available_models, supported_models


def _model_tides(
    model,
    x,
    y,
    time,
    directory,
    crs,
    crop,
    method,
    extrapolate,
    cutoff,
    output_units,
    mode,
):
    """Worker function applied in parallel by `model_tides`. Handles the
    extraction of tide modelling constituents and tide modelling using
    `pyTMD`.
    """
    # Obtain model details
    pytmd_model = pyTMD.io.model(directory).elevation(model)

    # Convert x, y to latitude/longitude
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())

    # Convert datetime
    timescale = pyTMD.time.timescale().from_datetime(time.flatten())

    # Calculate bounds for cropping
    buffer = 1  # one degree on either side
    bounds = [
        lon.min() - buffer,
        lon.max() + buffer,
        lat.min() - buffer,
        lat.max() + buffer,
    ]

    # Read tidal constants and interpolate to grid points
    if pytmd_model.format in ("OTIS", "ATLAS-compact", "TMD3"):
        amp, ph, D, c = pyTMD.io.OTIS.extract_constants(
            lon,
            lat,
            pytmd_model.grid_file,
            pytmd_model.model_file,
            pytmd_model.projection,
            type=pytmd_model.type,
            grid=pytmd_model.file_format,
            crop=crop,
            bounds=bounds,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
        )

        # Use delta time at 2000.0 to match TMD outputs
        deltat = np.zeros((len(timescale)), dtype=np.float64)

    elif pytmd_model.format in ("ATLAS-netcdf",):
        amp, ph, D, c = pyTMD.io.ATLAS.extract_constants(
            lon,
            lat,
            pytmd_model.grid_file,
            pytmd_model.model_file,
            type=pytmd_model.type,
            crop=crop,
            bounds=bounds,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
            scale=pytmd_model.scale,
            compressed=pytmd_model.compressed,
        )

        # Use delta time at 2000.0 to match TMD outputs
        deltat = np.zeros((len(timescale)), dtype=np.float64)

    elif pytmd_model.format in ("GOT-ascii", "GOT-netcdf"):
        amp, ph, c = pyTMD.io.GOT.extract_constants(
            lon,
            lat,
            pytmd_model.model_file,
            grid=pytmd_model.file_format,
            crop=crop,
            bounds=bounds,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
            scale=pytmd_model.scale,
            compressed=pytmd_model.compressed,
        )

        # Delta time (TT - UT1)
        deltat = timescale.tt_ut1

    elif pytmd_model.format in ("FES-ascii", "FES-netcdf"):
        amp, ph = pyTMD.io.FES.extract_constants(
            lon,
            lat,
            pytmd_model.model_file,
            type=pytmd_model.type,
            version=pytmd_model.version,
            crop=crop,
            bounds=bounds,
            method=method,
            extrapolate=extrapolate,
            cutoff=cutoff,
            scale=pytmd_model.scale,
            compressed=pytmd_model.compressed,
        )

        # Available model constituents
        c = pytmd_model.constituents

        # Delta time (TT - UT1)
        deltat = timescale.tt_ut1
    else:
        raise Exception(
            f"Unsupported model format ({pytmd_model.format}). This may be due to an incompatible version of `pyTMD`."
        )

    # Calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # Calculate constituent oscillation
    hc = amp * np.exp(cph)

    # Determine the number of points and times to process. If in
    # "one-to-many" mode, these counts are used to repeat our extracted
    # constituents and timesteps so we can extract tides for all
    # combinations of our input times and tide modelling points.
    # If in "one-to-one" mode, we avoid this step by setting counts to 1
    # (e.g. "repeat 1 times")
    points_repeat = len(x) if mode == "one-to-many" else 1
    time_repeat = len(time) if mode == "one-to-many" else 1

    # If in "one-to-many" mode, repeat constituents to length of time
    # and number of input coords before passing to `predict_tide_drift`
    t, hc, deltat = (
        np.tile(timescale.tide, points_repeat),
        hc.repeat(time_repeat, axis=0),
        np.tile(deltat, points_repeat),
    )

    # Predict tidal elevations at time and infer minor corrections
    npts = len(t)
    tide = np.ma.zeros((npts), fill_value=np.nan)
    tide.mask = np.any(hc.mask, axis=1)

    # Predict tides
    tide.data[:] = pyTMD.predict.drift(t, hc, c, deltat=deltat, corrections=pytmd_model.corrections)
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
    tide_df = pd.DataFrame({
        "time": np.tile(time, points_repeat),
        "x": np.repeat(x, time_repeat),
        "y": np.repeat(y, time_repeat),
        "tide_model": model,
        "tide_height": tide,
    }).set_index(["time", "x", "y"])

    # Optionally convert outputs to integer units (can save memory)
    if output_units == "m":
        tide_df["tide_height"] = tide_df.tide_height.astype(np.float32)
    elif output_units == "cm":
        tide_df["tide_height"] = (tide_df.tide_height * 100).astype(np.int16)
    elif output_units == "mm":
        tide_df["tide_height"] = (tide_df.tide_height * 1000).astype(np.int16)

    return tide_df


def _ensemble_model(
    tide_df,
    crs,
    ensemble_models,
    ensemble_func=None,
    ensemble_top_n=3,
    ranking_points="https://dea-public-data-dev.s3-ap-southeast-2.amazonaws.com/derivative/dea_intertidal/supplementary/rankings_ensemble_2017-2019.geojson",
    ranking_valid_perc=0.02,
    **idw_kwargs,
):
    """Combine multiple tide models into a single locally optimised
    ensemble tide model using external model ranking data (e.g.
    satellite altimetry or NDWI-tide correlations along the coastline)
    to inform the selection of the best local models.

    This function performs the following steps:
    1. Takes a dataframe of tide heights from multiple tide models, as
       produced by `eo_tides.model.model_tides`
    1. Loads model ranking points from a GeoJSON file, filters them
       based on the valid data percentage, and retains relevant columns
    2. Interpolates the model rankings into the "x" and "y" coordinates
       of the original dataframe using Inverse Weighted Interpolation (IDW)
    3. Uses rankings to combine multiple tide models into a single
       optimised ensemble model (by default, by taking the mean of the
       top 3 ranked models)
    4. Returns a new dataFrame with the combined ensemble model predictions

    Parameters
    ----------
    tide_df : pandas.DataFrame
        DataFrame produced by `eo_tides.model.model_tides`, containing
        tide model predictions with columns:
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
        Path to the GeoJSON file containing model ranking points. This
        dataset should include columns containing rankings for each tide
        model, named with the prefix "rank_". e.g. "rank_EOT20".
        Low values should represent high rankings (e.g. 1 = top ranked).
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
    # Extract x and y coords from dataframe
    x = tide_df.index.get_level_values(level="x")
    y = tide_df.index.get_level_values(level="y")

    # Load model ranks points and reproject to same CRS as x and y
    model_ranking_cols = [f"rank_{m}" for m in ensemble_models]
    model_ranks_gdf = (
        gpd.read_file(ranking_points)
        .to_crs(crs)
        .query(f"valid_perc > {ranking_valid_perc}")
        .dropna()[model_ranking_cols + ["geometry"]]
    )

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
        # Remore "rank_" prefix to get plain model names
        .replace({"^rank_": ""}, regex=True)
        # Set index columns and rank across groups
        .set_index(["tide_model", "x", "y"])
        .groupby(["x", "y"])
        .rank()
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
            # Calculate weighted mean and convert back to dataframe
            grouped.weighted.sum()
            .div(grouped.weights.sum())
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
    time: np.ndarray | pd.DatetimeIndex,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    crs: str = "EPSG:4326",
    crop: bool = True,
    method: str = "spline",
    extrapolate: bool = True,
    cutoff: float | None = None,
    mode: str = "one-to-many",
    parallel: bool = True,
    parallel_splits: int = 5,
    output_units: str = "m",
    output_format: str = "long",
    ensemble_models: list[str] | None = None,
    **ensemble_kwargs,
) -> pd.DataFrame:
    """
    Model tide heights at multiple coordinates and/or timesteps
    using using one or more ocean tide models.

    This function is parallelised to improve performance, and
    supports all tidal models supported by `pyTMD`, including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution tide models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global tide models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg direct data Assimilation Methods for Tides models (HAMTIDE11)

    This function requires access to tide model data files.
    These should be placed in a folder with subfolders matching
    the structure required by `pyTMD`. For more details:
    <https://geoscienceaustralia.github.io/eo-tides/setup/>
    <https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories>

    This function is a modification of the `pyTMD` package's
    `compute_tidal_elevations` function. For more info:
    <https://pytmd.readthedocs.io/en/latest/api_reference/compute_tidal_elevations.html>

    Parameters
    ----------
    x, y : float or list of float
        One or more x and y coordinates used to define
        the location at which to model tides. By default these
        coordinates should be lat/lon; use "crs" if they
        are in a custom coordinate reference system.
    time : Numpy datetime array or pandas.DatetimeIndex
        An array containing `datetime64[ns]` values or a
        `pandas.DatetimeIndex` providing the times at which to
        model tides in UTC time.
    model : str or list of str, optional
        The tide model (or models) to use to model tides.
        Defaults to "EOT20"; for a full list of available/supported
        models, run `eo_tides.model.list_models`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    crs : str, optional
        Input coordinate reference system for x and y coordinates.
        Defaults to "EPSG:4326" (WGS84; degrees latitude, longitude).
    crop : bool, optional
        Whether to crop tide model constituent files on-the-fly to
        improve performance. Cropping will be performed based on a
        1 degree buffer around all input points. Defaults to True.
    method : str, optional
        Method used to interpolate tidal constituents
        from model files. Options include:

        - "spline": scipy bivariate spline interpolation (default)
        - "bilinear": quick bilinear interpolation
        - "linear", "nearest": scipy regular grid interpolations
    extrapolate : bool, optional
        Whether to extrapolate tides for x and y coordinates outside of
        the valid tide modelling domain using nearest-neighbor.
    cutoff : float, optional
        Extrapolation cutoff in kilometers. The default is None, which
        will extrapolate for all points regardless of distance from the
        valid tide modelling domain.
    mode : str, optional
        The analysis mode to use for tide modelling. Supports two options:

        - "one-to-many": Models tides for every timestep in "time" at
        every input x and y coordinate point. This is useful if you
        want to model tides for a specific list of timesteps across
        multiple spatial points (e.g. for the same set of satellite
        acquisition times at various locations across your study area).
        - "one-to-one": Model tides using a unique timestep for each
        set of x and y coordinates. In this mode, the number of x and
        y points must equal the number of timesteps provided in "time".

    parallel : bool, optional
        Whether to parallelise tide modelling using `concurrent.futures`.
        If multiple tide models are requested, these will be run in
        parallel. Optionally, tide modelling can also be run in parallel
        across input x and y coordinates (see "parallel_splits" below).
        Default is True.
    parallel_splits : int, optional
        Whether to split the input x and y coordinates into smaller,
        evenly-sized chunks that are processed in parallel. This can
        provide a large performance boost when processing large numbers
        of coordinates. The default is 5 chunks, which will split
        coordinates into 5 parallelised chunks.
    output_units : str, optional
        Whether to return modelled tides in floating point metre units,
        or integer centimetre units (i.e. scaled by 100) or integer
        millimetre units (i.e. scaled by 1000. Returning outputs in
        integer units can be useful for reducing memory usage.
        Defaults to "m" for metres; set to "cm" for centimetres or "mm"
        for millimetres.
    output_format : str, optional
        Whether to return the output dataframe in long format (with
        results stacked vertically along "tide_model" and "tide_height"
        columns), or wide format (with a column for each tide model).
        Defaults to "long".
    ensemble_models : list of str, optional
        An optional list of models used to generate the ensemble tide
        model if "ensemble" tide modelling is requested. Defaults to
        ["FES2014", "TPXO9-atlas-v5", "EOT20", "HAMTIDE11", "GOT4.10",
        "FES2012", "TPXO8-atlas-v1"].
    **ensemble_kwargs :
        Keyword arguments used to customise the generation of optional
        ensemble tide models if "ensemble" modelling are requested.
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
    models_requested = list(np.atleast_1d(model))
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = np.atleast_1d(time)

    # Validate input arguments
    assert method in ("bilinear", "spline", "linear", "nearest")
    assert output_units in (
        "m",
        "cm",
        "mm",
    ), "Output units must be either 'm', 'cm', or 'mm'."
    assert output_format in (
        "long",
        "wide",
    ), "Output format must be either 'long' or 'wide'."
    assert len(x) == len(y), "x and y must be the same length."
    if mode == "one-to-one":
        assert len(x) == len(time), (
            "The number of supplied x and y points and times must be "
            "identical in 'one-to-one' mode. Use 'one-to-many' mode if "
            "you intended to model multiple timesteps at each point."
        )

    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Set tide modelling files directory. If no custom path is
    # provided, try global environment variable.
    directory = _set_directory(directory)

    # Get full list of supported models from pyTMD database;
    # add ensemble option to list of models
    available_models, valid_models = list_models(
        directory, show_available=False, show_supported=False, raise_error=True
    )
    # TODO: This is hacky, find a better way. Perhaps a kwarg that
    # turns ensemble functionality on, and checks that supplied
    # models match models expected for ensemble?
    available_models = available_models + ["ensemble"]
    valid_models = valid_models + ["ensemble"]

    # Error if any models are not supported
    if not all(m in valid_models for m in models_requested):
        error_text = (
            f"One or more of the requested models are not valid:\n"
            f"{models_requested}\n\n"
            "The following models are supported:\n"
            f"{valid_models}"
        )
        raise ValueError(error_text)

    # Error if any models are not available in `directory`
    if not all(m in available_models for m in models_requested):
        error_text = (
            f"One or more of the requested models are valid, but not available in `{directory}`:\n"
            f"{models_requested}\n\n"
            f"The following models are available in `{directory}`:\n"
            f"{available_models}"
        )
        raise ValueError(error_text)

    # If ensemble modelling is requested, use a custom list of models
    # for subsequent processing
    if "ensemble" in models_requested:
        print("Running ensemble tide modelling")
        models_to_process = (
            ensemble_models
            if ensemble_models is not None
            else [
                "FES2014",
                "TPXO9-atlas-v5",
                "EOT20",
                "HAMTIDE11",
                "GOT4.10",
                "FES2012",
                "TPXO8-atlas-v1",
            ]
        )

    # Otherwise, models to process are the same as those requested
    else:
        models_to_process = models_requested

    # Update tide modelling func to add default keyword arguments that
    # are used for every iteration during parallel processing
    iter_func = partial(
        _model_tides,
        directory=directory,
        crs=crs,
        crop=crop,
        method=method,
        extrapolate=extrapolate,
        cutoff=np.inf if cutoff is None else cutoff,
        output_units=output_units,
        mode=mode,
    )

    # Ensure requested parallel splits is not smaller than number of points
    parallel_splits = min(parallel_splits, len(x))

    # Parallelise if either multiple models or multiple splits requested
    if parallel & ((len(models_to_process) > 1) | (parallel_splits > 1)):
        with ProcessPoolExecutor() as executor:
            print(f"Modelling tides using {', '.join(models_to_process)} in parallel")

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
                )

            # Apply func in parallel, iterating through each input param
            model_outputs = list(
                tqdm(
                    executor.map(iter_func, model_iters, x_iters, y_iters, time_iters),
                    total=len(model_iters),
                ),
            )

    # Model tides in series if parallelisation is off
    else:
        model_outputs = []

        for model_i in models_to_process:
            print(f"Modelling tides using {model_i}")
            tide_df = iter_func(model_i, x, y, time)
            model_outputs.append(tide_df)

    # Combine outputs into a single dataframe
    tide_df = pd.concat(model_outputs, axis=0)

    # Optionally compute ensemble model and add to dataframe
    if "ensemble" in models_requested:
        ensemble_df = _ensemble_model(tide_df, crs, models_to_process, **ensemble_kwargs)

        # Update requested models with any custom ensemble models, then
        # filter the dataframe to keep only models originally requested
        models_requested = list(np.union1d(models_requested, ensemble_df.tide_model.unique()))
        tide_df = pd.concat([tide_df, ensemble_df]).query("tide_model in @models_requested")

    # Optionally convert to a wide format dataframe with a tide model in
    # each dataframe column
    if output_format == "wide":
        # Pivot into wide format with each time model as a column
        print("Converting to a wide format dataframe")
        tide_df = tide_df.pivot(columns="tide_model", values="tide_height")

        # If in 'one-to-one' mode, reindex using our input time/x/y
        # values to ensure the output is sorted the same as our inputs
        if mode == "one-to-one":
            output_indices = pd.MultiIndex.from_arrays([time, x, y], names=["time", "x", "y"])
            tide_df = tide_df.reindex(output_indices)

    return tide_df
