# Used to postpone evaluation of type annotations
from __future__ import annotations

import datetime
import os
import pathlib
import textwrap
import warnings
from collections import Counter
from typing import List, Union

import numpy as np
import odc.geo
import pandas as pd
import xarray as xr
from colorama import Style, init
from odc.geo.geom import BoundingBox
from pyTMD.io.model import load_database
from pyTMD.io.model import model as pytmd_model
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

# Type alias for all possible inputs to "time" params
DatetimeLike = Union[np.ndarray, pd.DatetimeIndex, pd.Timestamp, datetime.datetime, str, List[str]]


def _get_duplicates(array):
    """
    Return any duplicates in a list or array.
    """
    c = Counter(array)
    return [k for k in c if c[k] > 1]


def _set_directory(
    directory: str | os.PathLike | None = None,
) -> os.PathLike:
    """
    Set tide modelling files directory. If no custom
    path is provided, try global `EO_TIDES_TIDE_MODELS`
    environmental variable instead.
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


def _standardise_time(
    time: DatetimeLike | None,
) -> np.ndarray | None:
    """
    Accept any time format accepted by `pd.to_datetime`,
    and return a datetime64 ndarray. Return None if None
    passed.
    """
    # Return time as-is if None
    if time is None:
        return None

    # Use pd.to_datetime for conversion, then convert to numpy array
    time = pd.to_datetime(time).to_numpy().astype("datetime64[ns]")

    # Ensure that data has at least one dimension
    return np.atleast_1d(time)


def _standardise_models(
    model: str | list[str],
    directory: str | os.PathLike,
    ensemble_models: list[str] | None = None,
) -> tuple[list[str], list[str], list[str] | None]:
    """
    Take an input model name or list of names, and return a list
    of models to process, requested models, and ensemble models,
    as required by the `model_tides` function.

    Handles two special values passed to `model`: "all", which
    will model tides for all models available in `directory`, and
    "ensemble", which will model tides for all models in a list
    of custom ensemble models.
    """

    # Turn inputs into arrays for consistent handling
    models_requested = list(np.atleast_1d(model))

    # Raise error if list contains duplications
    duplicates = _get_duplicates(models_requested)
    if len(duplicates) > 0:
        raise ValueError(f"The model parameter contains duplicate values: {duplicates}")

    # Get full list of supported models from pyTMD database
    available_models, valid_models = list_models(
        directory, show_available=False, show_supported=False, raise_error=True
    )
    custom_options = ["ensemble", "all"]

    # Error if any models are not supported
    if not all(m in valid_models + custom_options for m in models_requested):
        error_text = (
            f"One or more of the requested models are not valid:\n"
            f"{models_requested}\n\n"
            "The following models are supported:\n"
            f"{valid_models}"
        )
        raise ValueError(error_text)

    # Error if any models are not available in `directory`
    if not all(m in available_models + custom_options for m in models_requested):
        error_text = (
            f"One or more of the requested models are valid, but not available in `{directory}`:\n"
            f"{models_requested}\n\n"
            f"The following models are available in `{directory}`:\n"
            f"{available_models}"
        )
        raise ValueError(error_text)

    # If "all" models are requested, update requested list to include available models
    if "all" in models_requested:
        models_requested = available_models + [m for m in models_requested if m != "all"]

    # If "ensemble" modeling is requested, use custom list of ensemble models
    if "ensemble" in models_requested:
        print("Running ensemble tide modelling")
        ensemble_models = (
            ensemble_models
            if ensemble_models is not None
            else [
                "EOT20",
                "FES2012",
                "FES2014_extrapolated",
                "FES2022_extrapolated",
                "GOT4.10",
                "GOT5.6_extrapolated",
                "TPXO10-atlas-v2-nc",
                "TPXO8-atlas-nc",
                "TPXO9-atlas-v5-nc",
            ]
        )

        # Error if any ensemble models are not available in `directory`
        if not all(m in available_models for m in ensemble_models):
            error_text = (
                f"One or more of the requested ensemble models are not available in `{directory}`:\n"
                f"{ensemble_models}\n\n"
                f"The following models are available in `{directory}`:\n"
                f"{available_models}"
            )
            raise ValueError(error_text)

        # Return set of all ensemble plus any other requested models
        models_to_process = sorted(list(set(ensemble_models + [m for m in models_requested if m != "ensemble"])))

    # Otherwise, models to process are the same as those requested
    else:
        models_to_process = models_requested

    return models_to_process, models_requested, ensemble_models


def _clip_model_file(
    nc: xr.Dataset,
    bbox: BoundingBox,
    ydim: str,
    xdim: str,
    ycoord: str,
    xcoord: str,
) -> xr.Dataset:
    """
    Clips tide model netCDF datasets to a bounding box.

    If the bounding box crosses 0 degrees longitude (e.g. Greenwich prime
    meridian), the dataset will be clipped into two parts and concatenated
    along the x-dimension to create a continuous result.

    Parameters
    ----------
    nc : xr.Dataset
        Input tide model xarray dataset.
    bbox : odc.geo.geom.BoundingBox
        A BoundingBox object for clipping the dataset in EPSG:4326
        degrees coordinates. For example:
        `BoundingBox(left=108, bottom=-48, right=158, top=-6, crs='EPSG:4326')`
    ydim : str
        The name of the xarray dimension representing the y-axis.
        Depending on the tide model, this may or may not contain
        actual latitude values.
    xdim : str
        The name of the xarray dimension representing the x-axis.
        Depending on the tide model, this may or may not contain
        actual longitude values.
    ycoord : str
        The name of the coordinate, variable or dimension containing
        actual latitude values used for clipping the data.
    xcoord : str
        The name of the coordinate, variable or dimension containing
        actual longitude values used for clipping the data.

    Returns
    -------
    xr.Dataset
        A dataset clipped to the specified bounding box, with
        appropriate adjustments if the bounding box crosses 0
        degrees longitude.

    Examples
    --------
    >>> nc = xr.open_dataset("GOT5.5/ocean_tides/2n2.nc")
    >>> bbox = BoundingBox(left=108, bottom=-48, right=158, top=-6, crs='EPSG:4326')
    >>> clipped_nc = _clip_model_file(nc, bbox,  xdim="lon", ydim="lat", ycoord="latitude", xcoord="longitude")
    """

    # Extract x and y coords from xarray and load into memory
    xcoords = nc[xcoord].compute()
    ycoords = nc[ycoord].compute()

    # Convert longitudes to 0-360 convention
    left = bbox.left % 360
    right = bbox.right % 360

    # If left coordinate is smaller than right, bbox does not cross
    # zero longitude and can be clipped directly
    if left <= right:  # bbox does not cross 0
        nc_clipped = nc.sel({
            ydim: (ycoords >= bbox.bottom) & (ycoords <= bbox.top),
            xdim: (xcoords >= left) & (xcoords <= right),
        })

    # If left coordinate is larger than right, bbox crosses zero longitude.
    # If so, extract left and right separately and then combine into one
    # concatenated dataset
    elif left > right:  # bbox crosses 0
        # Extract data from left of 0 longitude, and convert lon
        # coords to -180 to 0 range to enable continuous interpolation
        # across 0 boundary
        nc_left = nc.sel({
            ydim: (ycoords >= bbox.bottom) & (ycoords <= bbox.top),
            xdim: (xcoords >= left) & (xcoords <= 360),
        }).assign({xcoord: lambda x: x[xcoord] - 360})

        # Convert additional lon variables for TXPO
        if "lon_v" in nc_left:
            nc_left = nc_left.assign({
                "lon_v": lambda x: x["lon_v"] - 360,
                "lon_u": lambda x: x["lon_u"] - 360,
            })

        # Extract data to right of 0 longitude
        nc_right = nc.sel({
            ydim: (ycoords >= bbox.bottom) & (ycoords <= bbox.top),
            xdim: (xcoords > 0) & (xcoords <= right),
        })

        # Combine left and right data along x dimension
        nc_clipped = xr.concat([nc_left, nc_right], dim=xdim)

        # Hack fix to remove expanded x dim on lat variables issue
        # for TPXO data; remove x dim by selecting the first obs
        for i in ["lat_z", "lat_v", "lat_u", "con"]:
            try:
                nc_clipped[i] = nc_clipped[i].isel(nx=0)
            except:
                pass

    return nc_clipped


def clip_models(
    input_directory: str | os.PathLike,
    output_directory: str | os.PathLike,
    bbox: tuple[float, float, float, float],
    model: list | None = None,
    buffer: float = 5,
    overwrite: bool = False,
):
    """
    Clip NetCDF-format ocean tide models to a bounding box.

    This function identifies all NetCDF-format tide models in a
    given input directory, including "ATLAS-netcdf" (e.g. TPXO9-atlas-nc),
    "FES-netcdf" (e.g. FES2022, EOT20), and "GOT-netcdf" (e.g. GOT5.5)
    format files. Files for each model are then clipped to the extent of
    the provided bounding box, handling model-specific file structures.
    After each model is clipped, the result is exported to the output
    directory and verified with `pyTMD` to ensure the clipped data is
    suitable for tide modelling.

    For instructions on accessing and downloading tide models, see:
    <https://geoscienceaustralia.github.io/eo-tides/setup/>

    Parameters
    ----------
    input_directory : str or os.PathLike
        Path to directory containing input NetCDF-format tide model files.
    output_directory : str or os.PathLike
        Path to directory where clipped NetCDF files will be exported.
    bbox : tuple of float
        Bounding box for clipping the tide models in EPSG:4326 degrees
        coordinates, specified as `(left, bottom, right, top)`.
    model : str or list of str, optional
        The tide model (or models) to clip. Defaults to None, which
        will automatically identify and clip all NetCDF-format models
        in the input directly.
    buffer : float, optional
        Buffer distance (in degrees) added to the bounding box to provide
        sufficient data on edges of study area. Defaults to 5 degrees.
    overwrite : bool, optional
        If True, overwrite existing files in the output directory.
        Defaults to False.

    Examples
    --------
    >>> clip_models(
    ...     input_directory="tide_models/",
    ...     output_directory="tide_models_clipped/",
    ...     bbox=(-8.968392, 50.070574, 2.447160, 59.367122),
    ... )
    """

    # Get input and output paths
    input_directory = _set_directory(input_directory)
    output_directory = pathlib.Path(output_directory)

    # Prepare bounding box
    bbox = odc.geo.geom.BoundingBox(*bbox, crs="EPSG:4326").buffered(buffer)

    # Identify NetCDF models
    model_database = load_database()["elevation"]
    netcdf_formats = ["ATLAS-netcdf", "FES-netcdf", "GOT-netcdf"]
    netcdf_models = {k for k, v in model_database.items() if v["format"] in netcdf_formats}

    # Identify subset of available and requested NetCDF models
    available_models, _ = list_models(directory=input_directory, show_available=False, show_supported=False)
    requested_models = list(np.atleast_1d(model)) if model is not None else available_models
    available_netcdf_models = list(set(available_models) & set(requested_models) & set(netcdf_models))

    # Raise error if no valid models found
    if len(available_netcdf_models) == 0:
        raise ValueError(f"No valid NetCDF models found in {input_directory}.")

    # If model list is provided,
    print(f"Preparing to clip suitable NetCDF models: {available_netcdf_models}\n")

    # Loop through suitable models and export
    for m in available_netcdf_models:
        # Get model file and grid file list if they exist
        model_files = model_database[m].get("model_file", [])
        grid_file = model_database[m].get("grid_file", [])

        # Convert to list of strings and combine
        model_files = model_files if isinstance(model_files, list) else [model_files]
        grid_file = grid_file if isinstance(grid_file, list) else [grid_file]
        all_files = model_files + grid_file

        # Loop through each model file and clip
        for file in tqdm(all_files, desc=f"Clipping {m}"):
            # Skip if it exists in output directory
            if (output_directory / file).exists() and not overwrite:
                continue

            # Load model file
            nc = xr.open_mfdataset(input_directory / file)

            # Open file and clip according to model
            if m in (
                "GOT5.5",
                "GOT5.5_load",
                "GOT5.5_extrapolated",
                "GOT5.5D",
                "GOT5.5D_extrapolated",
                "GOT5.6",
                "GOT5.6_extrapolated",
            ):
                nc_clipped = _clip_model_file(
                    nc,
                    bbox,
                    xdim="lon",
                    ydim="lat",
                    ycoord="latitude",
                    xcoord="longitude",
                )

            elif m in ("HAMTIDE11",):
                nc_clipped = _clip_model_file(nc, bbox, xdim="LON", ydim="LAT", ycoord="LAT", xcoord="LON")

            elif m in (
                "EOT20",
                "EOT20_load",
                "FES2012",
                "FES2014",
                "FES2014_extrapolated",
                "FES2014_load",
                "FES2022",
                "FES2022_extrapolated",
                "FES2022_load",
            ):
                nc_clipped = _clip_model_file(nc, bbox, xdim="lon", ydim="lat", ycoord="lat", xcoord="lon")

            elif m in (
                "TPXO8-atlas-nc",
                "TPXO9-atlas-nc",
                "TPXO9-atlas-v2-nc",
                "TPXO9-atlas-v3-nc",
                "TPXO9-atlas-v4-nc",
                "TPXO9-atlas-v5-nc",
                "TPXO10-atlas-v2-nc",
            ):
                nc_clipped = _clip_model_file(
                    nc,
                    bbox,
                    xdim="nx",
                    ydim="ny",
                    ycoord="lat_z",
                    xcoord="lon_z",
                )

            else:
                raise Exception(f"Model {m} not supported")

            # Create directory and export
            (output_directory / file).parent.mkdir(parents=True, exist_ok=True)
            nc_clipped.to_netcdf(output_directory / file, mode="w")

        # Verify that models are ready
        pytmd_model(directory=output_directory).elevation(m=m).verify
        print(" ✅ Clipped model exported and verified")

    print(f"\nOutputs exported to {output_directory}")
    list_models(directory=output_directory, show_available=True, show_supported=False)


def list_models(
    directory: str | os.PathLike | None = None,
    show_available: bool = True,
    show_supported: bool = True,
    raise_error: bool = False,
) -> tuple[list[str], list[str]]:
    """
    List all tide models available for tide modelling.

    This function scans the specified tide model directory
    and returns a list of models that are available in the
    directory as well as the full list of all models supported
    by `eo-tides` and `pyTMD`.

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

        # Handle GOT5.6 differently to ensure we test for presence of GOT5.6 constituents
        if m in ("GOT5.6", "GOT5.6_extrapolated"):
            model_file = [file for file in model_file if "GOT5.6" in file][0]
        else:
            model_file = model_file[0] if isinstance(model_file, list) else model_file

        # Add path to dict
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
            model_file = pytmd_model(directory=directory).elevation(m=m)
            available_models.append(m)

            if show_available:
                # Mark available models with a green tick
                status = "✅"
                print(f"{status:^{status_width}}│ {m:<{name_width}} │ {expected_paths[m]:<{path_width}}")
        except FileNotFoundError:
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
        warning_msg = textwrap.dedent(
            f"""
            No valid tide models are available in `{directory}`.
            Are you sure you have provided the correct `directory` path, or set the
            `EO_TIDES_TIDE_MODELS` environment variable to point to the location of your
            tide model directory?
            """
        ).strip()

        if raise_error:
            raise Exception(warning_msg)
        else:
            warnings.warn(warning_msg, UserWarning)

    # Return list of available and supported models
    return available_models, supported_models


def idw(
    input_z,
    input_x,
    input_y,
    output_x,
    output_y,
    p=1,
    k=10,
    max_dist=None,
    k_min=1,
    epsilon=1e-12,
):
    """Perform Inverse Distance Weighting (IDW) interpolation.

    This function performs fast IDW interpolation by creating a KDTree
    from the input coordinates then uses it to find the `k` nearest
    neighbors for each output point. Weights are calculated based on the
    inverse distance to each neighbor, with weights descreasing with
    increasing distance.

    Code inspired by: <https://github.com/DahnJ/REM-xarray>

    Parameters
    ----------
    input_z : array-like
        Array of values at the input points. This can be either a
        1-dimensional array, or a 2-dimensional array where each column
        (axis=1) represents a different set of values to be interpolated.
    input_x : array-like
        Array of x-coordinates of the input points.
    input_y : array-like
        Array of y-coordinates of the input points.
    output_x : array-like
        Array of x-coordinates where the interpolation is to be computed.
    output_y : array-like
        Array of y-coordinates where the interpolation is to be computed.
    p : int or float, optional
        Power function parameter defining how rapidly weightings should
        decrease as distance increases. Higher values of `p` will cause
        weights for distant points to decrease rapidly, resulting in
        nearby points having more influence on predictions. Defaults to 1.
    k : int, optional
        Number of nearest neighbors to use for interpolation. `k=1` is
        equivalent to "nearest" neighbour interpolation. Defaults to 10.
    max_dist : int or float, optional
        Restrict neighbouring points to less than this distance.
        By default, no distance limit is applied.
    k_min : int, optional
        If `max_dist` is provided, some points may end up with less than
        `k` nearest neighbours, potentially producing less reliable
        interpolations. Set `k_min` to set any points with less than
        `k_min` neighbours to NaN. Defaults to 1.
    epsilon : float, optional
        Small value added to distances to prevent division by zero
        errors in the case that output coordinates are identical to
        input coordinates. Defaults to 1e-12.

    Returns
    -------
    interp_values : numpy.ndarray
        Interpolated values at the output coordinates. If `input_z` is
        1-dimensional, `interp_values` will also be 1-dimensional. If
        `input_z` is 2-dimensional, `interp_values` will have the same
        number of rows as `input_z`, with each column (axis=1)
        representing interpolated values for one set of input data.

    Examples
    --------
    >>> input_z = [1, 2, 3, 4, 5]
    >>> input_x = [0, 1, 2, 3, 4]
    >>> input_y = [0, 1, 2, 3, 4]
    >>> output_x = [0.5, 1.5, 2.5]
    >>> output_y = [0.5, 1.5, 2.5]
    >>> idw(input_z, input_x, input_y, output_x, output_y, k=2)
    array([1.5, 2.5, 3.5])

    """
    # Convert to numpy arrays
    input_x = np.atleast_1d(input_x)
    input_y = np.atleast_1d(input_y)
    input_z = np.atleast_1d(input_z)
    output_x = np.atleast_1d(output_x)
    output_y = np.atleast_1d(output_y)

    # Verify input and outputs have matching lengths
    if not (input_z.shape[0] == len(input_x) == len(input_y)):
        raise ValueError("All of `input_z`, `input_x` and `input_y` must be the same length.")
    if not (len(output_x) == len(output_y)):
        raise ValueError("Both `output_x` and `output_y` must be the same length.")

    # Verify k is smaller than total number of points, and non-zero
    if k > input_z.shape[0]:
        raise ValueError(
            f"The requested number of nearest neighbours (`k={k}`) "
            f"is smaller than the total number of points ({input_z.shape[0]}).",
        )
    if k == 0:
        raise ValueError("Interpolation based on `k=0` nearest neighbours is not valid.")

    # Create KDTree to efficiently find nearest neighbours
    points_xy = np.column_stack((input_y, input_x))
    tree = KDTree(points_xy)

    # Determine nearest neighbours and distances to each
    grid_stacked = np.column_stack((output_y, output_x))
    distances, indices = tree.query(grid_stacked, k=k, workers=-1)

    # If k == 1, add an additional axis for consistency
    if k == 1:
        distances = distances[..., np.newaxis]
        indices = indices[..., np.newaxis]

    # Add small epsilon to distances to prevent division by zero errors
    # if output coordinates are the same as input coordinates
    distances = np.maximum(distances, epsilon)

    # Set distances above max to NaN if specified
    if max_dist is not None:
        distances[distances > max_dist] = np.nan

    # Calculate weights based on distance to k nearest neighbours.
    weights = 1 / np.power(distances, p)
    weights = weights / np.nansum(weights, axis=1).reshape(-1, 1)

    # 1D case: Compute weighted sum of input_z values for each output point
    if input_z.ndim == 1:
        interp_values = np.nansum(weights * input_z[indices], axis=1)

    # 2D case: Compute weighted sum for each set of input_z values
    # weights[..., np.newaxis] adds a dimension for broadcasting
    else:
        interp_values = np.nansum(
            weights[..., np.newaxis] * input_z[indices],
            axis=1,
        )

    # Set any points with less than `k_min` valid weights to NaN
    interp_values[np.isfinite(weights).sum(axis=1) < k_min] = np.nan

    return interp_values
