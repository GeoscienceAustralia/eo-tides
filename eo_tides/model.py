import os
import pathlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import geopandas as gpd
import numpy as np
import odc.geo.xr
import pandas as pd
import pyproj
import pyTMD
from tqdm import tqdm

from eo_tides.utils import idw


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
    # import pyTMD.eop
    # import pyTMD.io
    # import pyTMD.io.model
    # import pyTMD.predict
    # import pyTMD.spatial
    # import pyTMD.time
    # import pyTMD.utilities

    # Get parameters for tide model; use custom definition file for
    # FES2012 (leave this as an undocumented feature for now)
    # if model == "FES2012":
    #     pytmd_model = pyTMD.io.model(directory).from_file(
    #         directory / "model_FES2012.def"
    #     )
    # elif model == "TPXO8-atlas-v1":
    #     pytmd_model = pyTMD.io.model(directory).from_file(directory / "model_TPXO8.def")
    # else:
    #     pytmd_model = pyTMD.io.model(
    #         directory, format="netcdf", compressed=False
    #     ).elevation(model)

    #     if model in NONSTANDARD_MODELS:
    #         model_params = NONSTANDARD_MODELS[model]
    #         model_params_bytes = io.BytesIO(json.dumps(model_params).encode("utf-8"))
    #         pytmd_model = pyTMD.io.model(directory).from_file(definition_file=model_params_bytes)

    #     else:

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
            grid=pytmd_model.type,
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
        "tide_m": tide,
    }).set_index(["time", "x", "y"])

    # Optionally convert outputs to integer units (can save memory)
    if output_units == "m":
        tide_df["tide_m"] = tide_df.tide_m.astype(np.float32)
    elif output_units == "cm":
        tide_df["tide_m"] = (tide_df.tide_m * 100).astype(np.int16)
    elif output_units == "mm":
        tide_df["tide_m"] = (tide_df.tide_m * 1000).astype(np.int16)

    return tide_df


def _ensemble_model(
    x,
    y,
    crs,
    tide_df,
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
    1. Loads model ranking points from a GeoJSON file, filters them
       based on the valid data percentage, and retains relevant columns
    2. Interpolates the model rankings into the requested x and y
       coordinates using Inverse Weighted Interpolation (IDW)
    3. Uses rankings to combine multiple tide models into a single
       optimised ensemble model (by default, by taking the mean of the
       top 3 ranked models)
    4. Returns a DataFrame with the combined ensemble model predictions

    Parameters
    ----------
    x : array-like
        Array of x-coordinates where the ensemble model predictions are
        required.
    y : array-like
        Array of y-coordinates where the ensemble model predictions are
        required.
    crs : string
        Input coordinate reference system for x and y coordinates. Used
        to ensure that interpolations are performed in the correct CRS.
    tide_df : pandas.DataFrame
        DataFrame containing tide model predictions with columns
        `["time", "x", "y", "tide_m", "tide_model"]`.
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
        model, named with the prefix "rank_". e.g. "rank_FES2014".
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
        "y", "tide_m", "tide_model"]`. By default the 'tide_model'
        column will be labeled "ensemble" for the combined model
        predictions (but if a custom dictionary of ensemble functions is
        provided via `ensemble_func`, each ensemble will be named using
        the provided dictionary keys).

    """
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
                weighted=lambda i: i.tide_m * i.weights,
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
            .to_frame("tide_m")
            # Label ensemble model and ensure indexes are in expected order
            .assign(tide_model=ensemble_n)
            .reorder_levels(["time", "x", "y"], axis=0)
        )

        ensemble_list.append(ensemble_df)

    # Combine all ensemble models and return as a single dataframe
    return pd.concat(ensemble_list)


def model_tides(
    x,
    y,
    time,
    model="FES2014",
    directory=None,
    crs="EPSG:4326",
    crop=True,
    method="spline",
    extrapolate=True,
    cutoff=None,
    mode="one-to-many",
    parallel=True,
    parallel_splits=5,
    output_units="m",
    output_format="long",
    ensemble_models=None,
    **ensemble_kwargs,
):
    """Compute tides at multiple points and times using tidal harmonics.

    This function supports all tidal models supported by `pyTMD`,
    including FES Finite Element Solution models, TPXO TOPEX/POSEIDON
    models, EOT Empirical Ocean Tide models, GOT Global Ocean Tide
    models, and HAMTIDE Hamburg direct data Assimilation Methods for
    Tides models.

    This function requires access to tide model data files.
    These should be placed in a folder with subfolders matching
    the formats specified by `pyTMD`:
    <https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories>

    For FES2014 (<https://www.aviso.altimetry.fr/es/data/products/auxiliary-products/global-tide-fes/description-fes2014.html>):

    - `{directory}/fes2014/ocean_tide/`

    For FES2022 (<https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html>):

    - `{directory}/fes2022b/ocean_tide/`

    For TPXO8-atlas (<https://www.tpxo.net/tpxo-products-and-registration>):

    - `{directory}/tpxo8_atlas/`

    For TPXO9-atlas-v5 (<https://www.tpxo.net/tpxo-products-and-registration>):

    - `{directory}/TPXO9_atlas_v5/`

    For EOT20 (<https://www.seanoe.org/data/00683/79489/>):

    - `{directory}/EOT20/ocean_tides/`

    For GOT4.10c (<https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models>):

    - `{directory}/GOT4.10c/grids_oceantide_netcdf/`

    For HAMTIDE (<https://www.cen.uni-hamburg.de/en/icdc/data/ocean/hamtide.html>):

    - `{directory}/hamtide/`

    This function is a modification of the `pyTMD` package's
    `compute_tide_corrections` function. For more info:
    <https://pytmd.readthedocs.io/en/stable/user_guide/compute_tide_corrections.html>

    Parameters
    ----------
    x, y : float or list of floats
        One or more x and y coordinates used to define
        the location at which to model tides. By default these
        coordinates should be lat/lon; use "crs" if they
        are in a custom coordinate reference system.
    time : A datetime array or pandas.DatetimeIndex
        An array containing `datetime64[ns]` values or a
        `pandas.DatetimeIndex` providing the times at which to
        model tides in UTC time.
    model : string, optional
        The tide model used to model tides. Options include:

        - "FES2014" (pre-configured on DEA Sandbox)
        - "FES2022"
        - "TPXO9-atlas-v5"
        - "TPXO8-atlas"
        - "EOT20"
        - "HAMTIDE11"
        - "GOT4.10"
        - "ensemble" (advanced ensemble tide model functionality;
          combining multiple models based on external model rankings)
    directory : string, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, otherwise "/var/share/tide_models".
        Tide modelling files should be stored in sub-folders for each
        model that match the structure provided by `pyTMD`.

        For example:

        - `{directory}/fes2014/ocean_tide/`
        - `{directory}/tpxo8_atlas/`
        - `{directory}/TPXO9_atlas_v5/`
    crs : str, optional
        Input coordinate reference system for x and y coordinates.
        Defaults to "EPSG:4326" (WGS84; degrees latitude, longitude).
    crop : bool optional
        Whether to crop tide model constituent files on-the-fly to
        improve performance. Cropping will be performed based on a
        1 degree buffer around all input points. Defaults to True.
    method : string, optional
        Method used to interpolate tidal constituents
        from model files. Options include:

        - "spline": scipy bivariate spline interpolation (default)
        - "bilinear": quick bilinear interpolation
        - "linear", "nearest": scipy regular grid interpolations
    extrapolate : bool, optional
        Whether to extrapolate tides for x and y coordinates outside of
        the valid tide modelling domain using nearest-neighbor.
    cutoff : int or float, optional
        Extrapolation cutoff in kilometers. The default is None, which
        will extrapolate for all points regardless of distance from the
        valid tide modelling domain.
    mode : string, optional
        The analysis mode to use for tide modelling. Supports two options:

        - "one-to-many": Models tides for every timestep in "time" at
        every input x and y coordinate point. This is useful if you
        want to model tides for a specific list of timesteps across
        multiple spatial points (e.g. for the same set of satellite
        acquisition times at various locations across your study area).
        - "one-to-one": Model tides using a different timestep for each
        x and y coordinate point. In this mode, the number of x and
        y points must equal the number of timesteps provided in "time".
    parallel : boolean, optional
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
        results stacked vertically along "tide_model" and "tide_m"
        columns), or wide format (with a column for each tide model).
        Defaults to "long".
    ensemble_models : list, optional
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
    # Set tide modelling files directory. If no custom path is provided,
    # first try global environmental var, then "/var/share/tide_models"
    if directory is None:
        if "EO_TIDES_TIDE_MODELS" in os.environ:
            directory = os.environ["EO_TIDES_TIDE_MODELS"]
        else:
            directory = "/var/share/tide_models"

    # Verify path exists
    directory = pathlib.Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError("Invalid tide directory")

    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Turn inputs into arrays for consistent handling
    models_requested = np.atleast_1d(model)
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

    # Verify that all provided models are supported
    valid_models = [
        # Standard built-in pyTMD models
        "EOT20",
        "FES2014",
        "FES2022",
        "GOT4.10",
        "HAMTIDE11",
        "TPXO8-atlas",  # binary version, not suitable for clipping
        "TPXO9-atlas-v5",
        # Non-standard models, defined internally
        "FES2012",
        "FES2014_extrapolated",
        "FES2022_extrapolated",
        "GOT5.6",
        "GOT5.6_extrapolated",
        "TPXO8-atlas-v1",  # netCDF version
        # Advanced ensemble model functionality
        "ensemble",
    ]
    if not all(m in valid_models for m in models_requested):
        raise ValueError(
            f"One or more of the models requested {models_requested} is "
            f"not valid. The following models are currently supported: "
            f"{valid_models}",
        )

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
        ensemble_df = _ensemble_model(x, y, crs, tide_df, models_to_process, **ensemble_kwargs)

        # Update requested models with any custom ensemble models, then
        # filter the dataframe to keep only models originally requested
        models_requested = np.union1d(models_requested, ensemble_df.tide_model.unique())
        tide_df = pd.concat([tide_df, ensemble_df]).query("tide_model in @models_requested")

    # Optionally convert to a wide format dataframe with a tide model in
    # each dataframe column
    if output_format == "wide":
        # Pivot into wide format with each time model as a column
        print("Converting to a wide format dataframe")
        tide_df = tide_df.pivot(columns="tide_model", values="tide_m")

        # If in 'one-to-one' mode, reindex using our input time/x/y
        # values to ensure the output is sorted the same as our inputs
        if mode == "one-to-one":
            output_indices = pd.MultiIndex.from_arrays([time, x, y], names=["time", "x", "y"])
            tide_df = tide_df.reindex(output_indices)

    return tide_df


def _pixel_tides_resample(
    tides_lowres,
    ds,
    resample_method="bilinear",
    dask_chunks="auto",
    dask_compute=True,
):
    """Resamples low resolution tides modelled by `pixel_tides` into the
    geobox (e.g. spatial resolution and extent) of the original higher
    resolution satellite dataset.

    Parameters
    ----------
    tides_lowres : xarray.DataArray
        The low resolution tide modelling data array to be resampled.
    ds : xarray.Dataset
        The dataset whose geobox will be used as the template for the
        resampling operation. This is typically the same satellite
        dataset originally passed to `pixel_tides`.
    resample_method : string, optional
        The resampling method to use. Defaults to "bilinear"; valid
        options include "nearest", "cubic", "min", "max", "average" etc.
    dask_chunks : str or tuple, optional
        Can be used to configure custom Dask chunking for the final
        resampling step. The default of "auto" will automatically set
        x/y chunks to match those in `ds` if they exist, otherwise will
        set x/y chunks that cover the entire extent of the dataset.
        For custom chunks, provide a tuple in the form `(y, x)`, e.g.
        `(2048, 2048)`.
    dask_compute : bool, optional
        Whether to compute results of the resampling step using Dask.
        If False, this will return `tides_highres` as a Dask array.

    Returns
    -------
    tides_highres, tides_lowres : tuple of xr.DataArrays
        In addition to `tides_lowres` (see above), a high resolution
        array of tide heights will be generated matching the
        exact spatial resolution and extent of `ds`.

    """
    # Determine spatial dimensions
    y_dim, x_dim = ds.odc.spatial_dims

    # Convert array to Dask, using no chunking along y and x dims,
    # and a single chunk for each timestep/quantile and tide model
    tides_lowres_dask = tides_lowres.chunk({d: None if d in [y_dim, x_dim] else 1 for d in tides_lowres.dims})

    # Automatically set Dask chunks for reprojection if set to "auto".
    # This will either use x/y chunks if they exist in `ds`, else
    # will cover the entire x and y dims) so we don't end up with
    # hundreds of tiny x and y chunks due to the small size of
    # `tides_lowres` (possible odc.geo bug?)
    if dask_chunks == "auto":
        if ds.chunks is not None:
            if (y_dim in ds.chunks) & (x_dim in ds.chunks):
                dask_chunks = (ds.chunks[y_dim], ds.chunks[x_dim])
            else:
                dask_chunks = ds.odc.geobox.shape
        else:
            dask_chunks = ds.odc.geobox.shape

    # Reproject into the GeoBox of `ds` using odc.geo and Dask
    tides_highres = tides_lowres_dask.odc.reproject(
        how=ds.odc.geobox,
        chunks=dask_chunks,
        resampling=resample_method,
    ).rename("tide_m")

    # Optionally process and load into memory with Dask
    if dask_compute:
        tides_highres.load()

    return tides_highres, tides_lowres


def pixel_tides(
    ds,
    times=None,
    resample=True,
    calculate_quantiles=None,
    resolution=None,
    buffer=None,
    resample_method="bilinear",
    model="FES2014",
    dask_chunks="auto",
    dask_compute=True,
    **model_tides_kwargs,
):
    """Obtain tide heights for each pixel in a dataset by modelling
    tides into a low-resolution grid surrounding the dataset,
    then (optionally) spatially resample this low-res data back
    into the original higher resolution dataset extent and resolution.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset whose geobox (`ds.odc.geobox`) will be used to define
        the spatial extent of the low resolution tide modelling grid.
    times : pandas.DatetimeIndex or list of pandas.Timestamps, optional
        By default, the function will model tides using the times
        contained in the `time` dimension of `ds`. Alternatively, this
        param can be used to model tides for a custom set of times
        instead. For example:
        `times=pd.date_range(start="2000", end="2001", freq="5h")`
    resample : bool, optional
        Whether to resample low resolution tides back into `ds`'s original
        higher resolution grid. Set this to `False` if you do not want
        low resolution tides to be re-projected back to higher resolution.
    calculate_quantiles : list or np.array, optional
        Rather than returning all individual tides, low-resolution tides
        can be first aggregated using a quantile calculation by passing in
        a list or array of quantiles to compute. For example, this could
        be used to calculate the min/max tide across all times:
        `calculate_quantiles=[0.0, 1.0]`.
    resolution : int, optional
        The desired resolution of the low-resolution grid used for tide
        modelling. The default None will create a 5000 m resolution grid
        if `ds` has a projected CRS (i.e. metre units), or a 0.05 degree
        resolution grid if `ds` has a geographic CRS (e.g. degree units).
        Note: higher resolutions do not necessarily provide better
        tide modelling performance, as results will be limited by the
        resolution of the underlying global tide model (e.g. 1/16th
        degree / ~5 km resolution grid for FES2014).
    buffer : int, optional
        The amount by which to buffer the higher resolution grid extent
        when creating the new low resolution grid. This buffering is
        important as it ensures that ensure pixel-based tides are seamless
        across dataset boundaries. This buffer will eventually be clipped
        away when the low-resolution data is re-projected back to the
        resolution and extent of the higher resolution dataset. To
        ensure that at least two pixels occur outside of the dataset
        bounds, the default None applies a 12000 m buffer if `ds` has a
        projected CRS (i.e. metre units), or a 0.12 degree buffer if
        `ds` has a geographic CRS (e.g. degree units).
    resample_method : string, optional
        If resampling is requested (see `resample` above), use this
        resampling method when converting from low resolution to high
        resolution pixels. Defaults to "bilinear"; valid options include
        "nearest", "cubic", "min", "max", "average" etc.
    model : string or list of strings
        The tide model or a list of models used to model tides, as
        supported by the `pyTMD` Python package. Options include:
        - "FES2014" (default; pre-configured on DEA Sandbox)
        - "FES2022"
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
        - "EOT20"
        - "HAMTIDE11"
        - "GOT4.10"
    dask_chunks : str or tuple, optional
        Can be used to configure custom Dask chunking for the final
        resampling step. The default of "auto" will automatically set
        x/y chunks to match those in `ds` if they exist, otherwise will
        set x/y chunks that cover the entire extent of the dataset.
        For custom chunks, provide a tuple in the form `(y, x)`, e.g.
        `(2048, 2048)`.
    dask_compute : bool, optional
        Whether to compute results of the resampling step using Dask.
        If False, this will return `tides_highres` as a Dask array.
    **model_tides_kwargs :
        Optional parameters passed to the `dea_tools.coastal.model_tides`
        function. Important parameters include "directory" (used to
        specify the location of input tide modelling files) and "cutoff"
        (used to extrapolate modelled tides away from the coast; if not
        specified here, cutoff defaults to `np.inf`).

    Returns
    -------
    If `resample` is False:

        tides_lowres : xr.DataArray
            A low resolution data array giving either tide heights every
            timestep in `ds` (if `times` is None), tide heights at every
            time in `times` (if `times` is not None), or tide height quantiles
            for every quantile provided by `calculate_quantiles`.

    If `resample` is True:

        tides_highres, tides_lowres : tuple of xr.DataArrays
            In addition to `tides_lowres` (see above), a high resolution
            array of tide heights will be generated that matches the
            exact spatial resolution and extent of `ds`. This will contain
            either tide heights every timestep in `ds` (if `times` is None),
            tide heights at every time in `times` (if `times` is not None),
            or tide height quantiles for every quantile provided by
            `calculate_quantiles`.

    """
    from odc.geo.geobox import GeoBox

    # First test if no time dimension and nothing passed to `times`
    if ("time" not in ds.dims) & (times is None):
        raise ValueError(
            "`ds` does not contain a 'time' dimension. Times are required "
            "for modelling tides: please pass in a set of custom tides "
            "using the `times` parameter. For example: "
            "`times=pd.date_range(start='2000', end='2001', freq='5h')`",
        )

    # If custom times are provided, convert them to a consistent
    # pandas.DatatimeIndex format
    if times is not None:
        if isinstance(times, list):
            time_coords = pd.DatetimeIndex(times)
        elif isinstance(times, pd.Timestamp):
            time_coords = pd.DatetimeIndex([times])
        else:
            time_coords = times

    # Otherwise, use times from `ds` directly
    else:
        time_coords = ds.coords["time"]

    # Set defaults passed to `model_tides`
    model_tides_kwargs.setdefault("cutoff", np.inf)

    # Standardise model into a list for easy handling
    model = [model] if isinstance(model, str) else model

    # Test if no time dimension and nothing passed to `times`
    if ("time" not in ds.dims) & (times is None):
        raise ValueError(
            "`ds` does not contain a 'time' dimension. Times are required "
            "for modelling tides: please pass in a set of custom tides "
            "using the `times` parameter. For example: "
            "`times=pd.date_range(start='2000', end='2001', freq='5h')`",
        )

    # If custom times are provided, convert them to a consistent
    # pandas.DatatimeIndex format
    if times is not None:
        if isinstance(times, list):
            time_coords = pd.DatetimeIndex(times)
        elif isinstance(times, pd.Timestamp):
            time_coords = pd.DatetimeIndex([times])
        else:
            time_coords = times

    # Otherwise, use times from `ds` directly
    else:
        time_coords = ds.coords["time"]

    # Determine spatial dimensions
    y_dim, x_dim = ds.odc.spatial_dims

    # Determine resolution and buffer, using different defaults for
    # geographic (i.e. degrees) and projected (i.e. metres) CRSs:
    crs_units = ds.odc.geobox.crs.units[0][0:6]
    if ds.odc.geobox.crs.geographic:
        if resolution is None:
            resolution = 0.05
        elif resolution > 360:
            raise ValueError(
                f"A resolution of greater than 360 was "
                f"provided, but `ds` has a geographic CRS "
                f"in {crs_units} units. Did you accidently "
                f"provide a resolution in projected "
                f"(i.e. metre) units?",
            )
        if buffer is None:
            buffer = 0.12
    else:
        if resolution is None:
            resolution = 5000
        elif resolution < 1:
            raise ValueError(
                f"A resolution of less than 1 was provided, "
                f"but `ds` has a projected CRS in "
                f"{crs_units} units. Did you accidently "
                f"provide a resolution in geographic "
                f"(degree) units?",
            )
        if buffer is None:
            buffer = 12000

    # Raise error if resolution is less than dataset resolution
    dataset_res = ds.odc.geobox.resolution.x
    if resolution < dataset_res:
        raise ValueError(
            f"The resolution of the low-resolution tide "
            f"modelling grid ({resolution:.2f}) is less "
            f"than `ds`'s pixel resolution ({dataset_res:.2f}). "
            f"This can cause extremely slow tide modelling "
            f"performance. Please select provide a resolution "
            f"greater than {dataset_res:.2f} using "
            f"`pixel_tides`'s 'resolution' parameter.",
        )

    # Create a new reduced resolution tide modelling grid after
    # first buffering the grid
    print(f"Creating reduced resolution {resolution} x {resolution} {crs_units} tide modelling array")
    buffered_geobox = ds.odc.geobox.buffered(buffer)
    rescaled_geobox = GeoBox.from_bbox(bbox=buffered_geobox.boundingbox, resolution=resolution)
    rescaled_ds = odc.geo.xr.xr_zeros(rescaled_geobox)

    # Flatten grid to 1D, then add time dimension
    flattened_ds = rescaled_ds.stack(z=(x_dim, y_dim))
    flattened_ds = flattened_ds.expand_dims(dim={"time": time_coords.values})

    # Model tides in parallel, returning a pandas.DataFrame
    tide_df = model_tides(
        x=flattened_ds[x_dim],
        y=flattened_ds[y_dim],
        time=flattened_ds.time,
        crs=f"EPSG:{ds.odc.geobox.crs.epsg}",
        model=model,
        **model_tides_kwargs,
    )

    # Convert our pandas.DataFrame tide modelling outputs to xarray
    tides_lowres = (
        # Rename x and y dataframe indexes to match x and y xarray dims
        tide_df.rename_axis(["time", x_dim, y_dim])
        # Add tide model column to dataframe indexes so we can convert
        # our dataframe to a multidimensional xarray
        .set_index("tide_model", append=True)
        # Convert to xarray and select our tide modelling xr.DataArray
        .to_xarray()
        .tide_m
        # Re-index and transpose into our input coordinates and dim order
        .reindex_like(rescaled_ds)
        .transpose("tide_model", "time", y_dim, x_dim)
    )

    # Optionally calculate and return quantiles rather than raw data.
    # Set dtype to dtype of the input data as quantile always returns
    # float64 (memory intensive)
    if calculate_quantiles is not None:
        print("Computing tide quantiles")
        tides_lowres = tides_lowres.quantile(q=calculate_quantiles, dim="time").astype(tides_lowres.dtype)

    # If only one tidal model exists, squeeze out "tide_model" dim
    if len(tides_lowres.tide_model) == 1:
        tides_lowres = tides_lowres.squeeze("tide_model")

    # Ensure CRS is present before we apply any resampling
    tides_lowres = tides_lowres.odc.assign_crs(ds.odc.geobox.crs)

    # Reproject into original high resolution grid
    if resample:
        print("Reprojecting tides into original array")
        tides_highres, tides_lowres = _pixel_tides_resample(
            tides_lowres,
            ds,
            resample_method,
            dask_chunks,
            dask_compute,
        )
        return tides_highres, tides_lowres

    print("Returning low resolution tide array")
    return tides_lowres


if __name__ == "__main__":  # pragma: no cover
    pass
