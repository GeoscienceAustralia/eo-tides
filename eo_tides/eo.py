"""Tools for integrating satellite EO data with tide modelling.

This module provides tools for combining satellite EO observations with
tide heights and phases using one or more ocean tide models, both at
the timestep and at the pixel level.
"""

# Used to postpone evaluation of type annotations
from __future__ import annotations

import textwrap
import warnings
from typing import TYPE_CHECKING

import numpy as np
import odc.geo.xr
import xarray as xr
from odc.geo.geobox import GeoBox

# Only import if running type checking
if TYPE_CHECKING:
    import os

    from odc.geo import Shape2d

from .model import model_phases, model_tides
from .utils import DatetimeLike, _standardise_time


def _resample_chunks(
    data: xr.DataArray | xr.Dataset | GeoBox,
    dask_chunks: tuple | None = None,
) -> tuple | Shape2d:
    """Create optimised dask chunks for reprojection.

    Automatically return optimised dask chunks
    for reprojection with `_pixel_tides_resample`.
    Use entire image if GeoBox or if no default
    chunks; use existing chunks if they exist.
    """
    # If dask_chunks is provided, return directly
    if dask_chunks is not None:
        return dask_chunks

    # If data is a GeoBox, return its shape
    if isinstance(data, GeoBox):
        return data.shape

    # if data has chunks, then return just spatial chunks
    if data.chunks:
        y_dim, x_dim = data.odc.spatial_dims
        return data.chunks[y_dim], data.chunks[x_dim]

    # if data has no chunks, then return entire image shape
    return data.odc.geobox.shape


def _standardise_inputs(
    data: xr.DataArray | xr.Dataset | GeoBox,
    time: DatetimeLike | None,
) -> tuple[GeoBox, np.ndarray | None]:
    """Standardise location and time inputs to tide modelling functions.

    Takes an xarray or GeoBox input and an optional custom times,
    and returns a standardised GeoBox and times (usually an
    array, but possibly None).
    """
    # If `data` is an xarray object, extract its GeoBox and time
    if isinstance(data, xr.DataArray | xr.Dataset):
        # Try to extract GeoBox
        try:
            gbox: GeoBox = data.odc.geobox
        except AttributeError:
            error_msg = """
            Cannot extract a valid GeoBox for `data`. This is required for
            extracting details about `data`'s CRS and spatial location.

            Import `odc.geo.xr` then run `data = data.odc.assign_crs(crs=...)`
            to prepare your data before passing it to this function.
            """
            raise Exception(textwrap.dedent(error_msg).strip()) from None

        # Use custom time by default if provided; otherwise try and extract from `data`
        if time is not None:
            time = _standardise_time(time)
        elif "time" in data.dims:
            time = np.asarray(data.coords["time"].values)
        else:
            err_msg = "`data` does not have a 'time' dimension, and no custom times were provided via `time`."
            raise ValueError(err_msg)

    # If `data` is a GeoBox, use it directly; raise an error if no time was provided
    elif isinstance(data, GeoBox):
        gbox = data
        if time is not None:
            time = _standardise_time(time)
        else:
            err_msg = "If `data` is a GeoBox, custom times must be provided via `time`."
            raise ValueError(err_msg)

    # Raise error if no valid inputs were provided
    else:
        err_msg = "`data` must be an xarray.DataArray, xarray.Dataset, or odc.geo.geobox.GeoBox."
        raise TypeError(err_msg)

    return gbox, time


def _pixel_tides_resample(
    tides_lowres,
    gbox,
    resample_method="bilinear",
    dask_chunks=None,
    dask_compute=True,
    name="tide_height",
):
    """Resample low resolution tides modelled by `pixel_tides` to higher resolution.

    Uses `odc-geo` to reproject data to match the geobox (e.g.
    spatial resolution and extent) of the original higher resolution
    satellite dataset.

    Parameters
    ----------
    tides_lowres : xarray.DataArray
        The low resolution tide modelling data array to be resampled.
    gbox : GeoBox
        The GeoBox to use as the template for the resampling operation.
        This is typically comes from the same satellite dataset originally
        passed to `pixel_tides` (e.g. `data.odc.geobox`).
    resample_method : string, optional
        The resampling method to use. Defaults to "bilinear"; valid
        options include "nearest", "cubic", "min", "max", "average" etc.
    dask_chunks : tuple of float, optional
        Can be used to configure custom Dask chunking for the final
        resampling step. For custom chunks, provide a tuple in the form
        (y, x), e.g. (2048, 2048).
    dask_compute : bool, optional
        Whether to compute results of the resampling step using Dask.
        If False, this will return `tides_highres` as a lazy loaded
        Dask-enabled array.
    name : str, optional
        The name used for the output array. Defaults to "tide_height".

    Returns
    -------
    tides_highres : xr.DataArray
        A high resolution array of tide heights matching the exact
        spatial resolution and extent of `gbox`.

    """
    # Determine spatial dimensions
    y_dim, x_dim = gbox.dimensions

    # Convert array to Dask, using no chunking along y and x dims,
    # and a single chunk for each timestep/quantile and tide model
    tides_lowres_dask = tides_lowres.chunk({d: None if d in [y_dim, x_dim] else 1 for d in tides_lowres.dims})

    # Reproject into the pixel grid of `gbox` using odc.geo and Dask
    tides_highres = tides_lowres_dask.odc.reproject(
        how=gbox,
        chunks=dask_chunks,
        resampling=resample_method,
    )

    # Set output name
    if name is not None:
        tides_highres = tides_highres.rename(name)

    # Optionally process and load into memory with Dask
    if dask_compute:
        tides_highres.load()

    return tides_highres


def tag_tides(
    data: xr.Dataset | xr.DataArray | GeoBox,
    time: DatetimeLike | None = None,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    tidepost_lat: float | None = None,
    tidepost_lon: float | None = None,
    return_phases: bool = False,
    **model_tides_kwargs,
) -> xr.DataArray | xr.Dataset:
    """Model tide heights and phases for every dataset timestep using multiple ocean tide models.

    Tides are modelled using the centroid of the dataset by
    default; use `tidepost_lat` and `tidepost_lon` to specify
    a custom tidal modelling location.

    The function supports all tidal models supported by `pyTMD`,
    including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution tide models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global tide models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg direct data Assimilation Methods for Tides models (HAMTIDE11)
    - Technical University of Denmark tide models (DTU23)

    This function requires access to tide model data files.
    For tide model setup instructions, refer to the guide:
    https://geoscienceaustralia.github.io/eo-tides/setup/

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or odc.geo.geobox.GeoBox
        A multi-dimensional dataset or GeoBox pixel grid that will
        be used to define the tide modelling location. If `data`
        is an xarray object, it should include a "time" dimension.
        If no "time" dimension exists or if `data` is a GeoBox,
        then times must be passed using the `time` parameter.
    time : DatetimeLike, optional
        By default, tides will be modelled using times from the
        "time" dimension of `data`. Alternatively, this param can
        be used to provide a custom set of times. Accepts any format
        that can be converted by `pandas.to_datetime()`. For example:
        `time=pd.date_range(start="2000", end="2001", freq="5h")`
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        If a list is provided, a new "tide_model" dimension will be
        added to the `xarray.DataArray` outputs. Defaults to "EOT20";
        specify "all" to use all models available in `directory`.
        For a full list of available and supported models, run
        `from eo_tides.utils import list_models; list_models()`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    tidepost_lat, tidepost_lon : float, optional
        Optional coordinates used to model tides. The default is None,
        which uses the centroid of the dataset as the tide modelling
        location.
    return_phases : bool, optional
        Whether to model and return tide phases in addition to tide heights.
        If True, outputs will be returned as an xr.Dataset containing both
        "tide_height" and "tide_phase" variables.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

    Returns
    -------
    tides_da : xr.DataArray or xr.Dataset
        If `return_phases=False`: a one-dimensional "tide_height" xr.DataArray.
        If `return_phases=True`: a one-dimensional xr.Dataset containing
        "tide_height" and "tide_phase" variables.
        Outputs will contain values for every timestep in `data`, or for
        every time in `times` if provided.

    """
    # Standardise data inputs, time and models
    gbox, time_coords = _standardise_inputs(data, time)
    model = [model] if isinstance(model, str) else model

    # If custom tide posts are not provided, use dataset centroid
    if tidepost_lat is None or tidepost_lon is None:
        lon, lat = gbox.geographic_extent.centroid.coords[0]
        print(f"Setting tide modelling location from dataset centroid: {lon:.2f}, {lat:.2f}")
    else:
        lon, lat = tidepost_lon, tidepost_lat
        print(f"Using tide modelling location: {lon:.2f}, {lat:.2f}")

    # Either model both tides and phases, or model only tides
    if return_phases:
        # Model tide phases and heights for each observation
        tide_df = model_phases(
            x=lon,
            y=lat,
            time=time_coords,
            model=model,
            directory=directory,
            crs="EPSG:4326",
            return_tides=True,
            **model_tides_kwargs,
        )

    else:
        # Model tide heights for each observation
        tide_df = model_tides(
            x=lon,
            y=lat,
            time=time_coords,
            model=model,
            directory=directory,
            crs="EPSG:4326",
            **model_tides_kwargs,
        )

    # If tides cannot be successfully modeled (e.g. if the centre of the
    # xarray dataset is located is over land), raise an exception
    if tide_df.tide_height.isna().all():
        err_msg = (
            f"Tides could not be modelled for dataset centroid located "
            f"at {tidepost_lon:.2f}, {tidepost_lat:.2f}. This can occur if "
            f"this coordinate occurs over land. Please manually specify "
            f"a tide modelling location located over water using the "
            f"`tidepost_lat` and `tidepost_lon` parameters.",
        )
        raise ValueError(err_msg)

    # Convert to xarray format, squeezing to return an xr.DataArray if
    # dataframe contains only one "tide_height" column
    tides_da = tide_df.reset_index().set_index(["time", "tide_model"]).drop(["x", "y"], axis=1).squeeze().to_xarray()

    # If only one tidal model exists, squeeze out "tide_model" dim
    if len(tides_da.tide_model) == 1:
        tides_da = tides_da.squeeze("tide_model")

    return tides_da


def pixel_tides(
    data: xr.Dataset | xr.DataArray | GeoBox,
    time: DatetimeLike | None = None,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    resample: bool = True,
    calculate_quantiles: np.ndarray | tuple[float, float] | None = None,
    resolution: float | None = None,
    buffer: float | None = None,
    resample_method: str = "bilinear",
    dask_chunks: tuple[float, float] | None = None,
    dask_compute: bool = True,
    **model_tides_kwargs,
) -> xr.DataArray:
    """Model tide heights for every dataset pixel using multiple ocean tide models.

    This function models tides into a low-resolution tide
    modelling grid covering the spatial extent of the input
    data (buffered to reduce potential edge effects). These
    modelled tides can then be resampled back into the original
    higher resolution dataset's extent and resolution to
    produce a modelled tide height for every pixel through time.

    This function uses the parallelised `model_tides` function
    under the hood. It supports all tidal models supported by
    `pyTMD`, including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution tide models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global tide models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg direct data Assimilation Methods for Tides models (HAMTIDE11)
    - Technical University of Denmark tide models (DTU23)

    This function requires access to tide model data files.
    For tide model setup instructions, refer to the guide:
    https://geoscienceaustralia.github.io/eo-tides/setup/

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or odc.geo.geobox.GeoBox
        A multi-dimensional dataset or GeoBox pixel grid that will
        be used to define the spatial tide modelling grid. If `data`
        is an xarray object, it should include a "time" dimension.
        If no "time" dimension exists or if `data` is a GeoBox,
        then times must be passed using the `time` parameter.
    time : DatetimeLike, optional
        By default, tides will be modelled using times from the
        "time" dimension of `data`. Alternatively, this param can
        be used to provide a custom set of times. Accepts any format
        that can be converted by `pandas.to_datetime()`. For example:
        `time=pd.date_range(start="2000", end="2001", freq="5h")`
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        If a list is provided, a new "tide_model" dimension will be
        added to the `xarray.DataArray` outputs. Defaults to "EOT20";
        specify "all" to use all models available in `directory`.
        For a full list of available and supported models, run
        `from eo_tides.utils import list_models; list_models()`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    resample : bool, optional
        Whether to resample low resolution tides back into `data`'s original
        higher resolution grid. Set this to `False` if you do not want
        low resolution tides to be re-projected back to higher resolution.
    calculate_quantiles : tuple of float or numpy.ndarray, optional
        Rather than returning all individual tides, low-resolution tides
        can be first aggregated using a quantile calculation by passing in
        a tuple or array of quantiles to compute. For example, this could
        be used to calculate the min/max tide across all times:
        `calculate_quantiles=(0.0, 1.0)`.
    resolution : float, optional
        The desired resolution of the low-resolution grid used for tide
        modelling. The default None will create a 5000 m resolution grid
        if `data` has a projected CRS (i.e. metre units), or a 0.05 degree
        resolution grid if `data` has a geographic CRS (e.g. degree units).
        Note: higher resolutions do not necessarily provide better
        tide modelling performance, as results will be limited by the
        resolution of the underlying global tide model (e.g. 1/16th
        degree / ~5 km resolution grid for FES2014).
    buffer : float, optional
        The amount by which to buffer the higher resolution grid extent
        when creating the new low resolution grid. This buffering
        ensures that modelled tides are seamless across analysis
        boundaries. This buffer is eventually be clipped away when
        the low-resolution modelled tides are re-projected back to the
        original resolution and extent of `data`. To ensure that at least
        two low-resolution grid pixels occur outside of the dataset
        bounds, the default None applies a 12000 m buffer if `data` has a
        projected CRS (i.e. metre units), or a 0.12 degree buffer if
        `data` has a geographic CRS (e.g. degree units).
    resample_method : str, optional
        If resampling is requested (see `resample` above), use this
        resampling method when resampling from low resolution to high
        resolution pixels. Defaults to "bilinear"; valid options include
        "nearest", "cubic", "min", "max", "average" etc.
    dask_chunks : tuple of float, optional
        Can be used to configure custom Dask chunking for the final
        resampling step. By default, chunks will be automatically set
        to match y/x chunks from `data` if they exist; otherwise chunks
        will be chosen to cover the entire y/x extent of the dataset.
        For custom chunks, provide a tuple in the form `(y, x)`, e.g.
        `(2048, 2048)`.
    dask_compute : bool, optional
        Whether to compute results of the resampling step using Dask.
        If False, `tides_highres` will be returned as a Dask-enabled array.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

    Returns
    -------
    tides_da : xr.DataArray
        A three-dimensional tide height array.
        If `resample=True` (default), a high-resolution array of tide
        heights will be returned that matches the exact spatial resolution
        and extents of `data`. This will contain either tide heights for
        every timestep in `data` (or in `times` if provided), or tide height
        quantiles for every quantile provided by `calculate_quantiles`.
        If `resample=False`, results for the intermediate low-resolution
        tide modelling grid will be returned instead.

    """
    # Standardise data inputs, time and models
    gbox, time_coords = _standardise_inputs(data, time)
    dask_chunks = _resample_chunks(data, dask_chunks)
    model = [model] if isinstance(model, str) else model

    # Determine spatial dimensions
    y_dim, x_dim = gbox.dimensions

    # Determine resolution and buffer, using different defaults for
    # geographic (i.e. degrees) and projected (i.e. metres) CRSs:
    assert gbox.crs is not None  # noqa: S101
    crs_units = gbox.crs.units[0][0:6]
    if gbox.crs.geographic:
        if resolution is None:
            resolution = 0.05
        elif resolution > 360:
            err_msg = (
                f"A resolution of greater than 360 was "
                f"provided, but `data` has a geographic CRS "
                f"in {crs_units} units. Did you accidentally "
                f"provide a resolution in projected "
                f"(i.e. metre) units?",
            )
            raise ValueError(err_msg)
        if buffer is None:
            buffer = 0.12
    else:
        if resolution is None:
            resolution = 5000
        elif resolution < 1:
            err_msg = (
                f"A resolution of less than 1 was provided, "
                f"but `data` has a projected CRS in "
                f"{crs_units} units. Did you accidentally "
                f"provide a resolution in geographic "
                f"(degree) units?",
            )
            raise ValueError(err_msg)
        if buffer is None:
            buffer = 12000

    # Raise error if resolution is less than dataset resolution
    dataset_res = gbox.resolution.x
    if resolution < dataset_res:
        err_msg = (
            f"The resolution of the low-resolution tide "
            f"modelling grid ({resolution:.2f}) is less "
            f"than `data`'s pixel resolution ({dataset_res:.2f}). "
            f"This can cause extremely slow tide modelling "
            f"performance. Please select provide a resolution "
            f"greater than {dataset_res:.2f} using "
            f"`pixel_tides`'s 'resolution' parameter.",
        )
        raise ValueError(err_msg)

    # Create a new reduced resolution tide modelling grid after
    # first buffering the grid
    print(f"Creating reduced resolution {resolution} x {resolution} {crs_units} tide modelling array")
    buffered_geobox = gbox.buffered(buffer)
    rescaled_geobox = GeoBox.from_bbox(bbox=buffered_geobox.boundingbox, resolution=resolution)
    rescaled_ds = odc.geo.xr.xr_zeros(rescaled_geobox)

    # Flatten grid to 1D, then add time dimension
    flattened_ds = rescaled_ds.stack(z=(x_dim, y_dim))
    flattened_ds = flattened_ds.expand_dims(dim={"time": time_coords})

    # Model tides in parallel, returning a pandas.DataFrame
    tide_df = model_tides(
        x=flattened_ds[x_dim],
        y=flattened_ds[y_dim],
        time=flattened_ds.time,
        crs=f"EPSG:{gbox.crs.epsg}",
        model=model,
        directory=directory,
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
        .tide_height
        # Re-index and transpose into our input coordinates and dim order
        .reindex_like(rescaled_ds)
        .transpose("tide_model", "time", y_dim, x_dim)
    )

    # Optionally calculate and return quantiles rather than raw data.
    # Set dtype to dtype of the input data as quantile always returns
    # float64 (memory intensive)
    if calculate_quantiles is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Computing tide quantiles")
            tides_lowres = tides_lowres.quantile(q=calculate_quantiles, dim="time").astype(tides_lowres.dtype)

    # If only one tidal model exists, squeeze out "tide_model" dim
    if len(tides_lowres.tide_model) == 1:
        tides_lowres = tides_lowres.squeeze("tide_model")

    # Ensure CRS is present before we apply any resampling
    tides_lowres = tides_lowres.odc.assign_crs(gbox.crs)

    # Reproject into original high resolution grid
    if resample:
        print("Reprojecting tides into original resolution")
        return _pixel_tides_resample(
            tides_lowres,
            gbox,
            resample_method,
            dask_chunks,
            dask_compute,
        )

    print("Returning low resolution tide array")
    return tides_lowres
