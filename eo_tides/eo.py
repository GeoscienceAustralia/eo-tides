# Used to postpone evaluation of type annotations
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import odc.geo.xr
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

# Only import if running type checking
if TYPE_CHECKING:
    import numpy as np

from .model import model_tides


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
    ).rename("tide_height")

    # Optionally process and load into memory with Dask
    if dask_compute:
        tides_highres.load()

    return tides_highres


def tag_tides(
    ds: xr.Dataset,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    tidepost_lat: float | None = None,
    tidepost_lon: float | None = None,
    ebb_flow: bool = False,
    swap_dims: bool = False,
    **model_tides_kwargs,
) -> xr.Dataset:
    """
    Model tide heights for every timestep in a multi-dimensional
    dataset, and add them as a new `tide_height` (and optionally,
    `ebb_flow`) variable that "tags" each observation with tide data.

    The function models tides at the centroid of the dataset
    by default, but a custom tidal modelling location can
    be specified using `tidepost_lat` and `tidepost_lon`.

    This function uses the parallelised `model_tides` function
    under the hood. It supports all tidal models supported by
    `pyTMD`, including:

    - Empirical Ocean Tide model (EOT20)
    - Finite Element Solution tide models (FES2022, FES2014, FES2012)
    - TOPEX/POSEIDON global tide models (TPXO10, TPXO9, TPXO8)
    - Global Ocean Tide models (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
    - Hamburg direct data Assimilation Methods for Tides models (HAMTIDE11)

    Parameters
    ----------
    ds : xarray.Dataset
        A multi-dimensional dataset (e.g. "x", "y", "time") to
        tag with tide heights. This dataset must contain a "time"
        dimension.
    model : str or list of str, optional
        The tide model (or models) to use to model tides. If a list is
        provided, a new "tide_model" dimension will be added to `ds`.
        Defaults to "EOT20"; for a full list of available/supported
        models, run `eo_tides.model.list_models`.
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
    ebb_flow : bool, optional
        An optional boolean indicating whether to compute if the
        tide phase was ebbing (falling) or flowing (rising) for each
        observation. The default is False; if set to True, a new
        "ebb_flow" variable will be added to the dataset with each
        observation labelled with "Ebb" or "Flow".
    swap_dims : bool, optional
        An optional boolean indicating whether to swap the `time`
        dimension in the original `ds` to the new "tide_height"
        variable. Defaults to False.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

    Returns
    -------
    ds : xr.Dataset
        The original `xarray.Dataset` with a new `tide_height` variable
        giving the height of the tide (and optionally, its ebb-flow phase)
        for each timestep in the data.

    """
    # Only datasets are supported
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset, not an xarray.DataArray or other data type.")

    # Standardise model into a list for easy handling. and verify only one
    model = [model] if isinstance(model, str) else model
    if (len(model) > 1) & swap_dims:
        raise ValueError("Can only swap dimensions when a single tide model is passed to `model`.")

    # If custom tide modelling locations are not provided, use the
    # dataset centroid
    if tidepost_lat is None or tidepost_lon is None:
        lon, lat = ds.odc.geobox.geographic_extent.centroid.coords[0]
        print(f"Setting tide modelling location from dataset centroid: {lon:.2f}, {lat:.2f}")
    else:
        lon, lat = tidepost_lon, tidepost_lat
        print(f"Using tide modelling location: {lon:.2f}, {lat:.2f}")

    # Model tide heights for each observation:
    tide_df = model_tides(
        x=lon,  # type: ignore
        y=lat,  # type: ignore
        time=ds.time,
        model=model,
        directory=directory,
        crs="EPSG:4326",
        **model_tides_kwargs,
    )

    # If tides cannot be successfully modeled (e.g. if the centre of the
    # xarray dataset is located is over land), raise an exception
    if tide_df.tide_height.isnull().all():
        raise ValueError(
            f"Tides could not be modelled for dataset centroid located "
            f"at {tidepost_lon:.2f}, {tidepost_lat:.2f}. This can occur if "
            f"this coordinate occurs over land. Please manually specify "
            f"a tide modelling location located over water using the "
            f"`tidepost_lat` and `tidepost_lon` parameters."
        )

    # Optionally calculate the tide phase for each observation
    if ebb_flow:
        # Model tides for a time 15 minutes prior to each previously
        # modelled satellite acquisition time. This allows us to compare
        # tide heights to see if they are rising or falling.
        print("Modelling tidal phase (e.g. ebb or flow)")
        tide_pre_df = model_tides(
            x=lon,  # type: ignore
            y=lat,  # type: ignore
            time=(ds.time - pd.Timedelta("15 min")),
            model=model,
            directory=directory,
            crs="EPSG:4326",
            **model_tides_kwargs,
        )

        # Compare tides computed for each timestep. If the previous tide
        # was higher than the current tide, the tide is 'ebbing'. If the
        # previous tide was lower, the tide is 'flowing'
        tide_df["ebb_flow"] = (tide_df.tide_height < tide_pre_df.tide_height.values).replace({
            True: "Ebb",
            False: "Flow",
        })

    # Convert to xarray format
    tide_xr = tide_df.reset_index().set_index(["time", "tide_model"]).drop(["x", "y"], axis=1).to_xarray()

    # If only one tidal model exists, squeeze out "tide_model" dim
    if len(tide_xr.tide_model) == 1:
        tide_xr = tide_xr.squeeze("tide_model", drop=True)

    # Add each array into original dataset
    for var in tide_xr.data_vars:
        ds[var] = tide_xr[var]

    # Swap dimensions and sort by tide height
    if swap_dims:
        ds = ds.swap_dims({"time": "tide_height"})
        ds = ds.sortby("tide_height")
        ds = ds.drop_vars("time")

    return ds


def pixel_tides(
    ds: xr.Dataset | xr.DataArray,
    times=None,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    resample: bool = True,
    calculate_quantiles: np.ndarray | tuple[float, float] | None = None,
    resolution: float | None = None,
    buffer: float | None = None,
    resample_method: str = "bilinear",
    dask_chunks: str | tuple[float, float] = "auto",
    dask_compute: bool = True,
    **model_tides_kwargs,
) -> xr.DataArray:
    """
    Model tide heights for every pixel in a multi-dimensional
    dataset, using one or more ocean tide models.

    This function models tides into a low-resolution tide
    modelling grid covering the spatial extent of the input
    data (buffered to reduce potential edge effects). These
    modelled tides are then (optionally) resampled back into
    the original higher resolution dataset's extent and
    resolution - resulting in a modelled tide height for every
    pixel through time.

    This function uses the parallelised `model_tides` function
    under the hood. It supports all tidal models supported by
    `pyTMD`, including:

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

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        A multi-dimensional dataset (e.g. "x", "y", "time") that will
        be used to define the tide modelling grid.
    times : pd.DatetimeIndex or list of pd.Timestamp, optional
        By default, the function will model tides using the times
        contained in the `time` dimension of `ds`. Alternatively, this
        param can be used to model tides for a custom set of times
        instead. For example:
        `times=pd.date_range(start="2000", end="2001", freq="5h")`
    model : str or list of str, optional
        The tide model (or models) used to model tides. If a list is
        provided, a new "tide_model" dimension will be added to the
        `xarray.DataArray` outputs. Defaults to "EOT20"; for a full
        list of available/supported models, run `eo_tides.model.list_models`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    resample : bool, optional
        Whether to resample low resolution tides back into `ds`'s original
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
        if `ds` has a projected CRS (i.e. metre units), or a 0.05 degree
        resolution grid if `ds` has a geographic CRS (e.g. degree units).
        Note: higher resolutions do not necessarily provide better
        tide modelling performance, as results will be limited by the
        resolution of the underlying global tide model (e.g. 1/16th
        degree / ~5 km resolution grid for FES2014).
    buffer : float, optional
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
    resample_method : str, optional
        If resampling is requested (see `resample` above), use this
        resampling method when converting from low resolution to high
        resolution pixels. Defaults to "bilinear"; valid options include
        "nearest", "cubic", "min", "max", "average" etc.
    dask_chunks : str or tuple of float, optional
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
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.
    Returns
    -------
    tides_da : xr.DataArray
        If `resample=True` (default), a high-resolution array
        of tide heights matching the exact spatial resolution and
        extents of `ds`. This will contain either tide heights every
        timestep in `ds` (if `times` is None), tide heights at every
        time in `times` (if `times` is not None), or tide height
        quantiles for every quantile provided by `calculate_quantiles`.
        If `resample=False`, results for the intermediate low-resolution
        tide modelling grid will be returned instead.
    """
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

    # Standardise model into a list for easy handling
    model = [model] if isinstance(model, str) else model

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
        print("Computing tide quantiles")
        tides_lowres = tides_lowres.quantile(q=calculate_quantiles, dim="time").astype(tides_lowres.dtype)

    # If only one tidal model exists, squeeze out "tide_model" dim
    if len(tides_lowres.tide_model) == 1:
        tides_lowres = tides_lowres.squeeze("tide_model")

    # Ensure CRS is present before we apply any resampling
    tides_lowres = tides_lowres.odc.assign_crs(ds.odc.geobox.crs)

    # Reproject into original high resolution grid
    if resample:
        print("Reprojecting tides into original resolution")
        tides_highres = _pixel_tides_resample(
            tides_lowres,
            ds,
            resample_method,
            dask_chunks,
            dask_compute,
        )
        return tides_highres

    print("Returning low resolution tide array")
    return tides_lowres
