"""Validation tools for comparing modelled tides to observed tide gauge data.

This module provides functions for loading, filtering, and analysing
observed tide gauge data to validate modelled tide heights.
"""

import datetime
import os
import warnings
from math import sqrt
from numbers import Number
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import tqdm
import xarray as xr
from odc.geo.geom import BoundingBox, point
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd
from scipy import stats
from shapely.geometry import Point
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .eo import tag_tides
from .stac import load_ndwi_mpc


def eval_metrics(x, y, round=3, all_regress=False):  # noqa: A002
    """Calculate common statistical validation metrics.

    These include:

    * Pearson correlation
    * Root Mean Squared Error
    * Mean Absolute Error
    * R-squared
    * Bias
    * Linear regression parameters (slope, p-value, intercept, standard error)

    Parameters
    ----------
    x : numpy.array
        An array providing "actual" variable values.
    y : numpy.array
        An array providing "predicted" variable values.
    round : int
        Number of decimal places to round each metric
        to. Defaults to 3.
    all_regress : bool
        Whether to return linear regression p-value,
        intercept and standard error (in addition to
        only regression slope). Defaults to False.

    Returns
    -------
    pandas.Series
        A `pd.Series` containing all calculated metrics.

    """
    # Create dataframe to drop na
    xy_df = pd.DataFrame({"x": x, "y": y}).dropna()

    # Compute linear regression
    lin_reg = stats.linregress(x=xy_df.x, y=xy_df.y)

    # Calculate statistics
    stats_dict = {
        "Correlation": xy_df.corr().iloc[0, 1],
        "RMSE": sqrt(mean_squared_error(xy_df.x, xy_df.y)),
        "MAE": mean_absolute_error(xy_df.x, xy_df.y),
        "R-squared": lin_reg.rvalue**2,
        "Bias": (xy_df.y - xy_df.x).mean(),
        "Regression slope": lin_reg.slope,
    }

    # Additional regression params
    if all_regress:
        stats_dict.update({
            "Regression p-value": lin_reg.pvalue,
            "Regression intercept": lin_reg.intercept,
            "Regression standard error": lin_reg.stderr,
        })

    # Return as
    return pd.Series(stats_dict).round(round)


def _round_date_strings(date, round_type="end"):
    """Round a date string up or down to the start or end of a time period.

    Parameters
    ----------
    date : str
        Date string of variable precision (e.g. "2020", "2020-01",
        "2020-01-01").
    round_type : str, optional
        Type of rounding to perform. Valid options are "start" or "end".
        If "start", date is rounded down to the start of the time period.
        If "end", date is rounded up to the end of the time period.
        Default is "end".

    Returns
    -------
    date_rounded : str
        The rounded date string.

    Examples
    --------
    >>> round_date_strings("2020")
    '2020-12-31 00:00:00'

    >>> round_date_strings("2020-01", round_type="start")
    '2020-01-01 00:00:00'

    >>> round_date_strings("2020-01", round_type="end")
    '2020-01-31 00:00:00'

    """
    # Determine precision of input date string
    date_segments = len(date.split("-"))

    # If provided date has no "-", treat it as having year precision
    if date_segments == 1 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + YearBegin(0))
    elif date_segments == 1 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + YearEnd(0))

    # If provided date has one "-", treat it as having month precision
    elif date_segments == 2 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + MonthBegin(0))
    elif date_segments == 2 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + MonthEnd(0))

    # If more than one "-", then return date as-is
    elif date_segments > 2:
        date_rounded = date

    return date_rounded


def _load_gauge_metadata(metadata_path):
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.columns = (
        metadata_df.columns.str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.lower()
    )
    metadata_df = metadata_df.set_index("site_code")

    # Convert metadata to GeoDataFrame
    metadata_gdf = gpd.GeoDataFrame(
        data=metadata_df,
        geometry=gpd.points_from_xy(metadata_df.longitude, metadata_df.latitude),
        crs="EPSG:4326",
    )

    return metadata_df, metadata_gdf


def _load_gesla_dataset(site, path, na_value):
    # Read dataset
    gesla_df = pd.read_csv(
        path,
        skiprows=41,
        names=["date", "time", "sea_level", "qc_flag", "use_flag"],
        sep=r"\s+",
        na_values=na_value,
    )

    # Combine two date fields
    return (
        gesla_df.assign(
            time=pd.to_datetime(gesla_df["date"] + " " + gesla_df["time"]),
            site_code=site,
        )
        .drop(columns=["date"])
        .set_index("time")
    )


def _nearest_row(gdf, x, y, max_distance=None):
    # Create a point to find the nearest neighbor for
    target_point = gpd.GeoDataFrame({"geometry": [Point(x, y)]}, crs="EPSG:4326")

    # Use sjoin_nearest to find the closest point
    return gpd.sjoin_nearest(target_point, gdf, how="left", max_distance=max_distance)


def load_gauge_gesla(
    x=None,
    y=None,
    site_code=None,
    time=None,
    max_distance=None,
    correct_mean=False,
    filter_use_flag=True,
    site_metadata=True,
    data_path="GESLA4_ALL",
    metadata_path="GESLA4_ALL.csv",
):
    """Load Global Extreme Sea Level Analysis (GESLA) tide gauge data.

    Load and process all available GESLA measured sea-level data
    with an `x, y, time` spatio-temporal query, or from a list of
    specific tide gauges. Can optionally filter by gauge quality
    and append detailed gauge metadata.

    Modified from original code in <https://github.com/philiprt/GeslaDataset>.

    Parameters
    ----------
    x, y : numeric or list/tuple, optional
        Coordinates (in degrees longitude, latitude) used to load GESLA
        tide gauge observations. If provided as singular values
        (e.g. `x=150, y=-32`), then the nearest tide gauge will be returned.
        If provided as a list or tuple (e.g. `x=(150, 152), y=(-32, -30)`),
        then all gauges within the provided bounding box will be loaded.
        Leave as `None` to return all available gauges, or if providing a
        list of site codes using `site_code`.
    site_code : str or list of str, optional
        GESLA site code(s) for which to load data (e.g. `site_code="62650"`).
        If `site_code` is provided, `x` and `y` will be ignored.
    time : tuple or list of str, optional
        Time range to consider, given as a tuple of start and end dates,
        e.g. `time=("2020", "2021")`. The default of None will return all
        tide observations from the year 1800 onward.
    max_distance : numeric, optional
        Optional max distance within which to return the nearest tide gauge
        when `x` and `y` are provided as singular coordinates. Defaults to
        None, which will always return a tide gauge no matter how far away
        it is located from `x` and `y`.
    correct_mean : bool, optional
        Whether to correct sea level measurements to a standardised mean
        sea level by subtracting the mean of all observed sea level
        observations. This can be useful when GESLA tide heights come
        from different or unknown tide datums. Note: the observed mean
        sea level calculated here may differ from true long-term/
        astronomical Mean Sea Level (MSL) datum.
    filter_use_flag : bool, optional
        Whether to filter out low quality observations with a "use_flag"
        value of 0 (do not use). Defaults to True.
    site_metadata : bool, optional
        Whether to add tide gauge station metadata as additional columns
        in the output DataFrame. Defaults to True.
    data_path : str, optional
        Path to the raw GESLA data files ("GESLA-4 DATA", accessible via:
        https://gesla787883612.wordpress.com/downloads/). Defaults to
        "GESLA4_ALL".
    metadata_path : str, optional
        Path to the GESLA station metadata file ("GESLA-4 CSV META-DATA FILE",
        accessible via: https://gesla787883612.wordpress.com/downloads/).
        Defaults to "GESLA4_ALL.csv".

    Returns
    -------
    pd.DataFrame
        Processed GESLA data as a DataFrame with columns including:

        - "time": Timestamps,
        - "sea_level": Observed sea level (m),
        - "qc_flag": Observed sea level QC flag,
        - "use_flag": Use-in-analysis flag (1 = use, 0 = do not use),

        ...and additional columns from station metadata.

    """
    # Expand and validate data and metadata paths
    data_path = Path(data_path).expanduser()
    metadata_path = Path(metadata_path).expanduser()

    if not data_path.exists():
        err_msg = (
            f"GESLA raw data directory not found at `data_path={data_path}`.\n"
            "Download 'GESLA-4 DATA' from: "
            "https://gesla787883612.wordpress.com/downloads/"
        )
        raise FileNotFoundError(err_msg)

    if not metadata_path.exists():
        err_msg = (
            f"GESLA station metadata file not found at: `metadata_path={metadata_path}`.\n"
            "Download the 'GESLA-4 CSV META-DATA FILE' from: "
            "https://gesla787883612.wordpress.com/downloads/"
        )
        raise FileNotFoundError(err_msg)

    # Load tide gauge metadata
    metadata_df, metadata_gdf = _load_gauge_metadata(metadata_path)

    # Use supplied site codes if available
    if site_code is not None:
        site_code = [site_code] if not isinstance(site_code, list) else site_code

    # If x and y are tuples, use xy bounds to identify sites
    elif isinstance(x, tuple | list) & isinstance(y, tuple | list):
        bbox = BoundingBox.from_xy(x, y)
        site_code = metadata_gdf.cx[bbox.left : bbox.right, bbox.top : bbox.bottom].index

    # If x and y are single numbers, select nearest row
    elif isinstance(x, Number) & isinstance(y, Number):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            site_code = (
                _nearest_row(metadata_gdf, x, y, max_distance).rename({"index_right": "site_code"}, axis=1).site_code
            )

        # Raise exception if no valid tide gauges are found
        if site_code.isna().all():
            err_msg = f"No tide gauge found within {max_distance} degrees of {x}, {y}."
            raise Exception(err_msg)

    # Otherwise if all are None, return all available site codes
    elif (site_code is None) & (x is None) & (y is None):
        site_code = metadata_df.index.to_list()

    else:
        err_msg = (
            "`x` and `y` must be provided as either singular coordinates (e.g. `x=150`), or as a tuple bounding box (e.g. `x=(150, 152)`).",
        )
        raise Exception(err_msg)

    # Prepare times
    if time is None:
        time = ["1800", str(datetime.datetime.now().year)]
    time = [time] if not isinstance(time, list | tuple) else time
    start_time = _round_date_strings(time[0], round_type="start")
    end_time = _round_date_strings(time[-1], round_type="end")

    # Identify paths to load and nodata values for each site
    metadata_df["file_name"] = data_path / metadata_df["file_name"]
    paths_na = metadata_df.loc[site_code, ["file_name", "null_value"]]

    # Load and combine into a single dataframe
    gauge_list = [
        _load_gesla_dataset(s, p, na_value=na)
        for s, p, na in tqdm.tqdm(paths_na.itertuples(), total=len(paths_na), desc="Loading GESLA gauges")
    ]
    data_df = pd.concat(gauge_list).sort_index().loc[slice(start_time, end_time)].reset_index().set_index("site_code")

    # Optionally filter by use flag column
    if filter_use_flag:
        data_df = data_df.loc[data_df.use_flag == 1]

    # Optionally insert metadata into dataframe
    if site_metadata:
        if data_df.empty:
            data_df = data_df.reindex(columns=[*data_df.columns, *metadata_df.columns])
        else:
            data_df[metadata_df.columns] = metadata_df.loc[site_code]

    # Add time to index and remove duplicates
    data_df = data_df.set_index("time", append=True)
    duplicates = data_df.index.duplicated()
    if duplicates.sum() > 0:
        warnings.warn("Duplicate timestamps were removed.", UserWarning, stacklevel=2)
        data_df = data_df.loc[~duplicates]

    # Remove observed mean sea level if requested
    if correct_mean:
        data_df["sea_level"] = data_df["sea_level"].sub(data_df.groupby("site_code")["sea_level"].transform("mean"))

    # If no rows are returned, raise a warning
    if data_df.empty:
        warnings.warn(
            f"No data found for site '{site_code}'. "
            "Are you trying to load data using `time` for a period that does not have tide gauge measurements?",
            UserWarning,
            stacklevel=2,
        )

    # Return data
    return data_df


def tide_correlation(
    x: float | None = None,
    y: float | None = None,
    time: tuple[str, str] = ("2022", "2024"),
    crs: str = "EPSG:4326",
    data: xr.DataArray | None = None,
    model: str | list[str] = "all",
    directory: str | os.PathLike | None = None,
    index_threshold: float = 0.0,
    freq_min: float = 0.01,
    freq_max: float = 0.99,
    corr_min: float = 0.15,
    buffer: float = 2500,
    cloud_cover: float = 90,
    load_ls: bool = True,
    load_s2: bool = True,
    **tag_tides_kwargs,
):
    """Ranks tide models based on their correlation with satellite-observed inundation patterns.

    Correlations are calculated between satellite-derived water index
    (e.g. Normalised Difference Water Index, NDWI) and tide heights
    across a buffered region around an input point. High correlations
    indicate that a tide model correctly sorted satellite imagery by
    tide height, with high tide observations being consistently wet,
    and low tide observations being consistently dry.

    By default Microsoft Planetary Computer is used for loading data;
    for advance use, a pre-loaded water index xarray.DataArray from any
    satellite data source can be provided via `data`.

    Parameters
    ----------
    x, y : float, optional
        X and Y coordinates of a coastal point of interest. Assumed
        to be "EPSG:4326" degrees lat/lon; use "crs" for custom CRSs.
    time : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2022", "2024")`. We recommend using a long enough time
        period (e.g. 3+ years) to ensure that results are not affected
        by tide aliasing; see `eo_tides.stats.tide_aliasing` for more
        information.
    crs : str, optional
        Input coordinate reference system for x and y coordinates.
        Defaults to "EPSG:4326" (WGS84; degrees latitude, longitude).
    data : xr.DataArray, optional
        For advanced use, an xarray.DataArray of water index (e.g. NDWI)
        data can be supplied. If so, all data loading parameters (`x`,
        `y`, `time`, `crs`, `buffer` `cloud_cover`, `load_ls`, `load_s2`)
        will be ignored.
    model : str or list of str, optional
        The tide model (or list of models) to compare. Defaults to
        "all", which will compare all models available in `directory`.
        For a full list of available and supported models,
        run `from eo_tides.utils import list_models; list_models()`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    index_threshold : float, optional
        Water index (e.g. NDWI) threshold above which a pixel is
        considered wet.
    freq_min : float, optional
        Minimum fraction of time a pixel must be wet to be considered
        intertidal and included in the analysis.
    freq_max : float, optional
        Maximum fraction of time a pixel can be wet to be considered
        intertidal and included in the analysis.
    corr_min : float, optional
        Minimum correlation between water index (e.g. NDWI) and tide
        heights to be considered intertidal and included in the analysis.
        To ensure a like-for-like comparison, model rankings are based on
        the average correlation across every pixel with a positive
        correlation of at least `corr_min` in any individual input model.
    buffer : float, optional
        Radius in meters for generating circular buffer around the
        input point.
    cloud_cover : int, optional
        The maximum threshold of cloud cover to load. Defaults to 90%.
    load_ls : bool, optional
        Whether to load Landsat water index (e.g. NDWI) data.
    load_s2 : bool, optional
        Whether to load Sentinel-2 water index (e.g. NDWI) data.
    **tag_tides_kwargs :
        Optional parameters passed to the `eo_tides.eo.tag_tides`
        function used to model tides.

    Returns
    -------
    corr_df : pandas.DataFrame
        DataFrame with correlation and ranking per tide model.
        Columns include:

        - 'correlation': mean correlation between water index
        (e.g. NDWI and tide heights each model
        - 'rank': model rank based on correlation (1 = highest)
    corr_da : xr.DataArray
        Per-pixel correlations for each model, restricted to likely
        intertidal pixels, with dynamic wetness frequency (e.g. not
        always dry or wet), and with a positive correlation with tide
        heights from at least one tide model.

    Examples
    --------
    >>> from eo_tides import tide_correlation
    >>> y, x = -16.99636, 123.61017
    >>> corr_df, corr_da = tide_correlation(x=x, y=y, directory="tide_models/", cloud_cover=10)

    """
    # Use custom xarray.DataArray if provided
    if data is not None:
        # Verify is xr.DataArray
        if not isinstance(data, xr.DataArray):
            err_msg = "Must provide an xarray.DataArray to `data`."
            raise Exception(err_msg)

        water_index = data
        x, y = data.odc.geobox.geographic_extent.centroid.coords[0]

    # Otherwise load data for point using MPC
    elif (x is not None) and (y is not None):
        # Create circular study area around point
        geom = point(x=x, y=y, crs=crs).to_crs("utm").buffer(buffer).to_crs("EPSG:4326")

        # Load time series water_index (e.g. NDWI) for selected time period and location
        water_index = load_ndwi_mpc(
            time=time,
            geopolygon=geom,
            mask_geopolygon=True,
            cloud_cover=cloud_cover,
            load_ls=load_ls,
            load_s2=load_s2,
        ).ndwi

    # Raise error if no valid inputs are provided
    else:
        err_msg = "Must provide both `x` and `y`, or `data`."
        raise Exception(err_msg)

    # Threshold water_index to identify wet pixels, then calculate
    # overall wetness frequency (making sure NaN pixels are
    # correctly masked to ensure correct statistics)
    wet = (water_index > index_threshold).where(water_index.notnull())
    freq = wet.mean(dim="time")

    # Model tides using selected models (all available by default).
    # Use cast to tell mypy this will always be an xr.DataArray.
    tides_da = cast("xr.DataArray", tag_tides(water_index, model=model, directory=directory, **tag_tides_kwargs))

    # Calculate correlation between wetness and each tide model
    corr = xr.corr(wet, tides_da, dim="time")

    # Restrict data to likely intertidal pixels, with dynamic
    # wetness frequency (e.g. not always dry or wet), and
    # with a positive correlation with tide heights from at
    # least one tide model
    corr_max = corr.max(dim="tide_model")
    corr_da = corr.where((freq >= freq_min) & (freq <= freq_max) & (corr_max >= corr_min))
    corr_da.load()

    # Calculate mean correlation per model and valid data coverage
    corr_mean = corr_da.mean(dim=["x", "y"])
    valid_perc = corr_da.notnull().mean().item()

    # Create DataFrame with correlation and rank
    corr_df = (
        # Convert to dataframe
        corr_mean.to_dataframe(name="correlation")
        .drop("spatial_ref", axis=1)
        # Add rankings
        .assign(rank=lambda df: df.correlation.rank(ascending=False))
        .astype("float32")
        # Add metadata
        .assign(x=x, y=y, valid_perc=valid_perc)
    )

    return corr_df, corr_da
