import datetime
import glob
import warnings
from math import sqrt
from numbers import Number

import geopandas as gpd
import pandas as pd
from odc.geo.geom import BoundingBox
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd
from scipy import stats
from shapely.geometry import Point
from sklearn.metrics import mean_absolute_error, mean_squared_error


def eval_metrics(x, y, round=3, all_regress=False):
    """
    Calculate a set of common statistical metrics
    based on two input actual and predicted vectors.

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
    """
    Round a date string up or down to the start or end of a given time
    period.

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
    >>> round_date_strings('2020')
    '2020-12-31 00:00:00'

    >>> round_date_strings('2020-01', round_type='start')
    '2020-01-01 00:00:00'

    >>> round_date_strings('2020-01', round_type='end')
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
    gesla_df = (
        pd.read_csv(
            path,
            skiprows=41,
            names=["date", "time", "sea_level", "qc_flag", "use_flag"],
            sep=r"\s+",  # sep="\s+",
            parse_dates=[[0, 1]],
            index_col=0,
            na_values=na_value,
        )
        .rename_axis("time")
        .assign(site_code=site)
    )

    return gesla_df


def _nearest_row(gdf, x, y, max_distance=None):
    # Create a point to find the nearest neighbor for
    target_point = gpd.GeoDataFrame({"geometry": [Point(x, y)]}, crs="EPSG:4326")

    # Use sjoin_nearest to find the closest point
    return gpd.sjoin_nearest(target_point, gdf, how="left", max_distance=max_distance)


def load_gauge_gesla(
    x=None,
    y=None,
    site_code=None,
    time=("2018", "2020"),
    max_distance=None,
    correct_mean=False,
    filter_use_flag=True,
    site_metadata=True,
    data_path="/gdata1/data/sea_level/gesla/",
    metadata_path="/gdata1/data/sea_level/GESLA3_ALL 2.csv",
):
    """
    Load and process all available Global Extreme Sea Level Analysis
    (GESLA) tide gauge data with an `x, y, time` spatiotemporal query,
    or from a list of specific tide gauges.

    Can optionally filter by gauge quality and append detailed gauge metadata.

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
        Path to the raw GESLA data files. Default is
        `/gdata1/data/sea_level/gesla/`.
    metadata_path : str, optional
        Path to the GESLA station metadata file.
        Default is `/gdata1/data/sea_level/GESLA3_ALL 2.csv`.

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
    # Load tide gauge metadata
    metadata_df, metadata_gdf = _load_gauge_metadata(metadata_path)

    # Use supplied site codes if available
    if site_code is not None:
        site_code = [site_code] if not isinstance(site_code, list) else site_code

    # If x and y are tuples, use xy bounds to identify sites
    elif isinstance(x, (tuple, list)) & isinstance(y, (tuple, list)):
        bbox = BoundingBox.from_xy(x, y)
        site_code = metadata_gdf.cx[bbox.left : bbox.right, bbox.top : bbox.bottom].index

    # If x and y are single numbers, select nearest row
    elif isinstance(x, Number) & isinstance(y, Number):
        site_code = _nearest_row(metadata_gdf, x, y, max_distance).site_code

        # Raise exception if no valid tide gauges are found
        if site_code.isnull().all():
            raise Exception(f"No tide gauge found within {max_distance} degrees of {x}, {y}.")

    # Otherwise if all are None, return all available site codes
    elif (site_code is None) & (x is None) & (y is None):
        site_code = metadata_df.index.to_list()

    else:
        raise TypeError(
            "`x` and `y` must be provided as either singular coordinates (e.g. `x=150`), or as a tuple bounding box (e.g. `x=(150, 152)`)."
        )

    # Prepare times
    if time is None:
        time = ["1800", str(datetime.datetime.now().year)]
    time = [time] if not isinstance(time, (list, tuple)) else time
    start_time = _round_date_strings(time[0], round_type="start")
    end_time = _round_date_strings(time[-1], round_type="end")

    # Identify paths to load and nodata values for each site
    metadata_df["file_name"] = data_path + metadata_df["file_name"]
    paths_na = metadata_df.loc[site_code, ["file_name", "null_value"]]

    # Load and combine into a single dataframe
    data_df = (
        pd.concat([_load_gesla_dataset(s, p, na_value=na) for s, p, na in paths_na.itertuples()])
        .sort_index()
        .loc[slice(start_time, end_time)]
        .reset_index()
        .set_index("site_code")
    )

    # Optionally filter by use flag column
    if filter_use_flag:
        data_df = data_df.loc[data_df.use_flag == 1]

    # Optionally insert metadata into dataframe
    if site_metadata:
        data_df[metadata_df.columns] = metadata_df.loc[site_code]

    # Add time to index and remove duplicates
    data_df = data_df.set_index("time", append=True)
    duplicates = data_df.index.duplicated()
    if duplicates.sum() > 0:
        warnings.warn("Duplicate timestamps were removed.")
        data_df = data_df.loc[~duplicates]

    # Remove observed mean sea level if requested
    if correct_mean:
        data_df["sea_level"] = data_df["sea_level"].sub(data_df.groupby("site_code")["sea_level"].transform("mean"))

    # Return data
    return data_df
