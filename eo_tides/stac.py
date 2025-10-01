"""Tools for loading satellite EO data using STAC.

This module provides utilities for loading EO data via
SpatioTemporal Asset Catalog (STAC) metadata, such as
those available on platforms like Microsoft Planetary Computer.
"""

# Used to postpone evaluation of type annotations
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import xarray as xr
from odc.geo.geom import BoundingBox
from odc.stac._mdtools import _normalize_geometry

# Only import if running type checking
if TYPE_CHECKING:
    from pystac import ItemCollection
    from xarray import Dataset


def _get_bbox(bbox=None, geopolygon=None, lon=None, lat=None):
    """Obtain an EPSG:4326 bounding box for STAC querying."""
    # If provided as a bounding box, either convert to `odc-geo`
    # bounding box or return as-is
    if bbox is not None:
        bbox_extracted = BoundingBox(*bbox, crs="EPSG:4326") if isinstance(bbox, list | tuple) else bbox

    # If data is provided as a geopolygon, normalise to `odc-geo`
    # geometry and extract bounding box
    elif geopolygon is not None:
        geopolygon = _normalize_geometry(geopolygon)
        bbox_extracted = geopolygon.boundingbox

    # If provided as lon/lat ranges, convert to an `odc-geo` bounding box
    elif (lon is not None) and (lat is not None):
        bbox_extracted = BoundingBox.from_xy(lon, lat, crs="EPSG:4326")

    # Raise error if no valid inputs are provided
    else:
        err_msg = "Must provide both `lon` and `lat`, or `geopolygon`, or `bbox`."
        raise Exception(err_msg)

    # Convert bounding box to EPSG:4326 if required
    if not bbox_extracted.crs.geographic:
        bbox_extracted = bbox_extracted.to_crs("EPSG:4326")

    return bbox_extracted, geopolygon


def stac_load(
    product: str,
    bands: str | list[str] | tuple[str, ...] | None = None,
    time: tuple[str, str] | None = None,
    lon: tuple[float, float] | None = None,
    lat: tuple[float, float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geopolygon: Any | None = None,
    mask_geopolygon: bool = False,
    stac_query: dict | None = None,
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    dtype: Any | None = None,
    **load_params,
) -> tuple[Dataset, ItemCollection]:
    """Query and load satellite data from a STAC API.

    Uses `pystac_client` to query a SpatioTemporal Asset Catalogue (STAC),
    then load the results as a multi-temporal `xarray.Dataset` using
    `odc-stac`.

    Defaults to using the Microsoft Planetary Computer STAC API.

    Parameters
    ----------
    product : str
        The name of the product (i.e. STAC "collection") to load.
    bands : str or list, optional
        List of band names to load. Defaults to all, also accepts a
        single band name (e.g. "red").
    time : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2020", "2021")`. If not provided, data will be loaded for
        all available timesteps.
    lon, lat : tuple, optional
        Tuples defining the spatial x and y extent to load in degrees.
    bbox : tuple, optional
        Load data into the extent of a bounding box (left, bottom, right, top).
    geopolygon : multiple types, optional
        Load data into the extents of a geometry. This could be an
        odc.geo Geometry, a GeoJSON dictionary, Shapely geometry, GeoPandas
        DataFrame or GeoSeries. GeoJSON and Shapely inputs are assumed to
        be in EPSG:4326 coordinates.
    mask_geopolygon : bool, optional
        Whether to mask pixels as `NaN` if they are outside the extent
        of a provided geopolygon. Defaults to False; note that this
        will convert all bands to `float32` dtype, so should be used with
        caution for any integer or boolean bands (e.g. cloud masks etc).
    stac_query : dict, optional
        A query dictionary to further filter the data using STAC metadata.
        If not provided, no additional filtering will be applied. For
        example: `stac_query = {"eo:cloud_cover": {"lt": 10}}`.
    stac_url : str, optional
        The URL of the STAC API endpoint to query and load data from.
        Defaults to "https://planetarycomputer.microsoft.com/api/stac/v1".
    dtype : optional
        Data type to load data into. The default will use the dataset's
        default dtype. If `mask_geopolygon=True`, data will be returned
        in `float32` with pixels outside the mask set to `NaN`.
    **load_params : dict
        Additional parameters to be passed to `odc.stac.load()` to customise
        how data is loaded.

    Returns
    -------
    ds : xarray.Dataset
        The loaded dataset as an `xarray.Dataset`.
    items : pystac.item_collection.ItemCollection
        STAC items returned by `pystac_client`.

    """
    # Connect to client
    catalog = pystac_client.Client.open(
        stac_url,
        modifier=(planetary_computer.sign_inplace if "planetarycomputer" in stac_url else None),
    )

    # Set dtype; use provided unless `mask_geopolygon` is provided,
    # in which case use `float32`.
    dtype = "float32" if mask_geopolygon else dtype

    # Set up time for query
    time = "/".join(time) if time is not None else None

    # Extract degree lat/lon bounding box for STAC query
    bbox_4326, geopolygon = _get_bbox(bbox=bbox, geopolygon=geopolygon, lon=lon, lat=lat)

    # Find matching items
    search = catalog.search(
        collections=product,
        bbox=(bbox_4326.left, bbox_4326.bottom, bbox_4326.right, bbox_4326.top),
        datetime=time,
        query=stac_query if stac_query is not None else None,
    )

    # Check how many items were returned
    items = search.item_collection()
    print(f"Found {len(items)} STAC items for {product}")

    # Load with ODC STAC
    ds = odc.stac.load(
        items=items,
        bands=bands,
        bbox=bbox,
        geopolygon=geopolygon,
        lon=lon,
        lat=lat,
        dtype=dtype,
        **load_params,
    )

    # Optionally mask areas outside of supplied geopolygon
    if mask_geopolygon & (geopolygon is not None):
        ds = ds.odc.mask(poly=geopolygon)

    return ds, items


def load_ndwi_mpc(
    time: tuple[str, str] | None = None,
    lon: tuple[float, float] | None = None,
    lat: tuple[float, float] | None = None,
    bbox: BoundingBox | tuple | list | None = None,
    geopolygon: Any | None = None,
    mask_geopolygon: bool = False,
    crs: str = "utm",
    resolution: float = 30,
    resampling: str = "cubic",
    cloud_cover: float = 90,
    load_ls: bool = True,
    load_s2: bool = True,
    chunks: dict[str, int] = {"x": 2048, "y": 2048},
    fail_on_error: bool = False,
) -> Dataset:
    """Load an NDWI time-series from Landsat and/or Sentinel-2 from Microsoft Planetary Computer.

    Landsat and Sentinel-2 satellite data are accessed from the Microsoft
    Planetary Computer ("landsat-c2-l2" and "sentinel-2-l2a") STAC API using
    a set of opinionated loading and cloud-masking parameters.
    These parameters may not be optimal for all use cases; we recommend
    reviewing and/or modifying them prior to formal analysis.

    Parameters
    ----------
    time : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2020", "2021")`. If not provided, data will be loaded for
        all available timesteps.
    lon, lat : tuple, optional
        Tuples defining the spatial x and y extent to load in degrees.
    bbox : BoundingBox, tuple or list, optional
        Load data into the extent of a bounding box (left, bottom, right, top).
    geopolygon : multiple types, optional
        Load data into the extents of a geometry. This could be an
        odc.geo Geometry, a GeoJSON dictionary, Shapely geometry, GeoPandas
        DataFrame or GeoSeries. GeoJSON and Shapely inputs are assumed to
        be in EPSG:4326 coordinates.
    mask_geopolygon : bool, optional
        Whether to mask pixels as nodata if they are outside the extent
        of a provided geopolygon. Defaults to False.
    crs : str, optional
        The Coordinate Reference System (CRS) to load data into. Defaults
        to "utm", which will attempt to load data into its native UTM
        CRS to minimise resampling.
    resolution : int, optional
        Spatial resolution to load data in. Defaults to 30 metres.
    resampling : str, optional
        Resampling method used for surface reflectance bands. Defaults
        to "cubic"; "nearest" will always be used for categorical cloud
        masking bands.
    cloud_cover : int, optional
        The maximum threshold of cloud cover to load. Defaults to 90%.
    load_ls : bool, optional
        Whether to query and load Landsat data ("landsat-c2-l2").
    load_s2 : bool, optional
        Whether to query and load Sentinel-2 data ("sentinel-2-l2a").
    chunks : dictionary, optional
        Dask chunking used to load data as lazy Dask backed arrays.
        Defaults to `{"x": 2048, "y": 2048}`.
    fail_on_error : bool, optional
        Whether to return an error if any individual satellite datasets
        cannot be loaded. Defaults to False, which prevents ephemeral
        cloud access issues from failing the analysis.

    Returns
    -------
    satellite_ds : xarray.Dataset
        The loaded dataset as an `xarray.Dataset`, containing a single
        "ndwi" `xarray.DataArray`.

    """
    # Assemble parameters used for querying STAC API
    query_params = {
        "time": time,
        "geopolygon": geopolygon,
        "bbox": bbox,
        "lon": lon,
        "lat": lat,
    }

    # Assemble parameters used for loading data into xarray format
    load_params = {
        "crs": crs,
        "resolution": resolution,
        "chunks": chunks,
        "fail_on_error": fail_on_error,
        "groupby": "solar_day",
        "resampling": {"qa_pixel": "nearest", "SCL": "nearest", "*": resampling},
    }

    # List to hold outputs for each sensor (Landsat, Sentinel-2)
    output_list = []

    if load_ls:
        # Load Landsat
        ds_ls, items_ls = stac_load(
            product="landsat-c2-l2",
            bands=("green", "nir08", "qa_pixel"),
            stac_query={
                "eo:cloud_cover": {"lt": cloud_cover},
                "landsat:collection_category": {"in": ["T1"]},
            },
            **query_params,  # type: ignore[arg-type]
            **load_params,  # type: ignore[arg-type]
        )

        # Apply simple Landsat cloud mask
        cloud_mask = (
            # Bit 3: high confidence cloud, bit 4: high confidence shadow
            # https://medium.com/analytics-vidhya/python-for-geosciences-
            # raster-bit-masks-explained-step-by-step-8620ed27141e
            np.bitwise_and(ds_ls.qa_pixel, 1 << 3) | np.bitwise_and(ds_ls.qa_pixel, 1 << 4)
        ) == 0
        ds_ls = ds_ls.where(cloud_mask).drop_vars("qa_pixel")

        # Rescale to between 0.0 and 1.0
        ds_ls = (ds_ls.where(ds_ls != 0) * 0.0000275 + -0.2).clip(0, 1)

        # Convert to NDWI
        ndwi_ls = (ds_ls.green - ds_ls.nir08) / (ds_ls.green + ds_ls.nir08)
        output_list.append(ndwi_ls)

    if load_s2:
        # Load Sentinel-2
        ds_s2, items_s2 = stac_load(
            product="sentinel-2-l2a",
            bands=("green", "nir", "SCL"),
            stac_query={
                "eo:cloud_cover": {"lt": cloud_cover},
            },
            **query_params,  # type: ignore[arg-type]
            **load_params,  # type: ignore[arg-type]
        )

        # Apply simple Sentinel-2 cloud mask
        # 1: defective, 3: shadow, 9: high confidence cloud
        cloud_mask = ~ds_s2.SCL.isin([0, 1, 3, 9])
        ds_s2 = ds_s2.where(cloud_mask).drop_vars("SCL")

        # Sentinel-2 Processing Baseline 4.0 introduced new offset/scaling
        # after January 25 2022. We have to split our data before and after
        # this date, and apply different scaling to each
        ds_s2_pre = ds_s2.sel(time=slice(None, "2022-01-25"))
        ds_s2_post = ds_s2.sel(time=slice("2022-01-26", None))
        ds_s2_pre = (ds_s2_pre.where(ds_s2_pre != 0) * 0.0001).clip(0, 1)
        ds_s2_post = ((ds_s2_post.where(ds_s2_post != 0) - 1000) * 0.0001).clip(0, 1)

        # Combine both rescaled datasets
        ds_s2 = xr.concat([ds_s2_pre, ds_s2_post], dim="time")

        # Convert to NDWI
        ndwi_s2 = (ds_s2.green - ds_s2.nir) / (ds_s2.green + ds_s2.nir)
        output_list.append(ndwi_s2)

    # Merge into a single dataset
    ndwi = xr.concat(output_list, dim="time").sortby("time").to_dataset(name="ndwi")

    # Optionally mask areas outside of supplied geopolygon (this has to be
    # applied here because applying it at the `stac_load` level converts
    # cloud masking bands to "float32".
    if mask_geopolygon & (geopolygon is not None):
        geopolygon = _normalize_geometry(geopolygon)
        ndwi = ndwi.odc.mask(poly=geopolygon)

    return ndwi
