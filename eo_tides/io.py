# Used to postpone evaluation of type annotations
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import xarray as xr
from odc.geo.geom import BoundingBox

# Only import if running type checking
if TYPE_CHECKING:
    from odc.geo.geom import Geometry
    from pystac import ItemCollection
    from xarray import Dataset


def stac_load(
    product: str,
    time_range: tuple[str, str] | None = None,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    geom: Geometry | None = None,
    stac_query: dict | None = None,
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    **load_params,
) -> tuple[Dataset, ItemCollection]:
    """
    Convenience function for querying satellite data from a
    SpatioTemporal Asset Catalogue (STAC) using `pystac_client`,
    then loading data as a multi-temporal `xarray.Dataset` using
    `odc-stac`.

    Defaults to using the Microsoft Planetary Computer STAC API.

    Parameters
    ----------
    product : str
        The name of the product (i.e. STAC "collection") to load.
    time_range : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2020", "2021")`. If not provided, data will be loaded for
        all available timesteps.
    x, y : tuple, optional
        Tuples defining the x and y bounding box to load, in WGS 84.
    geom : datacube Geometry, optional
        An `odc.geo.geom.Geometry` geometry object representing the
        spatial extents to load data for. If provided, `x` and `y`
        will be ignored.
    stac_query : dict, optional
        A query dictionary to further filter the data using STAC metadata.
        If not provided, no additional filtering will be applied. For
        example: `stac_query = {"eo:cloud_cover": {"lt": 10}}`.
    url : str, optional
        The URL of the STAC API endpoint to query and load data from.
        Defaults to "https://planetarycomputer.microsoft.com/api/stac/v1".
    **load_params : dict
        Additional parameters to be passed to `odc.stac.load()`.

    Returns
    -------
    ds : xarray.Dataset
        The loaded dataset as an `xarray.Dataset`.
    items : pystac.item_collection.ItemCollection
        STAC items returned by `pystac_client`.
    """

    # Connect to client
    catalog = pystac_client.Client.open(
        url,
        modifier=(planetary_computer.sign_inplace if "planetarycomputer" in url else None),
    )

    # Set up time for query
    time_range = "/".join(time_range) if time_range is not None else None

    # Set up bounding box for query
    if geom is not None:
        bbox = geom.boundingbox
    elif (x is not None) and (y is not None):
        bbox = BoundingBox.from_xy(x, y)
    else:
        raise Exception("Must provide either `x` and `y`, or `geom`")

    # Ensure longitude is between -180 to 180:
    if (bbox.left >= 180) or (bbox.right >= 180):
        bbox = BoundingBox(
            left=bbox.left - 360,
            bottom=bbox.bottom,
            right=bbox.right - 360,
            top=bbox.top,
        )

    # Find matching items
    search = catalog.search(
        collections=product,
        bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
        datetime=time_range,
        query=stac_query if stac_query is not None else None,
    )

    # Check how many items were returned
    items = search.item_collection()
    print(f"Found {len(items)} STAC items for {product}")

    # Load with ODC STAC
    ds = odc.stac.load(
        items=items,
        bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
        **load_params,
    )

    return ds, items


def load_ndwi_mpc(
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    geom: Geometry | None = None,
    time_range: tuple[str, str] = ("2022", "2024"),
    crs: str = "utm",
    resolution: int = 30,
    resampling: str = "cubic",
    cloud_cover: int = 90,
    load_ls: bool = True,
    load_s2: bool = True,
    chunks: dict[str, int] = {"x": 2048, "y": 2048},
    fail_on_error: bool = False,
) -> Dataset:
    """
    Load an NDWI time-series of Landsat ("landsat-c2-l2") and/or
    Sentinel-2 ("sentinel-2-l2a") satellite data hosted on Microsoft
    Planetary Computer, using a set of simple opinionated loading and
    cloud masking parameters.

    Note: These parameters may not be optimal for all use cases; we
    recommend reviewing and/or modifying them prior to formal analysis.

    Parameters
    ----------
    x, y : tuple, optional
        Tuples defining the x and y bounding box to load, in WGS 84.
    geom : datacube Geometry, optional
        An optional datacube geometry object representing the spatial extents to
        load data for. If provided, `x` and `y` will be ignored.
    time_range : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2020", "2021")`. If not provided, data will be loaded for
        a three year epoch from 2022 to 2024.
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
    cloud cover : int, optional
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
    query_params = dict(
        time_range=time_range,
        geom=geom if geom is not None else None,
        x=x if geom is None else None,
        y=y if geom is None else None,
    )

    # Assemble parameters used for loading data into xarray format
    load_params = dict(
        crs=crs,
        resolution=resolution,
        chunks=chunks,
        groupby="solar_day",
        resampling={"qa_pixel": "nearest", "SCL": "nearest", "*": resampling},
        fail_on_error=fail_on_error,
    )

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
            **query_params,  # type: ignore
            **load_params,  # type: ignore
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
            **query_params,  # type: ignore
            **load_params,  # type: ignore
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
    satellite_ds = xr.concat(output_list, dim="time").sortby("time").to_dataset(name="ndwi")

    return satellite_ds
