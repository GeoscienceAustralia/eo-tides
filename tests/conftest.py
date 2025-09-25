"""This module contains shared fixtures for eo_tides tests."""

from copy import deepcopy

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import pytest

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


@pytest.fixture
def measured_tides_ds():
    """Load measured sea level data from the Broome ABSLMP tidal station:
    http://www.bom.gov.au/oceanography/projects/abslmp/data/data.shtml
    """
    # Metadata for Broome ABSLMP tidal station:
    # http://www.bom.gov.au/oceanography/projects/abslmp/data/data.shtml
    ahd_offset = -5.322

    # Load measured tides from ABSLMP tide gauge data
    measured_tides_df = pd.read_csv(
        "tests/data/IDO71013_2020.csv",
        index_col=0,
        parse_dates=True,
        na_values=-9999,
    )[["Sea Level"]]

    # Update index and column names
    measured_tides_df.index.name = "time"
    measured_tides_df.columns = ["tide_height"]

    # Apply station AHD offset
    measured_tides_df += ahd_offset

    # Return as xarray dataset
    return measured_tides_df.to_xarray()


# Create test data in different CRSs and resolutions
@pytest.fixture(
    params=[
        ("EPSG:3577", 30),  # Australian Albers 30 m pixels
        ("EPSG:4326", 0.00025),  # WGS84, 0.0025 degree pixels
    ],
    ids=["satellite_ds_epsg3577", "satellite_ds_epsg4326"],
    scope="session",  # only load data once, but copy for each test
)
def satellite_ds_load(request):
    """Load a sample timeseries of Landsat 8 data from either
    Microsoft Planetary Computer or Digital Earth Australia's
    STAC APIs using odc-stac.
    """
    # Obtain CRS and resolution params
    crs, res = request.param

    # Bounding box
    bbox = [GAUGE_X - 0.08, GAUGE_Y - 0.08, GAUGE_X + 0.08, GAUGE_Y + 0.08]

    try:
        # Connect to STAC catalog
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        # Set cloud access defaults
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={"aws_unsigned": True},
        )

        # Build a query with the parameters above
        query = catalog.search(
            bbox=bbox,
            collections=["landsat-c2-l2"],
            datetime="2020-01/2020-02",
            query={
                "platform": {"in": ["landsat-8"]},
            },
        )

        # Search the STAC catalog for all items matching the query
        ds = odc.stac.load(
            list(query.items()),
            bands=["red"],
            crs=crs,
            resolution=res,
            groupby="solar_day",
            bbox=bbox,
            fail_on_error=False,
            chunks={},
        )

        # Rename for compatibility with original DEA tests
        ds["nbart_red"] = ds.red

        return ds

    except Exception as e:
        print(f"Failed to load data from Microsoft Planetary Computer with error {e}; trying DEA")

        # Connect to stac catalogue
        catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")

        # Set cloud defaults
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={"aws_unsigned": True},
        )

        # Build a query with the parameters above
        query = catalog.search(
            bbox=bbox,
            collections=["ga_ls8c_ard_3"],
            datetime="2020-01/2020-02",
        )

        # Search the STAC catalog for all items matching the query
        return odc.stac.load(
            list(query.items()),
            bands=["nbart_red"],
            crs=crs,
            resolution=res,
            groupby="solar_day",
            bbox=bbox,
            fail_on_error=False,
            chunks={},
        )


@pytest.fixture
def satellite_ds(satellite_ds_load):
    """Make a copy of the previously loaded satellite data for
    each test to ensure each test is independent
    """
    return deepcopy(satellite_ds_load)
