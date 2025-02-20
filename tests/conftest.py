"""
This module contains shared fixtures for eo_tides tests.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import pystac_client
import pytest
import xarray as xr

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


@pytest.fixture()
def measured_tides_ds():
    """
    Load measured sea level data from the Broome ABSLMP tidal station:
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
    """
    Load a sample timeseries of Landsat 8 data using odc-stac
    """
    # Obtain CRS and resolution params
    crs, res = request.param

    # Connect to stac catalogue
    catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")

    # Set cloud defaults
    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True},
    )

    # Build a query with the parameters above
    bbox = [GAUGE_X - 0.08, GAUGE_Y - 0.08, GAUGE_X + 0.08, GAUGE_Y + 0.08]
    query = catalog.search(
        bbox=bbox,
        collections=["ga_ls8c_ard_3"],
        datetime="2020-01/2020-02",
    )

    # Search the STAC catalog for all items matching the query
    ds = odc.stac.load(
        list(query.items()),
        bands=["nbart_red"],
        crs=crs,
        resolution=res,
        groupby="solar_day",
        bbox=bbox,
        fail_on_error=False,
        chunks={},
    )

    return ds


@pytest.fixture
def satellite_ds(satellite_ds_load):
    """
    Make a copy of the previously loaded satellite data for
    each test to ensure each test is independent
    """
    return deepcopy(satellite_ds_load)


# Run once per session to generate symethic HAMTIDE11 files; autouse=True
# allows this to run without being specifically called in tests
@pytest.fixture(scope="session", autouse=True)
def create_synthetic_model(base_dir="tests/data/tide_models_synthetic"):
    """
    Generates and exports synthetic HAMTIDE11 model data
    to test clipping functionality.
    """
    base_dir = Path(base_dir)  # Ensure base_dir is a Path object

    # Create coordinate arrays
    lon = np.arange(0, 360.125, 0.125)  # 2881 points
    lat = np.arange(-90, 90.125, 0.125)  # 1441 points

    # List of hamtide tidal constituents
    constituents = ["2n", "k1", "k2", "m2", "n2", "o1", "p1", "q1", "s2"]

    # Create hamtide output directory
    hamtide_dir = base_dir / "hamtide"
    hamtide_dir.mkdir(parents=True, exist_ok=True)

    # Create and save a NetCDF for each constituent
    for constituent in constituents:
        # Create synthetic hamtide dataset with random data
        shape = (len(lat), len(lon))  # 1441, 2881
        data = np.random.random(shape).astype(np.float32)
        ds = xr.Dataset(
            {
                "RE": (("LAT", "LON"), data),
                "IM": (("LAT", "LON"), data),
                "AMPL": (("LAT", "LON"), data),
                "PHAS": (("LAT", "LON"), data),
            },
            coords={"LON": lon, "LAT": lat},
            attrs={"title": f"HAMTIDE11a: {constituent} ocean tide"},
        )

        # Export
        filename = hamtide_dir / f"{constituent}.hamtide11a.nc"
        ds.to_netcdf(filename)
