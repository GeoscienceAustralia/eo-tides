import numpy as np
import odc.stac
import pandas as pd
import pystac_client
import pytest

from eo_tides.stats import tide_stats

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


# Create test data in different CRSs and resolutions
@pytest.fixture(
    params=[
        ("EPSG:3577", 30),  # Australian Albers 30 m pixels
        ("EPSG:4326", 0.00025),  # WGS84, 0.0025 degree pixels
    ],
    ids=["satellite_ds_epsg3577", "satellite_ds_epsg4326"],
)
def satellite_ds(request):
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


# Run test for multiple modelled frequencies
@pytest.mark.parametrize(
    "modelled_freq",
    [
        ("2h"),  # Model tides every two hours
        ("120min"),  # Model tides every 120 minutes
    ],
)
def test_tidal_stats(satellite_ds, modelled_freq):
    # Calculate tidal stats
    tidal_stats_df = tide_stats(satellite_ds, modelled_freq=modelled_freq)

    # Compare outputs to expected results (within 5% or 0.05 m)
    expected_results = pd.Series({
        "tidepost_lat": -18.001,
        "tidepost_lon": 122.218,
        "observed_mean_m": -0.417,
        "all_mean_m": -0.005,
        "observed_min_m": -2.141,
        "all_min_m": -4.321,
        "observed_max_m": 1.674,
        "all_max_m": 4.259,
        "observed_range_m": 3.814,
        "all_range_m": 8.580,
        "spread": 0.445,
        "low_tide_offset": 0.254,
        "high_tide_offset": 0.301,
    })
    assert np.allclose(tidal_stats_df, expected_results, atol=0.05)
