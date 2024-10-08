import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac_client
import pytest
import xarray as xr

from eo_tides.eo import pixel_tides, tag_tides
from eo_tides.validation import eval_metrics

GAUGE_X = 122.2183
GAUGE_Y = -18.0008
ENSEMBLE_MODELS = ["EOT20", "HAMTIDE11"]  # simplified for tests


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


@pytest.mark.parametrize(
    "ebb_flow, swap_dims, tidepost_lat, tidepost_lon",
    [
        (False, False, None, None),  # Run with default settings
        (True, False, None, None),  # Run with ebb_flow on
        (False, True, None, None),  # Run with swap_dims on
        (False, False, GAUGE_Y, GAUGE_X),  # Run with custom tide posts
    ],
)
def test_tag_tides(satellite_ds, measured_tides_ds, ebb_flow, swap_dims, tidepost_lat, tidepost_lon):
    # Use tag_tides to assign a "tide_height" variable to each observation
    tagged_tides_ds = tag_tides(
        satellite_ds,
        ebb_flow=ebb_flow,
        swap_dims=swap_dims,
        tidepost_lat=tidepost_lat,
        tidepost_lon=tidepost_lon,
    )

    # Verify tide_height variable was added
    assert "tide_height" in tagged_tides_ds

    # Verify ebb_flow variable was added if requested
    if ebb_flow:
        assert "ebb_flow" in tagged_tides_ds

    if swap_dims:
        # Verify "tide_height" is now a dimension
        assert "tide_height" in tagged_tides_ds.dims

        # Test that "tide_height" dim is same length as satellite "time" dim
        assert len(tagged_tides_ds.tide_height) == len(satellite_ds.time)

        # Test that first value on "tide_height" dim is lower than last
        # (data should be sorted in increasing tide height order)
        assert tagged_tides_ds.isel(tide_height=0).tide_height < tagged_tides_ds.isel(tide_height=-1).tide_height

    else:
        # Test that tagged tides have same timesteps as satellite data
        assert len(tagged_tides_ds.tide_height.time) == len(satellite_ds.time)

        # Interpolate measured tide data to same timesteps
        measured_tides_ds = measured_tides_ds.interp(time=satellite_ds.time, method="linear")

        # Compare measured and modelled tides
        val_stats = eval_metrics(x=measured_tides_ds.tide_height, y=tagged_tides_ds.tide_height)

        # Test that modelled tides meet expected accuracy
        assert val_stats["Correlation"] > 0.99
        assert val_stats["RMSE"] < 0.26
        assert val_stats["R-squared"] > 0.96
        assert abs(val_stats["Bias"]) < 0.20


def test_tag_tides_multiple(satellite_ds):
    # Model multiple models at once
    tagged_tides_ds = tag_tides(satellite_ds, model=["EOT20", "HAMTIDE11"], ebb_flow=True)

    assert "tide_model" in tagged_tides_ds.dims
    assert tagged_tides_ds.tide_height.dims == ("time", "tide_model")
    assert tagged_tides_ds.ebb_flow.dims == ("time", "tide_model")

    # Test that multiple tide models are correlated
    val_stats = eval_metrics(
        x=tagged_tides_ds.sel(tide_model="EOT20").tide_height,
        y=tagged_tides_ds.sel(tide_model="HAMTIDE11").tide_height,
    )
    assert val_stats["Correlation"] >= 0.99


# Run tests for default and custom resolutions
@pytest.mark.parametrize("resolution", [None, "custom"])
def test_pixel_tides(satellite_ds, measured_tides_ds, resolution):
    # Use different custom resolution depending on CRS
    if resolution == "custom":
        resolution = 0.1 if satellite_ds.odc.geobox.crs.geographic else 10000

    # Model tides using `pixel_tides`
    modelled_tides_ds = pixel_tides(satellite_ds, resolution=resolution)

    # Model tides using `pixel_tides` with resample=False
    modelled_tides_lowres = pixel_tides(satellite_ds, resample=False)

    # Interpolate measured tide data to same timesteps
    measured_tides_ds = measured_tides_ds.interp(time=satellite_ds.time, method="linear")

    # Assert that modelled tides have the same shape and dims as
    # arrays in `satellite_ds`
    assert modelled_tides_ds.shape == satellite_ds.nbart_red.shape
    assert modelled_tides_ds.dims == satellite_ds.nbart_red.dims

    # Assert that high res and low res data have the same dims
    assert modelled_tides_ds.dims == modelled_tides_lowres.dims

    # Test through time at tide gauge

    # Create tide gauge point, and reproject to dataset CRS
    tide_gauge_point = gpd.points_from_xy(
        x=[GAUGE_X],
        y=[GAUGE_Y],
        crs="EPSG:4326",
    ).to_crs(satellite_ds.odc.geobox.crs)

    try:
        modelled_tides_gauge = modelled_tides_ds.sel(
            y=tide_gauge_point[0].y,
            x=tide_gauge_point[0].x,
            method="nearest",
        )
    except KeyError:
        modelled_tides_gauge = modelled_tides_ds.sel(
            latitude=tide_gauge_point[0].y,
            longitude=tide_gauge_point[0].x,
            method="nearest",
        )

    # Calculate accuracy stats
    gauge_stats = eval_metrics(x=measured_tides_ds.tide_height, y=modelled_tides_gauge)

    # Assert pixel_tide outputs are accurate
    assert gauge_stats["Correlation"] > 0.99
    assert gauge_stats["RMSE"] < 0.26
    assert gauge_stats["R-squared"] > 0.96
    assert abs(gauge_stats["Bias"]) < 0.20

    # Test spatially for a single timestep at corners of array

    # Create test points, reproject to dataset CRS, and extract coords
    # as xr.DataArrays so we can select data from our array
    points = gpd.points_from_xy(
        x=[122.14438, 122.30304, 122.12964, 122.29235],
        y=[-17.91625, -17.92713, -18.07656, -18.08751],
        crs="EPSG:4326",
    ).to_crs(satellite_ds.odc.geobox.crs)
    x_coords = xr.DataArray(points.x, dims=["point"])
    y_coords = xr.DataArray(points.y, dims=["point"])

    # Extract modelled tides for each corner
    try:
        extracted_tides = modelled_tides_ds.sel(x=x_coords, y=y_coords, time="2020-01-29", method="nearest")
    except KeyError:
        extracted_tides = modelled_tides_ds.sel(
            longitude=x_coords, latitude=y_coords, time="2020-01-29", method="nearest"
        )

    # Test if extracted tides match expected results (to within ~5 cm)
    expected_tides = [-0.68, -0.84, -0.80, -0.88]
    assert np.allclose(extracted_tides.values, expected_tides, atol=0.05)


# Run tests for default and custom resolutions
def test_pixel_tides_times(satellite_ds, measured_tides_ds):
    custom_times = pd.date_range(
        start="2022-01-01",
        end="2022-01-05",
        freq="6H",
    )

    # Verify that correct times are included on output
    measured_tides_ds = pixel_tides(satellite_ds, times=custom_times)
    assert all(measured_tides_ds.time.values == custom_times)
    assert len(measured_tides_ds.time) == len(custom_times)

    # Verify passing a list
    measured_tides_ds = pixel_tides(satellite_ds, times=custom_times.tolist())
    assert len(measured_tides_ds.time) == len(custom_times)

    # Verify passing a single timestamp
    measured_tides_ds = pixel_tides(satellite_ds, times=custom_times.tolist()[0])
    assert len(measured_tides_ds) == 1

    # Verify that passing a dataset without time leads to error
    satellite_ds_notime = satellite_ds.isel(time=0)
    with pytest.raises(ValueError):
        pixel_tides(satellite_ds_notime)

    # Verify passing a dataset without time and custom times
    measured_tides_ds = pixel_tides(satellite_ds_notime, times=custom_times)
    assert len(measured_tides_ds.time) == len(custom_times)
