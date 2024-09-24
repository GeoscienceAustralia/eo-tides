import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac_client
import pytest
import xarray as xr

from eo_tides.model import model_tides, pixel_tides
from eo_tides.validation import eval_metrics

GAUGE_X = 122.2183
GAUGE_Y = -18.0008
ENSEMBLE_MODELS = ["FES2014", "HAMTIDE11"]  # simplified for tests


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
    measured_tides_df.columns = ["tide_m"]

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


# Run test for multiple input coordinates, CRSs and interpolation methods
@pytest.mark.parametrize(
    "x, y, crs, method",
    [
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "bilinear"),  # WGS84, bilinear interp
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "spline"),  # WGS84, spline interp
        (
            -1034913,
            -1961916,
            "EPSG:3577",
            "bilinear",
        ),  # Australian Albers, bilinear interp
    ],
)
def test_model_tides(measured_tides_ds, x, y, crs, method):
    # Run FES2014 tidal model for locations and timesteps in tide gauge data
    modelled_tides_df = model_tides(
        x=[x],
        y=[y],
        time=measured_tides_ds.time,
        crs=crs,
        method=method,
    )

    # Compare measured and modelled tides
    val_stats = eval_metrics(x=measured_tides_ds.tide_m, y=modelled_tides_df.tide_m)

    # Test that modelled tides contain correct headings and have same
    # number of timesteps
    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_m"]
    assert len(modelled_tides_df.index) == len(measured_tides_ds.time)

    # Test that modelled tides meet expected accuracy
    assert val_stats["Correlation"] > 0.99
    assert val_stats["RMSE"] < 0.26
    assert val_stats["R-squared"] > 0.96
    assert abs(val_stats["Bias"]) < 0.20


# Run tests for one or multiple models, and long and wide format outputs
@pytest.mark.parametrize(
    "models, output_format",
    [
        (["FES2014"], "long"),
        (["FES2014"], "wide"),
        (["FES2014", "HAMTIDE11"], "long"),
        (["FES2014", "HAMTIDE11"], "wide"),
    ],
    ids=[
        "single_model_long",
        "single_model_wide",
        "multiple_models_long",
        "multiple_models_wide",
    ],
)
def test_model_tides_multiplemodels(measured_tides_ds, models, output_format):
    # Model tides for one or multiple tide models and output formats
    modelled_tides_df = model_tides(
        x=[GAUGE_X],
        y=[GAUGE_Y],
        time=measured_tides_ds.time,
        model=models,
        output_format=output_format,
    )

    if output_format == "long":
        # Verify output has correct columns
        assert modelled_tides_df.index.names == ["time", "x", "y"]
        assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_m"]

        # Verify tide model column contains correct values
        assert modelled_tides_df.tide_model.unique().tolist() == models

        # Verify that dataframe has length of original timesteps multipled by
        # n models
        assert len(modelled_tides_df.index) == len(measured_tides_ds.time) * len(models)

    elif output_format == "wide":
        # Verify output has correct columns
        assert modelled_tides_df.index.names == ["time", "x", "y"]
        assert modelled_tides_df.columns.tolist() == models

        # Verify output has same length as orginal timesteps
        assert len(modelled_tides_df.index) == len(measured_tides_ds.time)


# Run tests for each unit, providing expected outputs
@pytest.mark.parametrize(
    "units, expected_range, expected_dtype",
    [("m", 10, "float32"), ("cm", 1000, "int16"), ("mm", 10000, "int16")],
    ids=["metres", "centimetres", "millimetres"],
)
def test_model_tides_units(measured_tides_ds, units, expected_range, expected_dtype):
    # Model tides
    modelled_tides_df = model_tides(
        x=[GAUGE_X],
        y=[GAUGE_Y],
        time=measured_tides_ds.time,
        output_units=units,
    )

    # Calculate tide range
    tide_range = modelled_tides_df.tide_m.max() - modelled_tides_df.tide_m.min()

    # Verify tide range and dtypes are as expected for unit
    assert np.isclose(tide_range, expected_range, rtol=0.01)
    assert modelled_tides_df.tide_m.dtype == expected_dtype


# Run test for each combination of mode, output format, and one or
# multiple tide models
@pytest.mark.parametrize(
    "mode, models, output_format",
    [
        ("one-to-many", ["FES2014"], "long"),
        ("one-to-one", ["FES2014"], "long"),
        ("one-to-many", ["FES2014"], "wide"),
        ("one-to-one", ["FES2014"], "wide"),
        ("one-to-many", ["FES2014", "HAMTIDE11"], "long"),
        ("one-to-one", ["FES2014", "HAMTIDE11"], "long"),
        ("one-to-many", ["FES2014", "HAMTIDE11"], "wide"),
        ("one-to-one", ["FES2014", "HAMTIDE11"], "wide"),
    ],
)
def test_model_tides_mode(mode, models, output_format):
    # Input params
    x = [122.14, 122.30, 122.12]
    y = [-17.91, -17.92, -18.07]
    times = pd.date_range("2020", "2021", periods=3)

    # Model tides
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        mode=mode,
        output_format=output_format,
        model=models,
    )

    if mode == "one-to-one":
        if output_format == "wide":
            # Should have the same number of rows as input x, y, times
            assert len(modelled_tides_df.index) == len(x)
            assert len(modelled_tides_df.index) == len(times)

            # Output indexes should match order of input x, y, times
            assert all(modelled_tides_df.index.get_level_values("time") == times)
            assert all(modelled_tides_df.index.get_level_values("x") == x)
            assert all(modelled_tides_df.index.get_level_values("y") == y)

        elif output_format == "long":
            # In "long" format, the number of x, y points multiplied by
            # the number of tide models
            assert len(modelled_tides_df.index) == len(x) * len(models)

            # Verify index values match expected x, y, time order
            assert all(modelled_tides_df.index.get_level_values("time") == np.tile(times, len(models)))
            assert all(modelled_tides_df.index.get_level_values("x") == np.tile(x, len(models)))
            assert all(modelled_tides_df.index.get_level_values("y") == np.tile(y, len(models)))

    if mode == "one-to-many":
        if output_format == "wide":
            # In "wide" output format, the number of rows should equal
            # the number of x, y points multiplied by timesteps
            assert len(modelled_tides_df.index) == len(x) * len(times)

            # TODO: Work out what order rows should be returned in in
            # "one-to-many" and "wide" mode

        elif output_format == "long":
            # In "long" output format, the number of rows should equal
            # the number of x, y points multiplied by timesteps and
            # the number of tide models
            assert len(modelled_tides_df.index) == len(x) * len(times) * len(models)

            # Verify index values match expected x, y, time order
            assert all(modelled_tides_df.index.get_level_values("time") == np.tile(times, len(x) * len(models)))
            assert all(modelled_tides_df.index.get_level_values("x") == np.tile(np.repeat(x, len(times)), len(models)))
            assert all(modelled_tides_df.index.get_level_values("y") == np.tile(np.repeat(y, len(times)), len(models)))


# Test ensemble modelling functionality
def test_model_tides_ensemble():
    # Input params
    x = [122.14, 144.910368]
    y = [-17.91, -37.919491]
    times = pd.date_range("2020", "2021", periods=2)

    # Default, only ensemble requested
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model="ensemble",
        ensemble_models=ENSEMBLE_MODELS,
    )

    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_m"]
    assert all(modelled_tides_df.tide_model == "ensemble")

    # Default, ensemble + other models requested
    models = ["FES2014", "HAMTIDE11", "ensemble"]
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        ensemble_models=ENSEMBLE_MODELS,
    )

    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_m"]
    assert set(modelled_tides_df.tide_model) == set(models)
    assert np.allclose(
        modelled_tides_df.tide_m,
        [
            -2.831,
            -1.897,
            -0.207,
            0.035,
            -2.655,
            -1.772,
            0.073,
            -0.071,
            -2.743,
            -1.835,
            -0.067,
            -0.018,
        ],
        atol=0.02,
    )

    # One-to-one mode
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        mode="one-to-one",
        ensemble_models=ENSEMBLE_MODELS,
    )

    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_m"]
    assert set(modelled_tides_df.tide_model) == set(models)

    # Wide mode, default
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        output_format="wide",
        ensemble_models=ENSEMBLE_MODELS,
    )

    # Check that expected models exist, and that ensemble is approx average
    # of other two models
    assert set(modelled_tides_df.columns) == set(models)
    assert np.allclose(
        0.5 * (modelled_tides_df.FES2014 + modelled_tides_df.HAMTIDE11),
        modelled_tides_df.ensemble,
    )

    # Wide mode, top n == 1
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        output_format="wide",
        ensemble_top_n=1,
        ensemble_models=ENSEMBLE_MODELS,
    )

    # Check that expected models exist, and that ensemble is equal to at
    # least one of the other models
    assert set(modelled_tides_df.columns) == set(models)
    assert all(
        (modelled_tides_df.FES2014 == modelled_tides_df.ensemble)
        | (modelled_tides_df.HAMTIDE11 == modelled_tides_df.ensemble)
    )

    # Check that correct model is the closest at each row
    closer_model = modelled_tides_df.apply(
        lambda row: (
            "FES2014"
            if abs(row["ensemble"] - row["FES2014"]) < abs(row["ensemble"] - row["HAMTIDE11"])
            else "HAMTIDE11"
        ),
        axis=1,
    ).tolist()
    assert closer_model == ["FES2014", "HAMTIDE11", "FES2014", "HAMTIDE11"]

    # Check values are expected
    assert np.allclose(modelled_tides_df.ensemble, [-2.830, 0.073, -1.900, -0.072], atol=0.02)

    # Wide mode, custom functions
    ensemble_funcs = {
        "ensemble-best": lambda x: x["rank"] == 1,
        "ensemble-worst": lambda x: x["rank"] == 2,
        "ensemble-mean-top2": lambda x: x["rank"].isin([1, 2]),
        "ensemble-mean-weighted": lambda x: 3 - x["rank"],
        "ensemble-mean": lambda x: x["rank"] <= 2,
    }
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        output_format="wide",
        ensemble_func=ensemble_funcs,
        ensemble_models=ENSEMBLE_MODELS,
    )

    # Check that expected models exist, and that valid data is produced
    assert set(modelled_tides_df.columns) == set([
        "FES2014",
        "HAMTIDE11",
        "ensemble-best",
        "ensemble-worst",
        "ensemble-mean-top2",
        "ensemble-mean-weighted",
        "ensemble-mean",
    ])
    assert all(modelled_tides_df.notnull())

    # Long mode, custom functions
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        output_format="long",
        ensemble_func=ensemble_funcs,
        ensemble_models=ENSEMBLE_MODELS,
    )

    # Check that expected models exist in "tide_model" column
    assert set(modelled_tides_df.tide_model) == set([
        "FES2014",
        "HAMTIDE11",
        "ensemble-best",
        "ensemble-worst",
        "ensemble-mean-top2",
        "ensemble-mean-weighted",
        "ensemble-mean",
    ])


# Run tests for default and custom resolutions
@pytest.mark.parametrize("resolution", [None, "custom"])
def test_pixel_tides(satellite_ds, measured_tides_ds, resolution):
    # Use different custom resolution depending on CRS
    if resolution == "custom":
        resolution = 0.1 if satellite_ds.odc.geobox.crs.geographic else 10000

    # Model tides using `pixel_tides`
    modelled_tides_ds, modelled_tides_lowres = pixel_tides(satellite_ds, resolution=resolution)

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
    gauge_stats = eval_metrics(x=measured_tides_ds.tide_m, y=modelled_tides_gauge)

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

    # Test if extracted tides match expected results (to within ~3 cm)
    expected_tides = [-0.66, -0.76, -0.75, -0.82]
    assert np.allclose(extracted_tides.values, expected_tides, atol=0.03)
