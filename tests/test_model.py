import pathlib

import numpy as np
import pandas as pd
import pytest
from pyTMD.compute import tide_elevations

from eo_tides.model import _set_directory, list_models, model_tides
from eo_tides.validation import eval_metrics

GAUGE_X = 122.2183
GAUGE_Y = -18.0008
ENSEMBLE_MODELS = ["EOT20", "HAMTIDE11"]  # simplified for tests


# Test available tide models
def test_list_models():
    # Using env var
    available_models, supported_models = list_models()
    assert available_models == ["EOT20", "GOT5.5", "HAMTIDE11"]
    assert len(supported_models) > 3

    # Not printing outputs
    available_models, supported_models = list_models(show_available=False, show_supported=False)
    assert available_models == ["EOT20", "GOT5.5", "HAMTIDE11"]

    # Providing a string path
    available_models, supported_models = list_models(directory="./tests/data/tide_models")
    assert available_models == ["EOT20", "GOT5.5", "HAMTIDE11"]

    # Providing a pathlib
    path = pathlib.Path("./tests/data/tide_models")
    available_models, supported_models = list_models(directory=path)
    assert available_models == ["EOT20", "GOT5.5", "HAMTIDE11"]


# Run test for multiple input coordinates, CRSs and interpolation methods
@pytest.mark.parametrize(
    "x, y, crs, method, model",
    [
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "bilinear", "EOT20"),  # WGS84, bilinear interp
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "spline", "EOT20"),  # WGS84, spline interp
        (
            -1034913,
            -1961916,
            "EPSG:3577",
            "bilinear",
            "EOT20",
        ),  # Australian Albers, bilinear interp
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "spline", "GOT5.5"),
        (GAUGE_X, GAUGE_Y, "EPSG:4326", "spline", "HAMTIDE11"),
    ],
)
def test_model_tides(measured_tides_ds, x, y, crs, method, model):
    # Run modelling for locations and timesteps in tide gauge data
    modelled_tides_df = model_tides(
        x=[x],
        y=[y],
        time=measured_tides_ds.time,
        crs=crs,
        method=method,
        model=model,
    )

    # Test that modelled tides contain correct headings and have same
    # number of timesteps
    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_height"]
    assert len(modelled_tides_df.index) == len(measured_tides_ds.time)

    # Run equivalent pyTMD code
    pytmd_tides = tide_elevations(
        x=x,
        y=y,
        delta_time=measured_tides_ds.time,
        DIRECTORY=_set_directory(None),
        MODEL=model,
        EPSG=int(crs[-4:]),
        TIME="datetime",
        EXTRAPOLATE=True,
        CUTOFF=np.inf,
        METHOD=method,
    )

    # Verify that pyTMD produces same results as `model_tides`
    assert np.allclose(modelled_tides_df.tide_height.values, pytmd_tides.data)

    # Compare measured and modelled tides
    val_stats = eval_metrics(x=measured_tides_ds.tide_height, y=modelled_tides_df.tide_height)

    # Test that modelled tides meet expected accuracy
    if model == "HAMTIDE11":
        assert val_stats["Correlation"] > 0.99
        assert val_stats["RMSE"] < 0.34
        assert val_stats["R-squared"] > 0.98
        assert abs(val_stats["Bias"]) < 0.20
    else:
        assert val_stats["Correlation"] > 0.99
        assert val_stats["RMSE"] < 0.27
        assert val_stats["R-squared"] > 0.99
        assert abs(val_stats["Bias"]) < 0.20


# Run tests for one or multiple models, and long and wide format outputs
@pytest.mark.parametrize(
    "models, output_format",
    [
        (["EOT20"], "long"),
        (["EOT20"], "wide"),
        (["EOT20", "GOT5.5", "HAMTIDE11"], "long"),
        (["EOT20", "GOT5.5", "HAMTIDE11"], "wide"),
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
        assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_height"]

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
    tide_range = modelled_tides_df.tide_height.max() - modelled_tides_df.tide_height.min()

    # Verify tide range and dtypes are as expected for unit
    assert np.isclose(tide_range, expected_range, rtol=0.01)
    assert modelled_tides_df.tide_height.dtype == expected_dtype


# Run test for each combination of mode, output format, and one or
# multiple tide models
@pytest.mark.parametrize(
    "mode, models, output_format",
    [
        ("one-to-many", ["EOT20"], "long"),
        ("one-to-one", ["EOT20"], "long"),
        ("one-to-many", ["EOT20"], "wide"),
        ("one-to-one", ["EOT20"], "wide"),
        ("one-to-many", ["EOT20", "HAMTIDE11"], "long"),
        ("one-to-one", ["EOT20", "HAMTIDE11"], "long"),
        ("one-to-many", ["EOT20", "HAMTIDE11"], "wide"),
        ("one-to-one", ["EOT20", "HAMTIDE11"], "wide"),
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
    good_hamtide11 = -17.58549, 123.59414
    good_eot20 = -17.1611, 123.3406
    y = [good_eot20[0], good_hamtide11[0]]
    x = [good_eot20[1], good_hamtide11[1]]

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
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_height"]
    assert all(modelled_tides_df.tide_model == "ensemble")

    # Default, ensemble + other models requested
    models = ["EOT20", "HAMTIDE11", "ensemble"]
    modelled_tides_df = model_tides(
        x=x,
        y=y,
        time=times,
        model=models,
        ensemble_models=ENSEMBLE_MODELS,
    )

    assert modelled_tides_df.index.names == ["time", "x", "y"]
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_height"]
    assert set(modelled_tides_df.tide_model) == set(models)
    assert np.allclose(
        modelled_tides_df.tide_height.values,
        [
            0.094,
            -3.202,
            0.409,
            -3.098,
            0.803,
            0.664,
            0.989,
            1.011,
            0.449,
            -1.269,
            0.699,
            -1.043,
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
    assert modelled_tides_df.columns.tolist() == ["tide_model", "tide_height"]
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
        0.5 * (modelled_tides_df.EOT20 + modelled_tides_df.HAMTIDE11),
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
        (modelled_tides_df.EOT20 == modelled_tides_df.ensemble)
        | (modelled_tides_df.HAMTIDE11 == modelled_tides_df.ensemble)
    )

    # Check that correct model is the closest at each row
    closer_model = modelled_tides_df.apply(
        lambda row: (
            "EOT20" if abs(row["ensemble"] - row["EOT20"]) < abs(row["ensemble"] - row["HAMTIDE11"]) else "HAMTIDE11"
        ),
        axis=1,
    ).tolist()
    assert closer_model == ["EOT20", "HAMTIDE11", "EOT20", "HAMTIDE11"]

    # Check values are expected
    assert np.allclose(modelled_tides_df.ensemble, [0.09, 0.98, -3.20, 1.01], atol=0.02)

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
        "EOT20",
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
        "EOT20",
        "HAMTIDE11",
        "ensemble-best",
        "ensemble-worst",
        "ensemble-mean-top2",
        "ensemble-mean-weighted",
        "ensemble-mean",
    ])
