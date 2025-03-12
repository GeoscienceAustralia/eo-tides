import pathlib
import tempfile
from datetime import datetime

import numpy as np
import odc.geo.geom
import pandas as pd
import pytest

from eo_tides.model import model_tides
from eo_tides.utils import _standardise_models, _standardise_time, clip_models, idw, list_models


@pytest.mark.parametrize(
    "model, ensemble_models, exp_process, exp_request, exp_ensemble",
    [
        # Case 1, 2: Specific model in str and list format
        ("EOT20", None, ["EOT20"], ["EOT20"], None),
        (["EOT20"], None, ["EOT20"], ["EOT20"], None),
        # Case 3, 4: Using "all" to request all available models
        ("all", None, ["EOT20", "GOT5.5", "HAMTIDE11"], ["EOT20", "GOT5.5", "HAMTIDE11"], None),
        (["all"], None, ["EOT20", "GOT5.5", "HAMTIDE11"], ["EOT20", "GOT5.5", "HAMTIDE11"], None),
        # Case 5, 6: Using "ensemble" to model tides for specific set of ensemble models
        ("ensemble", ["EOT20", "HAMTIDE11"], ["EOT20", "HAMTIDE11"], ["ensemble"], ["EOT20", "HAMTIDE11"]),
        (["ensemble"], ["EOT20", "HAMTIDE11"], ["EOT20", "HAMTIDE11"], ["ensemble"], ["EOT20", "HAMTIDE11"]),
        # Case 7: Modelling tides using ensemble set and an additional model
        (
            ["ensemble", "GOT5.5"],
            ["EOT20", "HAMTIDE11"],
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            ["ensemble", "GOT5.5"],
            ["EOT20", "HAMTIDE11"],
        ),
        # Case 8: Modelling tides for all available models, AND ensemble set
        (
            ["all", "ensemble"],
            ["EOT20", "HAMTIDE11"],
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            ["EOT20", "GOT5.5", "HAMTIDE11", "ensemble"],
            ["EOT20", "HAMTIDE11"],
        ),
    ],
)
def test_standardise_models(model, ensemble_models, exp_process, exp_request, exp_ensemble):
    # Return lists of models
    models_to_process, models_requested, ensemble_models = _standardise_models(
        model=model,
        directory="tests/data/tide_models",
        ensemble_models=ensemble_models,
    )

    assert models_to_process == exp_process
    assert models_requested == exp_request
    assert (sorted(ensemble_models) if ensemble_models else None) == (sorted(exp_ensemble) if exp_ensemble else None)


def test_clip_models():
    # Set input and output paths
    in_dir = "tests/data/tide_models"
    out_dir = pathlib.Path("tests/data/tide_models_clipped")

    # Clip models to bbox
    clip_models(
        input_directory=in_dir,
        output_directory=out_dir,
        bbox=(122.27, -18.07, 122.29, -18.05),
    )

    # Assert that files were exported for all available models
    output_files = set([i.stem for i in out_dir.iterdir()])
    assert output_files == set(["GOT5", "EOT20", "hamtide"])

    # Set modelling location
    x, y = 122.28, -18.06
    time = pd.date_range(start="2000-01", end="2001-02", freq="5h")

    # Model using unclipped vs clipped files
    df_unclipped = model_tides(
        x=x,
        y=y,
        time=time,
        model="HAMTIDE11",
        directory=in_dir,
        # crop=False,
    )
    df_clipped = model_tides(
        x=x,
        y=y,
        time=time,
        model="HAMTIDE11",
        directory=out_dir,
    )

    # Verify both produce the same results
    assert np.allclose(df_unclipped.tide_height, df_clipped.tide_height)


# Test clipping across multiple global locations using synthetic tide data
@pytest.mark.parametrize(
    "model, bbox, point, name",
    [
        ("EOT20", (-166, 14, -151, 29), (19.60, -155.46), "hawaii"),  # entirely W of prime meridian
        ("EOT20", (-13, 49, 6, 60), (51.47, 0.84), "uk"),  # crossing prime meridian
        ("EOT20", (105, -48, 160, -5), (-25.59, 153.03), "aus"),  # entirely E of prime meridian
        ("EOT20", (-257, 7, -120, 63), (19.59, -155.45), "pacific"),  # crossing antimeridian
        ("HAMTIDE11", (-166, 14, -151, 29), (19.60, -155.46), "hawaii"),  # entirely W of prime meridian
        ("HAMTIDE11", (-13, 49, 6, 60), (51.47, 0.84), "uk"),  # crossing prime meridian
        ("HAMTIDE11", (105, -48, 160, -5), (-25.59, 153.03), "aus"),  # entirely E of prime meridian
        ("HAMTIDE11", (-257, 7, -120, 63), (19.59, -155.45), "pacific"),  # crossing antimeridian
    ],
)
def test_clip_models_bbox(model, bbox, point, name):
    # Set input and output paths
    in_dir = "tests/data/tide_models_synthetic/"
    out_dir = f"tests/data/tide_models_synthetic_{name}/"

    # Clip models to input bbox
    clip_models(
        input_directory=in_dir,
        output_directory=out_dir,
        bbox=bbox,
        model=model,
        overwrite=True,
    )

    # Set modelling location based on bbox centroid
    y, x = point
    time = pd.date_range(start="2000-01", end="2001-02", freq="5h")

    # Model using unclipped vs clipped files
    df_unclipped = model_tides(
        x=x,
        y=y,
        time=time,
        model=model,
        directory=in_dir,
    )
    df_clipped = model_tides(
        x=x,
        y=y,
        time=time,
        model=model,
        directory=out_dir,
    )

    # Verify both produce the same results
    assert np.allclose(df_unclipped.tide_height, df_clipped.tide_height)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        # Case 1: None
        (None, None),
        # Case 2: Single datetime.datetime object
        (datetime(2020, 1, 12, 21, 14), np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]")),
        # Case 3: Single pandas.Timestamp
        (pd.Timestamp("2020-01-12 21:14"), np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]")),
        # Case 4: np.datetime64 scalar
        (np.datetime64("2020-01-12T21:14:00"), np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]")),
        # Case 5: 1D numpy array of np.datetime64
        (
            np.array(["2020-01-12T21:14:00", "2021-02-14T15:30:00"], dtype="datetime64[ns]"),
            np.array(["2020-01-12T21:14:00", "2021-02-14T15:30:00"], dtype="datetime64[ns]"),
        ),
        # Case 6: 1D numpy array of datetime.datetime
        (
            np.array([datetime(2020, 1, 12, 21, 14), datetime(2021, 2, 14, 15, 30)]),
            np.array(["2020-01-12T21:14:00", "2021-02-14T15:30:00"], dtype="datetime64[ns]"),
        ),
        # Case 7: pandas.DatetimeIndex
        (
            pd.date_range(start="2000-01-01", end="2000-01-02", periods=3),
            np.array(["2000-01-01T00:00:00", "2000-01-01T12:00:00", "2000-01-02T00:00:00"], dtype="datetime64[ns]"),
        ),
        # Case 8: Mixed array with datetime.datetime and np.datetime64
        (
            np.array([datetime(2020, 1, 12, 21, 14), np.datetime64("2021-02-14T15:30:00")]),
            np.array(["2020-01-12T21:14:00", "2021-02-14T15:30:00"], dtype="datetime64[ns]"),
        ),
        # Case 9: Single string datetime
        ("2020-01-12 21:14", np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]")),
        # Case 10: Array of string datetimes
        (
            ["2020-01-12 21:14", "2021-02-14 15:30"],
            np.array(["2020-01-12T21:14:00", "2021-02-14T15:30:00"], dtype="datetime64[ns]"),
        ),
    ],
)
def test_standardise_time(input_value, expected_output):
    result = _standardise_time(input_value)
    if result is None:
        assert result == expected_output
    else:
        assert np.array_equal(result, expected_output)


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


# Test Inverse Distance Weighted function
def test_idw():
    # Basic psuedo-1D example
    input_z = [1, 2, 3, 4, 5]
    input_x = [0, 1, 2, 3, 4]
    input_y = [0, 0, 0, 0, 0]
    output_x = [0.5, 1.5, 2.5, 3.5]
    output_y = [0.0, 0.0, 0.0, 0.0]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=2)
    assert np.allclose(out, [1.5, 2.5, 3.5, 4.5])

    # Verify that k > input points gives error
    with pytest.raises(ValueError):
        idw(input_z, input_x, input_y, output_x, output_y, k=6)

    # 2D nearest neighbour case
    input_z = [1, 2, 3, 4]
    input_x = [0, 4, 0, 4]
    input_y = [0, 0, 4, 4]
    output_x = [1, 4, 0, 3]
    output_y = [0, 1, 3, 4]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=1)
    assert np.allclose(out, [1, 2, 3, 4])

    # Two neighbours
    input_z = [1, 2, 3, 4]
    input_x = [0, 4, 0, 4]
    input_y = [0, 0, 4, 4]
    output_x = [2, 0, 4, 2]
    output_y = [0, 2, 2, 4]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=2)
    assert np.allclose(out, [1.5, 2, 3, 3.5])

    # Four neighbours
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4)
    assert np.allclose(out, [2.11, 2.30, 2.69, 2.88], rtol=0.01)

    # Four neighbours; max distance of 2
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4, max_dist=2)
    assert np.allclose(out, [1.5, 2, 3, 3.5])

    # Four neighbours; max distance of 2, k_min of 4 (should return NaN)
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4, max_dist=2, k_min=4)
    assert np.isnan(out).all()

    # Four neighbours; power function p=0
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4, p=0)
    assert np.allclose(out, [2.5, 2.5, 2.5, 2.5])

    # Four neighbours; power function p=2
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4, p=2)
    assert np.allclose(out, [1.83, 2.17, 2.83, 3.17], rtol=0.01)

    # Different units, nearest neighbour case
    input_z = [10, 20, 30, 40]
    input_x = [1125296, 1155530, 1125296, 1155530]
    input_y = [-4169722, -4169722, -4214782, -4214782]
    output_x = [1124952, 1159593, 1120439, 1155284]
    output_y = [-4169749, -4172892, -4211108, -4214332]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=1)
    assert np.allclose(out, [10, 20, 30, 40])

    # Verify distance works on different units
    output_x = [1142134, 1138930]
    output_y = [-4171232, -4213451]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=4, max_dist=20000)
    assert np.allclose(out, [15, 35], rtol=0.1)

    # Test multidimensional input
    input_z = np.column_stack(([1, 2, 3, 4], [10, 20, 30, 40]))
    input_x = [0, 4, 0, 4]
    input_y = [0, 0, 4, 4]
    output_x = [1, 4, 0, 3]
    output_y = [0, 1, 3, 4]
    out = idw(input_z, input_x, input_y, output_x, output_y, k=1)
    assert input_z.shape == out.shape
    assert np.allclose(out[:, 0], [1, 2, 3, 4])
    assert np.allclose(out[:, 1], [10, 20, 30, 40])
