import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from eo_tides.model import model_tides
from eo_tides.utils import (
    _set_directory,
    _standardise_models,
    _standardise_time,
    clip_models,
    idw,
    list_models,
)


# Run once per module run to generate symethic HAMTIDE11 files; autouse=True
# allows this to run without being specifically called in tests
@pytest.fixture(scope="module", autouse=True)
def create_synthetic_hamtide11(base_dir="tests/data/tide_models_synthetic"):
    """Generates and exports synthetic HAMTIDE11 model data
    to test clipping functionality.
    """
    base_dir = pathlib.Path(base_dir)  # Ensure base_dir is a Path object

    # Create coordinate arrays
    lon = np.arange(0, 360.125, 0.125)  # 2881 points
    lat = np.arange(-90, 90.125, 0.125)  # 1441 points

    # List of HAMTIDE11 tidal constituents
    constituents = ["2n", "k1", "k2", "m2", "n2", "o1", "p1", "q1", "s2"]

    # Create HAMTIDE11 output directory
    hamtide_dir = base_dir / "hamtide"
    hamtide_dir.mkdir(parents=True, exist_ok=True)

    # Create and save a NetCDF for each constituent
    for constituent in constituents:
        # Create synthetic HAMTIDE11 dataset with random data
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


# Run once per module run to generate symethic EOT20 files; autouse=True
# allows this to run without being specifically called in tests
@pytest.fixture(scope="module", autouse=True)
def create_synthetic_eot20(base_dir="tests/data/tide_models_synthetic"):
    """Generates and exports synthetic EOT20 model data
    to test clipping functionality.
    """
    base_dir = pathlib.Path(base_dir)  # Ensure base_dir is a Path object

    # Create coordinate arrays
    lon = np.arange(0, 360.125, 0.125)  # 2881 points
    lat = np.arange(-90, 90.125, 0.125)  # 1441 points

    # List of EOT20 tidal constituents
    constituents = [
        "2N2",
        "J1",
        "K1",
        "K2",
        "M2",
        "M4",
        "MF",
        "MM",
        "N2",
        "O1",
        "P1",
        "Q1",
        "S1",
        "S2",
        "SA",
        "SSA",
        "T2",
    ]

    # Create EOT20 output directory
    eot20_dir = base_dir / "EOT20/ocean_tides"
    eot20_dir.mkdir(parents=True, exist_ok=True)

    # Create and save a NetCDF for each constituent
    for constituent in constituents:
        # Create synthetic EOT20 dataset with random data
        shape = (len(lat), len(lon))  # 1441, 2881
        data = np.random.random(shape).astype(np.float64)

        # Add NaN values to match original
        mask = np.random.random(shape) < 0.2
        data[mask] = np.nan

        # Create the dataset
        ds = xr.Dataset(
            {
                "amplitude": (("lat", "lon"), data),
                "phase": (("lat", "lon"), data),
                "imag": (("lat", "lon"), data),
                "real": (("lat", "lon"), data),
            },
            coords={"lat": lat, "lon": lon},
            attrs={"title": "DGFI-TUM global empirical ocean tide model"},
        )

        # Export
        filename = eot20_dir / f"{constituent}_ocean_eot20.nc"
        ds.to_netcdf(filename)


@pytest.mark.parametrize(
    "model, ensemble_models, exp_process, exp_request, exp_ensemble",
    [
        # Case 1, 2: Specific model in str and list format
        ("EOT20", None, ["EOT20"], ["EOT20"], None),
        (["EOT20"], None, ["EOT20"], ["EOT20"], None),
        # Case 3, 4: Using "all" to request all available models
        (
            "all",
            None,
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            None,
        ),
        (
            ["all"],
            None,
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            ["EOT20", "GOT5.5", "HAMTIDE11"],
            None,
        ),
        # Case 5, 6: Using "ensemble" to model tides for specific set of ensemble models
        (
            "ensemble",
            ["EOT20", "HAMTIDE11"],
            ["EOT20", "HAMTIDE11"],
            ["ensemble"],
            ["EOT20", "HAMTIDE11"],
        ),
        (
            ["ensemble"],
            ["EOT20", "HAMTIDE11"],
            ["EOT20", "HAMTIDE11"],
            ["ensemble"],
            ["EOT20", "HAMTIDE11"],
        ),
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


# Test expected failures during model standardisation
@pytest.mark.parametrize(
    "model, ensemble_models, err_msg",
    [
        # Case 1: Duplicate models
        (["EOT20", "EOT20"], None, "duplicate values"),
        # Case 2: Invalid model
        (["bad_model"], None, "not valid"),
        # Case 3: Valid but unavailable model
        (["FES2012"], None, "not available"),
        # Case 4: Unavailable ensemble model
        (["ensemble"], ["EOT20", "FES2012"], "ensemble models are not available"),
    ],
    ids=["duplicate_model", "invalid_model", "unavailable_model", "unavailable_ensemble"],
)
def test_standardise_models_errors(model, ensemble_models, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        _standardise_models(
            model=model,
            directory="tests/data/tide_models",
            ensemble_models=ensemble_models,
        )


# Use monkeypatch to test setting and unsetting environment var
@pytest.mark.parametrize(
    "directory,env_var,expected_exception",
    [
        # Case 1: No directory, no env var → Exception
        (None, None, Exception),
        # Case 2: Directory set, but path doesn't exist → FileNotFoundError
        ("/some/nonexistent/path", None, FileNotFoundError),
        # Case 3: Env var set, but path doesn't exist → FileNotFoundError
        (None, "/some/nonexistent/path", FileNotFoundError),
    ],
    ids=["no_directory_or_env", "invalid_dir", "invalid_env_var"],
)
def test_set_directory_errors(monkeypatch, directory, env_var, expected_exception):
    # Remove or modify env var if required
    if env_var is None:
        monkeypatch.delenv("EO_TIDES_TIDE_MODELS", raising=False)
    else:
        monkeypatch.setenv("EO_TIDES_TIDE_MODELS", env_var)

    with pytest.raises(expected_exception):
        _set_directory(directory)


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
    output_files = {i.stem for i in out_dir.iterdir()}
    assert output_files == {"GOT5", "EOT20", "hamtide"}

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
        (
            "EOT20",
            (-166, 14, -151, 29),
            (19.60, -155.46),
            "hawaii",
        ),  # entirely W of prime meridian
        ("EOT20", (-13, 49, 6, 60), (51.47, 0.84), "uk"),  # crossing prime meridian
        (
            "EOT20",
            (105, -48, 160, -5),
            (-25.59, 153.03),
            "aus",
        ),  # entirely E of prime meridian
        (
            "EOT20",
            (-257, 7, -120, 63),
            (19.59, -155.45),
            "pacific",
        ),  # crossing antimeridian
        (
            "HAMTIDE11",
            (-166, 14, -151, 29),
            (19.60, -155.46),
            "hawaii",
        ),  # entirely W of prime meridian
        ("HAMTIDE11", (-13, 49, 6, 60), (51.47, 0.84), "uk"),  # crossing prime meridian
        (
            "HAMTIDE11",
            (105, -48, 160, -5),
            (-25.59, 153.03),
            "aus",
        ),  # entirely E of prime meridian
        (
            "HAMTIDE11",
            (-257, 7, -120, 63),
            (19.59, -155.45),
            "pacific",
        ),  # crossing antimeridian
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
        (
            datetime(2020, 1, 12, 21, 14),
            np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]"),
        ),
        # Case 3: Single pandas.Timestamp
        (
            pd.Timestamp("2020-01-12 21:14"),
            np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]"),
        ),
        # Case 4: np.datetime64 scalar
        (
            np.datetime64("2020-01-12T21:14:00"),
            np.array(["2020-01-12T21:14:00"], dtype="datetime64[ns]"),
        ),
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
            np.array(
                ["2000-01-01T00:00:00", "2000-01-01T12:00:00", "2000-01-02T00:00:00"],
                dtype="datetime64[ns]",
            ),
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
    assert len(supported_models) > 3  # noqa: PLR2004

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


# Test running extra_databases models from dict and file
@pytest.mark.parametrize(
    "extra_databases",
    [
        # Extra database as a JSON file
        ["./tests/data/extra_database.json"],
        # Extra database as a dictionary
        [
            {
                "elevation": {
                    "EOT20_custom": {
                        "format": "FES-netcdf",
                        "model_file": [
                            "EOT20/ocean_tides/2N2_ocean_eot20.nc",
                            "EOT20/ocean_tides/J1_ocean_eot20.nc",
                            "EOT20/ocean_tides/K1_ocean_eot20.nc",
                            "EOT20/ocean_tides/K2_ocean_eot20.nc",
                            "EOT20/ocean_tides/M2_ocean_eot20.nc",
                            "EOT20/ocean_tides/M4_ocean_eot20.nc",
                            "EOT20/ocean_tides/MF_ocean_eot20.nc",
                            "EOT20/ocean_tides/MM_ocean_eot20.nc",
                            "EOT20/ocean_tides/N2_ocean_eot20.nc",
                            "EOT20/ocean_tides/O1_ocean_eot20.nc",
                            "EOT20/ocean_tides/P1_ocean_eot20.nc",
                            "EOT20/ocean_tides/Q1_ocean_eot20.nc",
                            "EOT20/ocean_tides/S1_ocean_eot20.nc",
                            "EOT20/ocean_tides/S2_ocean_eot20.nc",
                            "EOT20/ocean_tides/SA_ocean_eot20.nc",
                            "EOT20/ocean_tides/SSA_ocean_eot20.nc",
                            "EOT20/ocean_tides/T2_ocean_eot20.nc",
                        ],
                        "name": "EOT20_custom",
                        "reference": "https://doi.org/10.17882/79489",
                        "scale": 0.01,
                        "type": "z",
                        "variable": "tide_ocean",
                        "version": "EOT20",
                    }
                }
            }
        ],
    ],
    ids=["file", "dict"],
)
def test_list_models_extra_databases(extra_databases):
    # Verify that custom models are added to lists of
    # available and supported models
    available_models, supported_models = list_models(
        extra_databases=extra_databases,
    )
    assert "EOT20_custom" in available_models
    assert "EOT20_custom" in supported_models


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
