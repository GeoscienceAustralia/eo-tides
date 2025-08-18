import numpy as np
import pandas as pd
import pytest

from eo_tides.stats import pixel_stats, tide_stats, tide_aliasing, MAJOR_CONSTITUENTS

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


# Run test for multiple modelled frequencies
@pytest.mark.parametrize(
    "modelled_freq, tidepost_lon, tidepost_lat",
    [
        ("2h", None, None),  # Model tides every two hours
        ("120min", None, None),  # Model tides every 120 minutes
        ("2h", 122.218, -18.001),  # Custom tidepost
    ],
)
def test_tidal_stats(satellite_ds, modelled_freq, tidepost_lon, tidepost_lat):
    # Calculate tidal stats
    tidal_stats_df = tide_stats(
        satellite_ds,
        modelled_freq=modelled_freq,
        tidepost_lon=tidepost_lon,
        tidepost_lat=tidepost_lat,
    )

    # Compare outputs to expected results (within 2% or 0.02 m)
    expected_results = pd.Series({
        "mot": -0.407,
        "mat": -0.005,
        "hot": 1.684,
        "hat": 4.275,
        "lot": -2.141,
        "lat": -4.339,
        "otr": 3.825,
        "tr": 8.614,
        "spread": 0.444,
        "offset_low": 0.255,
        "offset_high": 0.301,
        "x": 122.218,
        "y": -18.001,
    })
    assert np.allclose(tidal_stats_df, expected_results, atol=0.02)


# Run test for one or multiple model inputs
@pytest.mark.parametrize(
    "models",
    [
        (["EOT20"]),
        (["EOT20", "GOT5.5"]),
    ],
)
def test_tidal_stats_models(satellite_ds, models):
    # Calculate tidal stats
    tidal_stats_df = tide_stats(
        satellite_ds,
        model=models,
    )

    # If multiple models, verify data is a pandas.DataFrame with expected rows
    if len(models) > 1:
        assert isinstance(tidal_stats_df, pd.DataFrame)
        assert len(tidal_stats_df.index) == len(models)
        assert models == tidal_stats_df.index.get_level_values("tide_model").tolist()

    # If just one, verify data is a pandas.Series
    else:
        assert isinstance(tidal_stats_df, pd.Series)


# Test if plotting a custom variable runs without errors
def test_tide_stats_plotvar(satellite_ds):
    # Test on custom coordinate
    satellite_ds_withcoords = satellite_ds.assign_coords(coord=("time", [1, 1, 2, 2, 3, 4, 5]))
    tide_stats(
        satellite_ds_withcoords,
        plot_var="coord",
    )

    # Test on custom data variable
    satellite_ds_withvar = satellite_ds.assign(var=("time", [1, 1, 2, 2, 3, 4, 5]))
    tide_stats(
        satellite_ds_withvar,
        plot_var="var",
    )

    # Test configuring color when plotting variable
    tide_stats(
        satellite_ds_withvar,
        plot_var="var",
        point_col="red",
    )

    # Test when not plotting variable
    tide_stats(
        satellite_ds_withvar,
        point_col="red",
    )


# Run test for multiple models and with resampling on and off
@pytest.mark.parametrize(
    "models, resample",
    [
        (["EOT20"], False),
        (["EOT20", "GOT5.5"], False),
        (["EOT20"], True),
    ],
)
def test_pixel_stats(satellite_ds, models, resample):
    stats_ds = pixel_stats(
        satellite_ds,
        model=models,
        resample=resample,
    )

    # Verify dims are correct
    assert stats_ds.odc.spatial_dims == satellite_ds.odc.spatial_dims

    # Verify vars are as expected
    expected_vars = [
        "mot",
        "mat",
        "hot",
        "hat",
        "lot",
        "lat",
        "otr",
        "tr",
        "spread",
        "offset_low",
        "offset_high",
    ]
    assert set(expected_vars) == set(stats_ds.data_vars)

    # Verify tide models are correct
    assert all(stats_ds["tide_model"].values == models)
    if len(models) > 1:
        assert "tide_model" in stats_ds.dims

    # If resample, assert that statistics have the same shape and dims
    # as `satellite_ds`
    if resample:
        assert satellite_ds.odc.geobox.shape == stats_ds.odc.geobox.shape

    # Verify values are roughly expected
    assert np.allclose(stats_ds.offset_high.mean().item(), 0.30, atol=0.02)
    assert np.allclose(stats_ds.offset_low.mean().item(), 0.27, atol=0.02)
    assert np.allclose(stats_ds.spread.mean().item(), 0.43, atol=0.02)


@pytest.mark.parametrize(
    "satellites, constituents, units, style, expect_error",
    [
        (["landsat"], ["m2", "k1"], "days", False, None),
        (["sentinel-2"], None, "hours", True, None),
        (["swot", "landsat"], ["k1"], "years", False, None),
        (["landsat"], ["mm"], "days", False, None),
        (["invalid-sat"], ["m2"], "days", False, ValueError),
        (["landsat"], ["m2"], "centuries", False, ValueError),
        ({"custom-sat": 6}, None, "hours", True, None),
        ({"custom-sat1": 6, "custom-sat2": 10}, None, "hours", True, None),
        (["landsat"], ["invalid-constituent"], "days", False, ValueError),
    ],
)
def test_tide_aliasing(satellites, constituents, units, style, expect_error):
    if expect_error:
        with pytest.raises(expect_error):
            tide_aliasing(satellites, constituents=constituents, units=units, style=style)
    else:
        result = tide_aliasing(satellites, constituents=constituents, units=units, style=style)

        # Verify output is a dataframe
        if style:
            assert isinstance(result, pd.io.formats.style.Styler)
        else:
            assert isinstance(result, pd.DataFrame)

        # Verify correct columns are included
        assert "name" in result.columns
        assert "period" in result.columns

        # Verify correct satellites are included
        for sat in satellites:
            assert ("aliasing_period", sat) in result.columns

        # Verify correct constituents are included
        if constituents is not None:
            assert result.index.equals(pd.Index(constituents, name="constituents"))
        else:
            assert result.index.equals(pd.Index(MAJOR_CONSTITUENTS.keys(), name="constituents"))
