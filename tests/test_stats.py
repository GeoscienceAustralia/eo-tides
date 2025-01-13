import numpy as np
import pandas as pd
import pytest

from eo_tides.stats import pixel_stats, tide_stats

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
        "mot": -0.417,
        "mat": -0.005,
        "hot": 1.674,
        "hat": 4.259,
        "lot": -2.141,
        "lat": -4.321,
        "otr": 3.814,
        "tr": 8.580,
        "spread": 0.445,
        "offset_low": 0.254,
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
    expected_vars = ["mot", "mat", "hot", "hat", "lot", "lat", "otr", "tr", "spread", "offset_low", "offset_high"]
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
