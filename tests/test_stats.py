import numpy as np
import pandas as pd
import pytest

from eo_tides.stats import pixel_stats, tide_stats

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


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
    tidal_stats_df = tide_stats(
        satellite_ds,
        modelled_freq=modelled_freq,
    )

    # Compare outputs to expected results (within 2% or 0.02 m)
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
    assert np.allclose(tidal_stats_df, expected_results, atol=0.02)

    # Test linear regression
    tidal_stats_linreg_df = tide_stats(
        satellite_ds,
        modelled_freq=modelled_freq,
        linear_reg=True,
    )

    # Compare outputs to expected results (within 2% or 0.02 m)
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
        "observed_slope": 6.952,
        "observed_pval": 0.573,
    })
    assert np.allclose(tidal_stats_linreg_df, expected_results, atol=0.02)


# Run test for multiple modelled frequencies
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
    expected_vars = ["hat", "hot", "lat", "lot", "otr", "tr", "spread", "offset_low", "offset_high"]
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
