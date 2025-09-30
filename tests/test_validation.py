import numpy as np
import pytest
from odc.geo.geom import point

from eo_tides.validation import load_gauge_gesla, ndwi_tide_corr

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


# Run test for different spatial searches
@pytest.mark.parametrize(
    ("x", "y", "site_code", "max_distance", "correct_mean", "expected"),
    [
        # Test nearest gauge lookup
        (GAUGE_X, GAUGE_Y, None, None, False, ["62650"]),
        (-117.4, 32.6, None, None, False, ["569A"]),
        (152.0, -33.0, None, None, True, ["60370"]),
        pytest.param(
            GAUGE_X + 1,
            GAUGE_Y,
            None,
            0.1,
            False,
            ["62650"],
            marks=pytest.mark.xfail(reason="No nearest gauge"),
        ),
        # Test bounding box lookup
        ((GAUGE_X - 0.2, GAUGE_X + 0.2), (GAUGE_Y - 0.2, GAUGE_Y + 0.2), None, None, False, ["62650"]),
        ((100, 160), (-5, -45), None, None, False, ["60370", "62650"]),
        # Test site_code lookup
        (None, None, "62650", None, False, ["62650"]),
        (None, None, ["60370", "62650"], None, False, ["60370", "62650"]),
    ],
    ids=[
        "broome_xy",
        "sandiego_xy",
        "syd_xy_correctmean",
        "no_nearest",
        "broome_bbox",
        "aus_bbox",
        "broome_code",
        "aus_code",
    ],
)
def test_load_gauge_gesla(x, y, site_code, max_distance, correct_mean, expected):
    # Load gauge data
    gauge_df = load_gauge_gesla(
        x=x,
        y=y,
        site_code=site_code,
        max_distance=max_distance,
        correct_mean=correct_mean,
        time=("2018-01-01", "2018-01-20"),
        data_path="tests/data/GESLA4_ALL/",
        metadata_path="tests/data/GESLA4_ALL.csv",
    )

    assert "sea_level" in gauge_df.columns
    assert set(gauge_df.index.unique(level="site_code")) == set(expected)

    # Verify that mean is near 0 after subtracting mean from time series
    if correct_mean:
        assert np.isclose(gauge_df.sea_level.mean().item(), 0.0, atol=0.01)


def test_ndwi_tide_corr():
    # Sample point in King Sound with variable model performance
    y, x = -16.99636, 123.61017

    # Calculate NDWI-tide correlations
    corr_df, corr_da = ndwi_tide_corr(
        x=x,
        y=y,
        time=("2024-09", "2024-12"),
        cloud_cover=30,
    )

    # Verify HAMTIDE11 comes out with lowest rank
    assert corr_df.loc["HAMTIDE11", "rank"] == 3

    # Verify correlations are approximately correct
    assert np.allclose(corr_df.correlation, [0.77, 0.77, -0.12], atol=0.02)

    # Verify valid percentages are between 0 and 1
    assert corr_df["valid_perc"].between(0, 1).all()

    # Verify data array contains expected dimensions and values
    assert "tide_model" in corr_da.dims
    assert "time" not in corr_da.dims
    assert set(corr_da.tide_model.values) == set(["EOT20", "GOT5.5", "HAMTIDE11"])

    # Assert that data envelops original point
    corr_da.odc.geobox.extent.intersects(point(x, y, crs="EPSG:4326").to_crs(corr_da.odc.crs))
