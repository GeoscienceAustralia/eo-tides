import numpy as np
import pytest

from eo_tides.validation import load_gauge_gesla

GAUGE_X = 122.2183
GAUGE_Y = -18.0008


# Run test for different spatial searches
@pytest.mark.parametrize(
    "x, y, site_code, max_distance, correct_mean, expected",
    [
        # Test nearest gauge lookup
        (GAUGE_X, GAUGE_Y, None, None, False, ["62650"]),
        (-117.4, 32.6, None, None, False, ["569A"]),
        (152.0, -33.0, None, None, True, ["60370"]),
        pytest.param(
            GAUGE_X + 1, GAUGE_Y, None, 0.1, False, ["62650"], marks=pytest.mark.xfail(reason="No nearest gauge")
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
        data_path="tests/data/",
        metadata_path="tests/data/GESLA3_ALL 2.csv",
    )

    assert "sea_level" in gauge_df.columns
    assert set(gauge_df.index.unique(level="site_code")) == set(expected)

    # Verify that mean is near 0 after subtracting mean from time series
    if correct_mean:
        assert np.isclose(gauge_df.sea_level.mean().item(), 0.0, atol=0.01)
