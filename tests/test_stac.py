import pytest
import geopandas as gpd
from odc.geo.geom import polygon, BoundingBox
from eo_tides.stac import _get_bbox

# A reference polygon in EPSG:4326
GEOPOLYGON_4326 = polygon(
    outer=[
        (122.14, -18.08),
        (122.14, -17.92),
        (122.30, -17.92),
        (122.30, -18.08),
        (122.14, -18.08),
    ],
    crs="EPSG:4326",
)


# Test extraction of bounding box for STAC query
@pytest.mark.parametrize(
    "kwargs",
    [
        {"bbox": GEOPOLYGON_4326.boundingbox},
        {"bbox": tuple(GEOPOLYGON_4326.boundingbox)},
        {"bbox": list(GEOPOLYGON_4326.boundingbox)},
        {"geopolygon": gpd.GeoDataFrame(geometry=[GEOPOLYGON_4326.geom], crs="EPSG:4326")},
        {"geopolygon": gpd.GeoDataFrame(geometry=[GEOPOLYGON_4326.geom], crs="EPSG:4326").to_crs("EPSG:3577")},
        {"geopolygon": gpd.GeoSeries(GEOPOLYGON_4326.geom)},
        {"geopolygon": GEOPOLYGON_4326.geom},
        {"geopolygon": GEOPOLYGON_4326},
        {"lon": (122.14, 122.30), "lat": (-18.08, -17.92)},
    ],
)
def test_get_bbox(kwargs):
    bbox_4326, _ = _get_bbox(**kwargs)

    # Always geographic
    assert bbox_4326.crs.geographic

    # Always a BoundingBox
    assert isinstance(bbox_4326, BoundingBox)

    # Always intersects the reference polygon
    assert bbox_4326.polygon.intersects(GEOPOLYGON_4326)


def test_get_bbox_no_inputs():
    # Expect an exception when no inputs are provided
    with pytest.raises(Exception, match="Must provide both `lon` and `lat`, or `geopolygon`, or `bbox`."):
        _get_bbox()
