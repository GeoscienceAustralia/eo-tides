import numpy as np
import pytest

from eo_tides.utils import idw


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
