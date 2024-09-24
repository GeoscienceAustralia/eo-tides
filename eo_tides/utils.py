import numpy as np
from scipy.spatial import cKDTree as KDTree


def idw(
    input_z,
    input_x,
    input_y,
    output_x,
    output_y,
    p=1,
    k=10,
    max_dist=None,
    k_min=1,
    epsilon=1e-12,
):
    """Perform Inverse Distance Weighting (IDW) interpolation.

    This function performs fast IDW interpolation by creating a KDTree
    from the input coordinates then uses it to find the `k` nearest
    neighbors for each output point. Weights are calculated based on the
    inverse distance to each neighbor, with weights descreasing with
    increasing distance.

    Code inspired by: https://github.com/DahnJ/REM-xarray

    Parameters
    ----------
    input_z : array-like
        Array of values at the input points. This can be either a
        1-dimensional array, or a 2-dimensional array where each column
        (axis=1) represents a different set of values to be interpolated.
    input_x : array-like
        Array of x-coordinates of the input points.
    input_y : array-like
        Array of y-coordinates of the input points.
    output_x : array-like
        Array of x-coordinates where the interpolation is to be computed.
    output_y : array-like
        Array of y-coordinates where the interpolation is to be computed.
    p : int or float, optional
        Power function parameter defining how rapidly weightings should
        decrease as distance increases. Higher values of `p` will cause
        weights for distant points to decrease rapidly, resulting in
        nearby points having more influence on predictions. Defaults to 1.
    k : int, optional
        Number of nearest neighbors to use for interpolation. `k=1` is
        equivalent to "nearest" neighbour interpolation. Defaults to 10.
    max_dist : int or float, optional
        Restrict neighbouring points to less than this distance.
        By default, no distance limit is applied.
    k_min : int, optional
        If `max_dist` is provided, some points may end up with less than
        `k` nearest neighbours, potentially producing less reliable
        interpolations. Set `k_min` to set any points with less than
        `k_min` neighbours to NaN. Defaults to 1.
    epsilon : float, optional
        Small value added to distances to prevent division by zero
        errors in the case that output coordinates are identical to
        input coordinates. Defaults to 1e-12.

    Returns
    -------
    interp_values : numpy.ndarray
        Interpolated values at the output coordinates. If `input_z` is
        1-dimensional, `interp_values` will also be 1-dimensional. If
        `input_z` is 2-dimensional, `interp_values` will have the same
        number of rows as `input_z`, with each column (axis=1)
        representing interpolated values for one set of input data.

    Examples
    --------
    >>> input_z = [1, 2, 3, 4, 5]
    >>> input_x = [0, 1, 2, 3, 4]
    >>> input_y = [0, 1, 2, 3, 4]
    >>> output_x = [0.5, 1.5, 2.5]
    >>> output_y = [0.5, 1.5, 2.5]
    >>> idw(input_z, input_x, input_y, output_x, output_y, k=2)
    array([1.5, 2.5, 3.5])

    """
    # Convert to numpy arrays
    input_x = np.atleast_1d(input_x)
    input_y = np.atleast_1d(input_y)
    input_z = np.atleast_1d(input_z)
    output_x = np.atleast_1d(output_x)
    output_y = np.atleast_1d(output_y)

    # Verify input and outputs have matching lengths
    if not (input_z.shape[0] == len(input_x) == len(input_y)):
        raise ValueError("All of `input_z`, `input_x` and `input_y` must be the same length.")
    if not (len(output_x) == len(output_y)):
        raise ValueError("Both `output_x` and `output_y` must be the same length.")

    # Verify k is smaller than total number of points, and non-zero
    if k > input_z.shape[0]:
        raise ValueError(
            f"The requested number of nearest neighbours (`k={k}`) "
            f"is smaller than the total number of points ({input_z.shape[0]}).",
        )
    if k == 0:
        raise ValueError("Interpolation based on `k=0` nearest neighbours is not valid.")

    # Create KDTree to efficiently find nearest neighbours
    points_xy = np.column_stack((input_y, input_x))
    tree = KDTree(points_xy)

    # Determine nearest neighbours and distances to each
    grid_stacked = np.column_stack((output_y, output_x))
    distances, indices = tree.query(grid_stacked, k=k, workers=-1)

    # If k == 1, add an additional axis for consistency
    if k == 1:
        distances = distances[..., np.newaxis]
        indices = indices[..., np.newaxis]

    # Add small epsilon to distances to prevent division by zero errors
    # if output coordinates are the same as input coordinates
    distances = np.maximum(distances, epsilon)

    # Set distances above max to NaN if specified
    if max_dist is not None:
        distances[distances > max_dist] = np.nan

    # Calculate weights based on distance to k nearest neighbours.
    weights = 1 / np.power(distances, p)
    weights = weights / np.nansum(weights, axis=1).reshape(-1, 1)

    # 1D case: Compute weighted sum of input_z values for each output point
    if input_z.ndim == 1:
        interp_values = np.nansum(weights * input_z[indices], axis=1)

    # 2D case: Compute weighted sum for each set of input_z values
    # weights[..., np.newaxis] adds a dimension for broadcasting
    else:
        interp_values = np.nansum(
            weights[..., np.newaxis] * input_z[indices],
            axis=1,
        )

    # Set any points with less than `k_min` valid weights to NaN
    interp_values[np.isfinite(weights).sum(axis=1) < k_min] = np.nan

    return interp_values
