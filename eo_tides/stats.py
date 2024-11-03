# Used to postpone evaluation of type annotations
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# Only import if running type checking
if TYPE_CHECKING:
    import xarray as xr
    from odc.geo.geobox import GeoBox

from .eo import _standardise_inputs, pixel_tides, tag_tides
from .model import model_tides
from .utils import DatetimeLike


def _plot_biases(
    all_tides_df,
    obs_tides_da,
    lat,
    lot,
    hat,
    hot,
    offset_low,
    offset_high,
    spread,
    plot_col,
    obs_linreg,
    obs_x,
    all_timerange,
):
    """
    Plot tide bias statistics as a figure, including both
    satellite observations and all modelled tides.
    """

    # Create plot and add all time and observed tide data
    fig, ax = plt.subplots(figsize=(10, 6))
    all_tides_df.reset_index(["x", "y"]).tide_height.plot(ax=ax, alpha=0.4, label="Modelled tides")

    # Look through custom column values if provided
    if plot_col is not None:
        # Create a list of marker styles
        markers = [
            "o",
            "^",
            "s",
            "D",
            "v",
            "<",
            ">",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "d",
            "|",
            "_",
        ]
        for i, value in enumerate(np.unique(plot_col)):
            obs_tides_da.sel(time=plot_col == value).plot.line(
                ax=ax,
                linewidth=0.0,
                color="black",
                marker=markers[i % len(markers)],
                markersize=4,
                label=value,
            )
    # Otherwise, plot all data at once
    else:
        obs_tides_da.plot.line(
            ax=ax,
            marker="o",
            linewidth=0.0,
            color="black",
            markersize=3.5,
            label="Satellite observations",
        )

    # Add legend and remove title
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=20,
        borderaxespad=0,
        frameon=False,
    )
    ax.set_title("")

    # Add linear regression line
    if obs_linreg is not None:
        ax.plot(
            obs_tides_da.time.isel(time=[0, -1]),
            obs_linreg.intercept + obs_linreg.slope * obs_x[[0, -1]],
            "r",
            label="fitted line",
        )

    # Add horizontal lines for spread/offsets
    ax.axhline(lot, color="black", linestyle=":", linewidth=1)
    ax.axhline(hot, color="black", linestyle=":", linewidth=1)
    ax.axhline(lat, color="black", linestyle=":", linewidth=1)
    ax.axhline(hat, color="black", linestyle=":", linewidth=1)

    # Add text annotations for spread/offsets
    ax.annotate(
        f"    High tide\n    offset ({offset_high:.0%})",
        xy=(all_timerange.max(), np.mean([hat, hot])),
        va="center",
    )
    ax.annotate(
        f"    Spread\n    ({spread:.0%})",
        xy=(all_timerange.max(), np.mean([lot, hot])),
        va="center",
    )
    ax.annotate(
        f"    Low tide\n    offset ({offset_low:.0%})",
        xy=(all_timerange.max(), np.mean([lat, lot])),
    )

    # Remove top right axes and add labels
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Tide height (m)")
    ax.set_xlabel("")
    ax.margins(x=0.015)

    return fig


def tide_stats(
    data: xr.Dataset | xr.DataArray | GeoBox,
    time: DatetimeLike | None = None,
    model: str = "EOT20",
    directory: str | os.PathLike | None = None,
    tidepost_lat: float | None = None,
    tidepost_lon: float | None = None,
    plain_english: bool = True,
    plot: bool = True,
    plot_col: str | None = None,
    modelled_freq: str = "3h",
    linear_reg: bool = False,
    min_max_q: tuple = (0.0, 1.0),
    round_stats: int = 3,
    **model_tides_kwargs,
) -> pd.Series:
    """
    Takes a multi-dimensional dataset and generate tide statistics
    and satellite-observed tide bias metrics, calculated based on
    every timestep in the satellte data and the geographic centroid
    of the imagery.

    By comparing the subset of tides observed by satellites
    against the full astronomical tidal range, we can evaluate
    whether the tides observed by satellites are biased
    (e.g. fail to observe either the highest or lowest tides).

    For more information about the tidal statistics computed by this
    function, refer to Figure 8 in Bishop-Taylor et al. 2018:
    <https://www.sciencedirect.com/science/article/pii/S0272771418308783#fig8>

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or odc.geo.geobox.GeoBox
        A multi-dimensional dataset or GeoBox pixel grid that will
        be used to calculate tide statistics. If `data` is an
        xarray object, it should include a "time" dimension.
        If no "time" dimension exists or if `data` is a GeoBox,
        then times must be passed using the `time` parameter.
    time : DatetimeLike, optional
        By default, tides will be modelled using times from the
        "time" dimension of `data`. Alternatively, this param can
        be used to provide a custom set of times. Accepts any format
        that can be converted by `pandas.to_datetime()`. For example:
        `time=pd.date_range(start="2000", end="2001", freq="5h")`
    model : str, optional
        The tide model to use to model tides. Defaults to "EOT20";
        for a full list of available/supported models, run
        `eo_tides.model.list_models`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    tidepost_lat, tidepost_lon : float or int, optional
        Optional coordinates used to model tides. The default is None,
        which uses the centroid of the dataset as the tide modelling
        location.
    plain_english : bool, optional
        An optional boolean indicating whether to print a plain english
        version of the tidal statistics to the screen. Defaults to True.
    plot : bool, optional
        An optional boolean indicating whether to plot how satellite-
        observed tide heights compare against the full tidal range.
        Defaults to True.
    plot_col : str, optional
        Optional name of a coordinate, dimension or variable in the array
        that will be used to plot observations with unique symbols.
        Defaults to None, which will plot all observations as circles.
    modelled_freq : str, optional
        An optional string giving the frequency at which to model tides
        when computing the full modelled tidal range. Defaults to '3h',
        which computes a tide height for every three hours across the
        temporal extent of `data`.
    linear_reg: bool, optional
        Whether to return linear regression statistics that assess
        whether satellite-observed tides show any decreasing  or
        increasing trends over time. This may indicate whether your
        satellite data may produce misleading trends based on uneven
        sampling of the local tide regime.
    min_max_q : tuple, optional
        Quantiles used to calculate max and min observed and modelled
        astronomical tides. By default `(0.0, 1.0)` which is equivalent
        to minimum and maximum; to use a softer threshold that is more
        robust to outliers, use e.g. `(0.1, 0.9)`.
    round_stats : int, optional
        The number of decimal places used to round the output statistics.
        Defaults to 3.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

    Returns
    -------
    stats_df : pandas.Series
        A `pandas.Series` containing the following statistics:

        - `y`: latitude used for modelling tide heights
        - `x`: longitude used for modelling tide heights
        - `mot`: mean tide height observed by the satellite (metres)
        - `mat`: mean modelled astronomical tide height (metres)
        - `lot`: minimum tide height observed by the satellite (metres)
        - `lat`: minimum tide height from modelled astronomical tidal range (metres)
        - `hot`: maximum tide height observed by the satellite (metres)
        - `hat`: maximum tide height from modelled astronomical tidal range (metres)
        - `otr`: tidal range observed by the satellite (metres)
        - `tr`: modelled astronomical tide range (metres)
        - `spread`: proportion of the full modelled tidal range observed by the satellite
        - `offset_low`: proportion of the lowest tides never observed by the satellite
        - `offset_high`: proportion of the highest tides never observed by the satellite

        If `linear_reg = True`, the output will also contain:

        - `observed_slope`: slope of any relationship between observed tide heights and time
        - `observed_pval`: significance/p-value of any relationship between observed tide heights and time
    """
    # Standardise data inputs, time and models
    gbox, time_coords = _standardise_inputs(data, time)

    # Verify that only one tide model is provided
    if isinstance(model, list):
        raise Exception("Only single tide models are supported by `tide_stats`.")

    # If custom tide modelling locations are not provided, use the
    # dataset centroid
    if not tidepost_lat or not tidepost_lon:
        tidepost_lon, tidepost_lat = gbox.geographic_extent.centroid.coords[0]

    # Model tides for each observation in the supplied xarray object
    assert time_coords is not None
    obs_tides_da = tag_tides(
        gbox,
        time=time_coords,
        model=model,
        directory=directory,
        tidepost_lat=tidepost_lat,  # type: ignore
        tidepost_lon=tidepost_lon,  # type: ignore
        return_tideposts=True,
        **model_tides_kwargs,
    )
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        obs_tides_da = obs_tides_da.reindex_like(data)

    # Generate range of times covering entire period of satellite record
    all_timerange = pd.date_range(
        start=time_coords.min().item(),
        end=time_coords.max().item(),
        freq=modelled_freq,
    )

    # Model tides for each timestep
    all_tides_df = model_tides(
        x=tidepost_lon,  # type: ignore
        y=tidepost_lat,  # type: ignore
        time=all_timerange,
        model=model,
        directory=directory,
        crs="EPSG:4326",
        **model_tides_kwargs,
    )

    # Get coarse statistics on all and observed tidal ranges
    obs_mean = obs_tides_da.mean().item()
    all_mean = all_tides_df.tide_height.mean()
    obs_min, obs_max = obs_tides_da.quantile(min_max_q).values
    all_min, all_max = all_tides_df.tide_height.quantile(min_max_q).values

    # Calculate tidal range
    obs_range = obs_max - obs_min
    all_range = all_max - all_min

    # Calculate Bishop-Taylor et al. 2018 tidal metrics
    spread = obs_range / all_range
    low_tide_offset_m = abs(all_min - obs_min)
    high_tide_offset_m = abs(all_max - obs_max)
    low_tide_offset = low_tide_offset_m / all_range
    high_tide_offset = high_tide_offset_m / all_range

    # Plain text descriptors
    mean_diff = "higher" if obs_mean > all_mean else "lower"
    mean_diff_icon = "⬆️" if obs_mean > all_mean else "⬇️"
    spread_icon = "🟢" if spread >= 0.9 else "🟡" if 0.7 < spread <= 0.9 else "🔴"
    low_tide_icon = "🟢" if low_tide_offset <= 0.1 else "🟡" if 0.1 <= low_tide_offset < 0.2 else "🔴"
    high_tide_icon = "🟢" if high_tide_offset <= 0.1 else "🟡" if 0.1 <= high_tide_offset < 0.2 else "🔴"

    # Extract x (time in decimal years) and y (distance) values
    obs_x = (
        obs_tides_da.time.dt.year + ((obs_tides_da.time.dt.dayofyear - 1) / 365) + ((obs_tides_da.time.dt.hour) / 24)
    )
    obs_y = obs_tides_da.values.astype(np.float32)

    # Compute linear regression
    obs_linreg = stats.linregress(x=obs_x, y=obs_y)

    if plain_english:
        print(f"\n\n🌊 Modelled astronomical tide range: {all_range:.2f} metres.")
        print(f"🛰️ Observed tide range: {obs_range:.2f} metres.\n")
        print(f"{spread_icon} {spread:.0%} of the modelled astronomical tide range was observed at this location.")
        print(
            f"{high_tide_icon} The highest {high_tide_offset:.0%} ({high_tide_offset_m:.2f} metres) of the tide range was never observed."
        )
        print(
            f"{low_tide_icon} The lowest {low_tide_offset:.0%} ({low_tide_offset_m:.2f} metres) of the tide range was never observed.\n"
        )
        print(f"🌊 Mean modelled astronomical tide height: {all_mean:.2f} metres.")
        print(f"🛰️ Mean observed tide height: {obs_mean:.2f} metres.\n")
        print(
            f"{mean_diff_icon} The mean observed tide height was {obs_mean - all_mean:.2f} metres {mean_diff} than the mean modelled astronomical tide height."
        )

        if linear_reg:
            if obs_linreg.pvalue > 0.01:
                print("➖ Observed tides showed no significant trends over time.")
            else:
                obs_slope_desc = "decreasing" if obs_linreg.slope < 0 else "increasing"
                print(
                    f"⚠️ Observed tides showed a significant {obs_slope_desc} trend over time (p={obs_linreg.pvalue:.3f}, {obs_linreg.slope:.2f} metres per year)"
                )

    if plot:
        _plot_biases(
            all_tides_df=all_tides_df,
            obs_tides_da=obs_tides_da,
            lat=all_min,
            lot=obs_min,
            hat=all_max,
            hot=obs_max,
            offset_low=low_tide_offset,
            offset_high=high_tide_offset,
            spread=spread,
            plot_col=data[plot_col] if plot_col else None,
            obs_linreg=obs_linreg if linear_reg else None,
            obs_x=obs_x,
            all_timerange=all_timerange,
        )

    # Export pandas.Series containing tidal stats
    output_stats = {
        "y": tidepost_lat,
        "x": tidepost_lon,
        "mot": obs_mean,
        "mat": all_mean,
        "lot": obs_min,
        "lat": all_min,
        "hot": obs_max,
        "hat": all_max,
        "otr": obs_range,
        "tr": all_range,
        "spread": spread,
        "offset_low": low_tide_offset,
        "offset_high": high_tide_offset,
    }

    if linear_reg:
        output_stats.update({
            "observed_slope": obs_linreg.slope,
            "observed_pval": obs_linreg.pvalue,
        })

    # Return pandas data
    stats_df = pd.Series(output_stats).round(round_stats)
    return stats_df


def pixel_stats(
    data: xr.Dataset | xr.DataArray | GeoBox,
    time: DatetimeLike | None = None,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    resample: bool = False,
    modelled_freq: str = "3h",
    min_max_q: tuple[float, float] = (0.0, 1.0),
    extrapolate: bool = True,
    cutoff: float = 10,
    **pixel_tides_kwargs,
) -> xr.Dataset:
    """
    Takes a multi-dimensional dataset and generate two-dimensional
    tide statistics and satellite-observed tide bias metrics,
    calculated based on every timestep in the satellte data and
    modelled into the spatial extent of the imagery.

    By comparing the subset of tides observed by satellites
    against the full astronomical tidal range, we can evaluate
    whether the tides observed by satellites are biased
    (e.g. fail to observe either the highest or lowest tides).

    Compared to `tide_stats`, this function models tide metrics
    spatially to produce a two-dimensional output.

    For more information about the tidal statistics computed by this
    function, refer to Figure 8 in Bishop-Taylor et al. 2018:
    <https://www.sciencedirect.com/science/article/pii/S0272771418308783#fig8>

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or odc.geo.geobox.GeoBox
        A multi-dimensional dataset or GeoBox pixel grid that will
        be used to calculate 2D tide statistics. If `data`
        is an xarray object, it should include a "time" dimension.
        If no "time" dimension exists or if `data` is a GeoBox,
        then times must be passed using the `time` parameter.
    time : DatetimeLike, optional
        By default, tides will be modelled using times from the
        "time" dimension of `data`. Alternatively, this param can
        be used to provide a custom set of times. Accepts any format
        that can be converted by `pandas.to_datetime()`. For example:
        `time=pd.date_range(start="2000", end="2001", freq="5h")`
    model : str or list of str, optional
        The tide model (or models) to use to model tides. If a list is
        provided, a new "tide_model" dimension will be added to `data`.
        Defaults to "EOT20"; for a full list of available/supported
        models, run `eo_tides.model.list_models`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    resample : bool, optional
        Whether to resample tide statistics back into `data`'s original
        higher resolution grid. Defaults to False, which will return
        lower-resolution statistics that are typically sufficient for
        most purposes.
    modelled_freq : str, optional
        An optional string giving the frequency at which to model tides
        when computing the full modelled tidal range. Defaults to '3h',
        which computes a tide height for every three hours across the
        temporal extent of `data`.
    min_max_q : tuple, optional
        Quantiles used to calculate max and min observed and modelled
        astronomical tides. By default `(0.0, 1.0)` which is equivalent
        to minimum and maximum; to use a softer threshold that is more
        robust to outliers, use e.g. `(0.1, 0.9)`.
    extrapolate : bool, optional
        Whether to extrapolate tides for x and y coordinates outside of
        the valid tide modelling domain using nearest-neighbor. Defaults
        to True.
    cutoff : float, optional
        Extrapolation cutoff in kilometers. To avoid producing tide
        statistics too far inland, the default is 10 km.
    **pixel_tides_kwargs :
        Optional parameters passed to the `eo_tides.eo.pixel_tides`
        function.

    Returns
    -------
    stats_ds : xarray.Dataset
        An `xarray.Dataset` containing the following statistics as two-dimensional data variables:

        - `lot`: minimum tide height observed by the satellite (metres)
        - `lat`: minimum tide height from modelled astronomical tidal range (metres)
        - `hot`: maximum tide height observed by the satellite (metres)
        - `hat`: maximum tide height from modelled astronomical tidal range (metres)
        - `otr`: tidal range observed by the satellite (metres)
        - `tr`: modelled astronomical tide range (metres)
        - `spread`: proportion of the full modelled tidal range observed by the satellite
        - `offset_low`: proportion of the lowest tides never observed by the satellite
        - `offset_high`: proportion of the highest tides never observed by the satellite

    """
    # Standardise data inputs, time and models
    gbox, time_coords = _standardise_inputs(data, time)
    model = [model] if isinstance(model, str) else model

    # Model observed tides
    assert time_coords is not None
    obs_tides = pixel_tides(
        gbox,
        time=time_coords,
        resample=False,
        model=model,
        directory=directory,
        calculate_quantiles=min_max_q,
        extrapolate=extrapolate,
        cutoff=cutoff,
        **pixel_tides_kwargs,
    )

    # Generate times covering entire period of satellite record
    all_timerange = pd.date_range(
        start=time_coords.min().item(),
        end=time_coords.max().item(),
        freq=modelled_freq,
    )

    # Model all tides
    all_tides = pixel_tides(
        gbox,
        time=all_timerange,
        model=model,
        directory=directory,
        calculate_quantiles=min_max_q,
        resample=False,
        extrapolate=extrapolate,
        cutoff=cutoff,
        **pixel_tides_kwargs,
    )

    # # Calculate means
    # TODO: Find way to make this work with `calculate_quantiles`
    # mot = obs_tides.mean(dim="time")
    # mat = all_tides.mean(dim="time")

    # Calculate min and max tides
    lot = obs_tides.isel(quantile=0)
    hot = obs_tides.isel(quantile=-1)
    lat = all_tides.isel(quantile=0)
    hat = all_tides.isel(quantile=-1)

    # Calculate tidal range
    otr = hot - lot
    tr = hat - lat

    # Calculate Bishop-Taylor et al. 2018 tidal metrics
    spread = otr / tr
    offset_low_m = abs(lat - lot)
    offset_high_m = abs(hat - hot)
    offset_low = offset_low_m / tr
    offset_high = offset_high_m / tr

    # Combine into a single dataset
    stats_ds = (
        xr.merge(
            [
                # mot.rename("mot"),
                # mat.rename("mat"),
                hot.rename("hot"),
                hat.rename("hat"),
                lot.rename("lot"),
                lat.rename("lat"),
                otr.rename("otr"),
                tr.rename("tr"),
                spread.rename("spread"),
                offset_low.rename("offset_low"),
                offset_high.rename("offset_high"),
            ],
            compat="override",
        )
        .drop_vars("quantile")
        .odc.assign_crs(crs=gbox.crs)
    )

    # Optionally resample into the original pixel grid of `data`
    if resample:
        stats_ds = stats_ds.odc.reproject(how=gbox, resample_method="bilinear")

    return stats_ds
