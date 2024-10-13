# Used to postpone evaluation of type annotations
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import odc.geo.xr
import pandas as pd
from scipy import stats

# Only import if running type checking
if TYPE_CHECKING:
    import xarray as xr

from .eo import tag_tides
from .model import model_tides


def tide_stats(
    ds: xr.Dataset,
    model: str = "EOT20",
    directory: str | os.PathLike | None = None,
    tidepost_lat: float | None = None,
    tidepost_lon: float | None = None,
    plain_english: bool = True,
    plot: bool = True,
    plot_col: str | None = None,
    modelled_freq: str = "2h",
    linear_reg: bool = False,
    min_max_q: tuple = (0.0, 1.0),
    round_stats: int = 3,
    **model_tides_kwargs,
) -> pd.Series:
    """
    Takes a multi-dimensional dataset and generate statistics
    about the data's astronomical and satellite-observed tide
    conditions.

    By comparing the subset of tides observed by satellites
    against the full astronomical tidal range, we can evaluate
    whether the tides observed by satellites are biased
    (e.g. fail to observe either the highest or lowest tides).

    For more information about the tidal statistics computed by this
    function, refer to Figure 8 in Bishop-Taylor et al. 2018:
    <https://www.sciencedirect.com/science/article/pii/S0272771418308783#fig8>

    Parameters
    ----------
    ds : xarray.Dataset
        A multi-dimensional dataset (e.g. "x", "y", "time") to
        use to calculate tide statistics. This dataset must contain
        a "time" dimension.
    model : string, optional
        The tide model to use to model tides. Defaults to "EOT20";
        for a full list of available/supported models, run
        `eo_tides.model.list_models`.
    directory : string, optional
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
        when computing the full modelled tidal range. Defaults to '2h',
        which computes a tide height for every two hours across the
        temporal extent of `ds`.
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
    A `pandas.Series` containing the following statistics:

        - `tidepost_lat`: latitude used for modelling tide heights
        - `tidepost_lon`: longitude used for modelling tide heights
        - `observed_min_m`: minimum tide height observed by the satellite
        - `all_min_m`: minimum tide height from all available tides
        - `observed_max_m`: maximum tide height observed by the satellite
        - `all_max_m`: maximum tide height from all available tides
        - `observed_range_m`: tidal range observed by the satellite
        - `all_range_m`: full astronomical tidal range based on all available tides
        - `spread_m`: proportion of the full astronomical tidal range observed by the satellite (see Bishop-Taylor et al. 2018)
        - `low_tide_offset`: proportion of the lowest tides never observed by the satellite (see Bishop-Taylor et al. 2018)
        - `high_tide_offset`: proportion of the highest tides never observed by the satellite (see Bishop-Taylor et al. 2018)

    If `linear_reg = True`, the output will also contain:

        - `observed_slope`: slope of any relationship between observed tide heights and time
        - `observed_pval`: significance/p-value of any relationship between observed tide heights and time

    """
    # Verify that only one tide model is provided
    if isinstance(model, list):
        raise Exception("Only single tide models are supported by `tide_stats`.")

    # If custom tide modelling locations are not provided, use the
    # dataset centroid
    if not tidepost_lat or not tidepost_lon:
        tidepost_lon, tidepost_lat = ds.odc.geobox.geographic_extent.centroid.coords[0]

    # Model tides for each observation in the supplied xarray object
    ds_tides = tag_tides(
        ds,
        model=model,
        directory=directory,
        tidepost_lat=tidepost_lat,  # type: ignore
        tidepost_lon=tidepost_lon,  # type: ignore
        return_tideposts=True,
        **model_tides_kwargs,
    )
    ds_tides = ds_tides.sortby("time")

    # Drop spatial ref for nicer plotting
    ds_tides = ds_tides.drop_vars("spatial_ref")

    # Generate range of times covering entire period of satellite record
    all_timerange = pd.date_range(
        start=ds_tides.time.min().item(),
        end=ds_tides.time.max().item(),
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
    obs_mean = ds_tides.tide_height.mean().item()
    all_mean = all_tides_df.tide_height.mean()
    obs_min, obs_max = ds_tides.tide_height.quantile(min_max_q).values
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
    all_times = all_tides_df.index.get_level_values("time")
    all_x = all_times.year + ((all_times.dayofyear - 1) / 365) + ((all_times.hour) / 24)
    time_period = all_x.max() - all_x.min()

    # Extract x (time in decimal years) and y (distance) values
    obs_x = ds_tides.time.dt.year + ((ds_tides.time.dt.dayofyear - 1) / 365) + ((ds_tides.time.dt.hour) / 24)
    obs_y = ds_tides.tide_height.values.astype(np.float32)

    # Compute linear regression
    obs_linreg = stats.linregress(x=obs_x, y=obs_y)

    # return obs_linreg

    if plain_english:
        print(f"\n\n🌊 Modelled astronomical tide range: {all_range:.2f} metres.")
        print(f"🛰️ Observed tide range: {obs_range:.2f} metres.\n")
        print(f"    {spread_icon} {spread:.0%} of the modelled astronomical tide range was observed at this location.")
        print(
            f"    {high_tide_icon} The highest {high_tide_offset:.0%} ({high_tide_offset_m:.2f} metres) of the tide range was never observed."
        )
        print(
            f"    {low_tide_icon} The lowest {low_tide_offset:.0%} ({low_tide_offset_m:.2f} metres) of the tide range was never observed.\n"
        )
        print(f"🌊 Mean modelled astronomical tide height: {all_mean:.2f} metres.")
        print(f"🛰️ Mean observed tide height: {obs_mean:.2f} metres.\n")
        print(
            f"    {mean_diff_icon} The mean observed tide height was {obs_mean - all_mean:.2f} metres {mean_diff} than the mean modelled astronomical tide height."
        )

        if linear_reg:
            if obs_linreg.pvalue > 0.01:
                print("    ➖ Observed tides showed no significant trends over time.")
            else:
                obs_slope_desc = "decreasing" if obs_linreg.slope < 0 else "increasing"
                print(
                    f"    ⚠️ Observed tides showed a significant {obs_slope_desc} trend over time (p={obs_linreg.pvalue:.3f}, {obs_linreg.slope:.2f} metres per year)"
                )

    if plot:
        # Create plot and add all time and observed tide data
        fig, ax = plt.subplots(figsize=(10, 6))
        all_tides_df.reset_index(["x", "y"]).tide_height.plot(ax=ax, alpha=0.4, label="Modelled tides")

        # Look through custom column values if provided
        if plot_col is not None:
            # Create a list of marker styles
            markers = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x", "d", "|", "_"]
            for i, value in enumerate(np.unique(ds_tides[plot_col])):
                ds_tides.where(ds_tides[plot_col] == value, drop=True).tide_height.plot.line(
                    ax=ax,
                    linewidth=0.0,
                    color="black",
                    marker=markers[i % len(markers)],
                    markersize=4,
                    label=value,
                )
        # Otherwise, plot all data at once
        else:
            ds_tides.tide_height.plot.line(
                ax=ax, marker="o", linewidth=0.0, color="black", markersize=3.5, label="Satellite observations"
            )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=20, borderaxespad=0, frameon=False)

        ax.plot(
            ds_tides.time.isel(time=[0, -1]),
            obs_linreg.intercept + obs_linreg.slope * obs_x[[0, -1]],
            "r",
            label="fitted line",
        )

        # Add horizontal lines for spread/offsets
        ax.axhline(obs_min, color="black", linestyle=":", linewidth=1)
        ax.axhline(obs_max, color="black", linestyle=":", linewidth=1)
        ax.axhline(all_min, color="black", linestyle=":", linewidth=1)
        ax.axhline(all_max, color="black", linestyle=":", linewidth=1)

        # Add text annotations for spread/offsets
        ax.annotate(
            f"    High tide\n    offset ({high_tide_offset:.0%})",
            xy=(all_timerange.max(), np.mean([all_max, obs_max])),
            va="center",
        )
        ax.annotate(
            f"    Spread\n    ({spread:.0%})",
            xy=(all_timerange.max(), np.mean([obs_min, obs_max])),
            va="center",
        )
        ax.annotate(
            f"    Low tide\n    offset ({low_tide_offset:.0%})",
            xy=(all_timerange.max(), np.mean([all_min, obs_min])),
        )

        # Remove top right axes and add labels
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("Tide height (m)")
        ax.set_xlabel("")
        ax.margins(x=0.015)

    # Export pandas.Series containing tidal stats
    output_stats = {
        "tidepost_lat": tidepost_lat,
        "tidepost_lon": tidepost_lon,
        "observed_mean_m": obs_mean,
        "all_mean_m": all_mean,
        "observed_min_m": obs_min,
        "all_min_m": all_min,
        "observed_max_m": obs_max,
        "all_max_m": all_max,
        "observed_range_m": obs_range,
        "all_range_m": all_range,
        "spread": spread,
        "low_tide_offset": low_tide_offset,
        "high_tide_offset": high_tide_offset,
    }

    if linear_reg:
        output_stats.update({
            "observed_slope": obs_linreg.slope,
            "observed_pval": obs_linreg.pvalue,
        })

    return pd.Series(output_stats).round(round_stats)
