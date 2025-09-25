"""Tools for analysing local tide dynamics and satellite biases.

This module provides functions to assess how well satellite EO data
captures real-world tides, and reveals potential tide biases in
satellite EO data coverage.
"""

# Used to postpone evaluation of type annotations
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyTMD.arguments import aliasing_period, frequency

from .eo import _pixel_tides_resample, _resample_chunks, _standardise_inputs, pixel_tides, tag_tides

# Only import if running type checking
if TYPE_CHECKING:
    import os

    from odc.geo.geobox import GeoBox

    from .utils import DatetimeLike


# Satellite revisit times (assuming at equator)
REVISIT_DICT = {
    # SWOT (non-sun-synchronous, KaRIn swath altimetry)
    "swot": 0.99349,
    # Landsat series (sun-synchronous, optical)
    "landsat": 8,  # Combined revisit (e.g. two satallites)
    "landsat-5": 16,
    "landsat-7": 16,
    "landsat-8": 16,
    "landsat-9": 16,
    # Sentinel-2 (sun-synchronous, optical)
    "sentinel-2": 5,  # Combined revisit (e.g. two satallites)
    "sentinel-2a": 10,
    "sentinel-2b": 10,
    "sentinel-2c": 10,
    "sentinel-2d": 10,
    # Sentinel-1 (sun-synchronous, C-band SAR)
    "sentinel-1": 6,  # Combined revisit (e.g. two satallites)
    "sentinel-1a": 12,
    "sentinel-1b": 12,
    "sentinel-1c": 12,
    "sentinel-1d": 12,
    # Sentinel-3 OLCI (sun-synchronous, Ocean and Land Color Instrument)
    "sentinel-3a-olci": 2,
    "sentinel-3b-olci": 2,
    "sentinel-3c-olci": 2,
    "sentinel-3d-olci": 2,
    # Sentinel-3 SLSTR (sun-synchronous, Sea and Land Surface Temperature Radiometer)
    "sentinel-3a-slstr": 1.7,
    "sentinel-3b-slstr": 1.7,
    "sentinel-3c-slstr": 1.7,
    "sentinel-3d-slstr": 1.7,
    # Sentinel-3 SRAL (sun-synchronous, SAR Radar Altimeter)
    "sentinel-3a-sral": 27,
    "sentinel-3b-sral": 27,
    "sentinel-3c-sral": 27,
    "sentinel-3d-sral": 27,
    # NISAR (sun-synchronous, L- and S-band SAR)
    "nisar": 12,
}

# List of major constituents from pyTMD table:
# https://pytmd.readthedocs.io/en/latest/background/Constituent-Table.html
# TODO: read these from code once available in pyTMD package
MAJOR_CONSTITUENTS = {
    "sa": "Solar annual",
    "ssa": "Solar semiannual",
    "mm": "Lunar monthly",
    "msf": "Lunisolar synodic fortnightly",
    "mf": "Lunar declinational fortnightly",
    "mt": "Termensual",
    "2q1": "Smaller elliptical diurnal",
    "sigma1": "Lunar variational diurnal",
    "q1": "Larger lunar elliptical diurnal",
    "rho1": "Larger lunar evectional diurnal",
    "o1": "Lunar diurnal",
    "tau1": "",
    "m1": "Smaller lunar elliptical diurnal",
    "chi1": "Smaller evectional diurnal",
    "pi1": "Solar elliptical diurnal",
    "p1": "Principal solar diurnal",
    "s1": "Raditional solar diurnal",
    "k1": "Principal declinational diurnal",
    "psi1": "Smaller solar elliptical diurnal",
    "phi1": "Second-order solar diurnal",
    "theta1": "Evectional diurnal",
    "j1": "Smaller lunar elliptical diurnal",
    "oo1": "Second-order lunar diurnal",
    "eps2": "",
    "2n2": "Second-order lunar elliptical semidiurnal",
    "mu2": "Lunar variational",
    "n2": "Larger lunar elliptical semidiurnal",
    "nu2": "Larger lunar evectional semidiurnal",
    "m2": "Principal lunar semidiurnal",
    "lambda2": "Smaller lunar evectional",
    "l2": "Smaller lunar elliptical semidiurnal",
    "t2": "Larger solar elliptical semidiurnal",
    "s2": "Principal solar semidiurnal",
    "r2": "Smaller solar elliptical semidiurnal",
    "k2": "Lunisolar declinational semidiurnal",
    "eta2": "",
    "m3": "Principal lunar terdiurnal",
}


def _tide_statistics(
    obs_tides: xr.DataArray,
    all_tides: xr.DataArray,
    min_max_q: tuple = (0.0, 1.0),
    dim: str = "time",
) -> xr.Dataset:
    # Calculate means of observed and modelled tides
    mot = obs_tides.mean(dim=dim)
    mat = all_tides.mean(dim=dim)

    # Identify highest and lowest observed tides
    obs_tides_q = obs_tides.quantile(q=min_max_q, dim=dim).astype("float32")
    lot = obs_tides_q.isel(quantile=0, drop=True)
    hot = obs_tides_q.isel(quantile=-1, drop=True)

    # Identify highest and lowest modelled tides
    all_tides_q = all_tides.quantile(q=min_max_q, dim=dim).astype("float32")
    lat = all_tides_q.isel(quantile=0, drop=True)
    hat = all_tides_q.isel(quantile=-1, drop=True)

    # Calculate tidal range
    otr = hot - lot
    tr = hat - lat

    # Calculate Bishop-Taylor et al. 2018 tidal metrics
    spread = otr / tr
    offset_low_m = lot - lat
    offset_high_m = hat - hot
    offset_low = offset_low_m / tr
    offset_high = offset_high_m / tr

    # Combine into a single dataset
    return xr.merge(
        [
            mot.rename("mot"),
            mat.rename("mat"),
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


def _stats_plain_english(
    mot,
    mat,
    hot,
    hat,
    lot,
    lat,
    otr,
    tr,
    spread,
    offset_low,
    offset_high,
) -> None:
    # Plain text descriptors
    mean_diff = "higher" if mot > mat else "lower"
    mean_diff_icon = "⬆️" if mot > mat else "⬇️"
    spread_icon = "🟢" if spread >= 0.9 else "🟡" if 0.7 < spread <= 0.9 else "🔴"
    low_tide_icon = "🟢" if offset_low <= 0.1 else "🟡" if 0.1 <= offset_low < 0.2 else "🔴"
    high_tide_icon = "🟢" if offset_high <= 0.1 else "🟡" if 0.1 <= offset_high < 0.2 else "🔴"

    # Print summary
    print(f"\n\n🌊 Modelled astronomical tide range: {tr:.2f} m ({lat:.2f} to {hat:.2f} m).")
    print(f"🛰️ Observed tide range: {otr:.2f} m ({lot:.2f} to {hot:.2f} m).\n")
    print(f"{spread_icon} {spread:.0%} of the modelled astronomical tide range was observed at this location.")
    print(
        f"{high_tide_icon} The highest {offset_high:.0%} ({offset_high * tr:.2f} m) of the tide range was never observed.",
    )
    print(
        f"{low_tide_icon} The lowest {offset_low:.0%} ({offset_low * tr:.2f} m) of the tide range was never observed.\n",
    )
    print(f"🌊 Mean modelled astronomical tide height: {mat:.2f} m.")
    print(f"🛰️ Mean observed tide height: {mot:.2f} m.")
    print(
        f"{mean_diff_icon} The mean observed tide height was {mot - mat:.2f} m {mean_diff} than the mean modelled astronomical tide height.",
    )


def _stats_figure(
    all_tides_da,
    obs_tides_da,
    hot,
    hat,
    lot,
    lat,
    spread,
    offset_low,
    offset_high,
    plot_var,
    point_col=None,
):
    """Plot tide bias statistics as a figure comparing satellite observations and all modelled tides."""
    # Create plot and add all modelled tides
    fig, ax = plt.subplots(figsize=(10, 6))
    all_tides_da.plot(ax=ax, alpha=0.4, label="Modelled tides")

    # Loop through custom variable values if provided
    if plot_var is not None:
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

        # Sort values to allow correct grouping
        obs_tides_da = obs_tides_da.sortby("time")
        plot_var = plot_var.sortby("time")

        # Iterate and plot each group
        for i, (label, group) in enumerate(obs_tides_da.groupby(plot_var)):
            group.plot.line(
                ax=ax,
                linewidth=0.0,
                color=point_col,
                marker=markers[i % len(markers)],
                label=label,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

    # Otherwise, plot all data at once
    else:
        obs_tides_da.plot.line(
            ax=ax,
            marker="o",
            linewidth=0.0,
            color="black" if point_col is None else point_col,
            markeredgecolor="black",
            markeredgewidth=0.6,
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

    # Add horizontal lines for spread/offsets
    ax.axhline(lot, color="black", linestyle=":", linewidth=1)
    ax.axhline(hot, color="black", linestyle=":", linewidth=1)
    ax.axhline(lat, color="black", linestyle=":", linewidth=1)
    ax.axhline(hat, color="black", linestyle=":", linewidth=1)

    # Add text annotations for spread/offsets
    ax.annotate(
        f"    High tide\n    offset ({offset_high:.0%})",
        xy=(all_tides_da.time.max(), np.mean([hat, hot])),
        va="center",
    )
    ax.annotate(
        f"    Spread\n    ({spread:.0%})",
        xy=(all_tides_da.time.max(), np.mean([lot, hot])),
        va="center",
    )
    ax.annotate(
        f"    Low tide\n    offset ({offset_low:.0%})",
        xy=(all_tides_da.time.max(), np.mean([lat, lot])),
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
    plot_var: str | None = None,
    point_col: str | None = None,
    modelled_freq: str = "3h",
    min_max_q: tuple = (0.0, 1.0),
    round_stats: int = 3,
    **tag_tides_kwargs,
) -> pd.Series:
    """Generate tide statistics and satellite tide bias metrics for every dataset timestep.

    Takes a multi-dimensional dataset and generate tide statistics
    and satellite-observed tide bias metrics, calculated based on
    every timestep in the satellite data and the geographic centroid
    of the imagery.

    By comparing the subset of tides observed by satellites
    against the full astronomical tidal range, we can evaluate
    whether the tides observed by satellites are biased
    (e.g. fail to observe either the highest or lowest tides) due
    to tide aliasing interactions with sun-synchronous satellite
    overpasses.

    For more information about the tidal statistics computed by this
    function, refer to Figure 8 in Bishop-Taylor et al. 2018:
    https://www.sciencedirect.com/science/article/pii/S0272771418308783#fig8

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
    model : str or list of str, optional
        The tide model (or list of models) to use to model tides.
        If a list is provided, the resulting statistics will be
        returned as a `pandas.Dataframe`; otherwise a `pandas.Series`
        will be returned. Defaults to "EOT20"; specify "all" to use
        all models available in `directory`. For a full list of
        available and supported models, run
        `from eo_tides.utils import list_models; list_models()`.
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
        version of the tidal statistics to the screen. Defaults to True;
        only supported when a single tide model is passed to `model`.
    plot : bool, optional
        An optional boolean indicating whether to plot how satellite-
        observed tide heights compare against the full tidal range.
        Defaults to True; only supported when a single tide model is
        passed to `model`.
    plot_var : str, optional
        Optional name of a coordinate, dimension or variable in the array
        that will be used to plot observations with unique symbols.
        Defaults to None, which will plot all observations as circles.
    point_col : str, optional
        Colour used to plot points on the graph. Defaults to None which
        will automatically select colours.
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
    round_stats : int, optional
        The number of decimal places used to round the output statistics.
        Defaults to 3.
    **tag_tides_kwargs :
        Optional parameters passed to the `eo_tides.eo.tag_tides`
        function that is used to model tides for each observed and
        modelled timestep.

    Returns
    -------
    stats_df : pandas.Series or pandas.Dataframe
        A pandas object containing the following statistics:

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

    """
    # Standardise data inputs, time and models
    gbox, obs_times = _standardise_inputs(data, time)

    # Generate range of times covering entire period of satellite record
    assert obs_times is not None  # noqa: S101
    all_times = pd.date_range(
        start=obs_times.min().item(),
        end=obs_times.max().item(),
        freq=modelled_freq,
    )

    # If custom tide modelling locations are not provided, use the
    # dataset centroid
    if not tidepost_lat or not tidepost_lon:
        tidepost_lon, tidepost_lat = gbox.geographic_extent.centroid.coords[0]

    # Model tides for observed timesteps
    obs_tides_da = tag_tides(
        gbox,
        time=obs_times,
        model=model,
        directory=directory,
        tidepost_lat=tidepost_lat,
        tidepost_lon=tidepost_lon,
        **tag_tides_kwargs,
    )

    # Model tides for all modelled timesteps
    all_tides_da = tag_tides(
        gbox,
        time=all_times,
        model=model,
        directory=directory,
        tidepost_lat=tidepost_lat,
        tidepost_lon=tidepost_lon,
        **tag_tides_kwargs,
    )

    # Calculate statistics
    # # (cast ensures typing knows these are always DataArrays)
    stats_ds = _tide_statistics(
        cast("xr.DataArray", obs_tides_da),
        cast("xr.DataArray", all_tides_da),
        min_max_q=min_max_q,
    )

    # Convert to pandas and add tide post coordinates
    stats_df = stats_ds.to_pandas().astype("float32")
    stats_df["x"] = tidepost_lon
    stats_df["y"] = tidepost_lat

    # Convert coordinates to index if dataframe
    if isinstance(stats_df, pd.DataFrame):
        stats_df = stats_df.set_index(["x", "y"], append=True)

    # If a series, print and plot summaries
    else:
        if plain_english:
            _stats_plain_english(
                mot=stats_df.mot,
                mat=stats_df.mat,
                hot=stats_df.hot,
                hat=stats_df.hat,
                lot=stats_df.lot,
                lat=stats_df.lat,
                otr=stats_df.otr,
                tr=stats_df.tr,
                spread=stats_df.spread,
                offset_low=stats_df.offset_low,
                offset_high=stats_df.offset_high,
            )

        if plot:
            _stats_figure(
                all_tides_da=all_tides_da,
                obs_tides_da=obs_tides_da,
                hot=stats_df.hot,
                hat=stats_df.hat,
                lot=stats_df.lot,
                lat=stats_df.lat,
                spread=stats_df.spread,
                offset_low=stats_df.offset_low,
                offset_high=stats_df.offset_high,
                plot_var=data[plot_var] if plot_var else None,
                point_col=point_col,
            )

    # Return in Pandas format
    return stats_df.round(round_stats)


def pixel_stats(
    data: xr.Dataset | xr.DataArray | GeoBox,
    time: DatetimeLike | None = None,
    model: str | list[str] = "EOT20",
    directory: str | os.PathLike | None = None,
    resample: bool = True,
    modelled_freq: str = "3h",
    min_max_q: tuple[float, float] = (0.0, 1.0),
    resample_method: str = "bilinear",
    dask_chunks: tuple[float, float] | None = None,
    dask_compute: bool = True,
    extrapolate: bool = True,
    cutoff: float = 10,
    **pixel_tides_kwargs,
) -> xr.Dataset:
    """Generate tide statistics and satellite tide bias metrics for every dataset pixel.

    Takes a multi-dimensional dataset and generate pixel-level
    tide statistics and satellite-observed tide bias metrics,
    calculated based on every timestep in the satellite data and
    modelled into the spatial extent of the imagery.

    By comparing the subset of tides observed by satellites
    against the full astronomical tidal range, we can evaluate
    whether the tides observed by satellites are biased
    (e.g. fail to observe either the highest or lowest tides)
    due to tide aliasing interactions with sun-synchronous satellite
    overpasses.

    Compared to `tide_stats`, this function models tide metrics
    spatially to produce a two-dimensional output for each statistic.

    For more information about the tidal statistics computed by this
    function, refer to Figure 8 in Bishop-Taylor et al. 2018:
    <https://www.sciencedirect.com/science/article/pii/S0272771418308783#fig8>

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or odc.geo.geobox.GeoBox
        A multi-dimensional dataset or GeoBox pixel grid that will
        be used to calculate spatial tide statistics. If `data`
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
        The tide model (or list of models) to use to model tides.
        If a list is provided, a new "tide_model" dimension will be
        added to the `xarray.Dataset` output. Defaults to "EOT20";
        specify "all" to use all models available in `directory`.
        For a full list of available and supported models, run
        `from eo_tides.utils import list_models; list_models()`.
    directory : str, optional
        The directory containing tide model data files. If no path is
        provided, this will default to the environment variable
        `EO_TIDES_TIDE_MODELS` if set, or raise an error if not.
        Tide modelling files should be stored in sub-folders for each
        model that match the structure required by `pyTMD`
        (<https://geoscienceaustralia.github.io/eo-tides/setup/>).
    resample : bool, optional
        Whether to resample tide statistics back into `data`'s original
        higher resolution grid. Set this to `False` if you want to return
        lower-resolution tide statistics (which can be useful for
        assessing tide biases across large spatial extents).
    modelled_freq : str, optional
        An optional string giving the frequency at which to model tides
        when computing the full modelled tidal range. Defaults to '3h',
        which computes a tide height for every three hours across the
        temporal extent of `data`.
    min_max_q : tuple, optional
        Quantiles used to calculate max and min observed and modelled
        astronomical tides. By default `(0.0, 1.0)` which is equivalent
        to minimum and maximum; for a softer threshold that is more
        robust to outliers use e.g. `(0.1, 0.9)`.
    resample_method : str, optional
        If resampling is requested (see `resample` above), use this
        resampling method when resampling from low resolution to high
        resolution pixels. Defaults to "bilinear"; valid options include
        "nearest", "cubic", "min", "max", "average" etc.
    dask_chunks : tuple of float, optional
        Can be used to configure custom Dask chunking for the final
        resampling step. By default, chunks will be automatically set
        to match y/x chunks from `data` if they exist; otherwise chunks
        will be chosen to cover the entire y/x extent of the dataset.
        For custom chunks, provide a tuple in the form `(y, x)`, e.g.
        `(2048, 2048)`.
    dask_compute : bool, optional
        Whether to compute results of the resampling step using Dask.
        If False, `stats_ds` will be returned as a Dask-enabled array.
    extrapolate : bool, optional
        Whether to extrapolate tides into x and y coordinates outside of
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

    """
    # Standardise data inputs, time and models
    gbox, obs_times = _standardise_inputs(data, time)
    dask_chunks = _resample_chunks(data, dask_chunks)
    model = [model] if isinstance(model, str) else model

    # Generate range of times covering entire period of satellite record
    assert obs_times is not None  # noqa: S101
    all_times = pd.date_range(
        start=obs_times.min().item(),
        end=obs_times.max().item(),
        freq=modelled_freq,
    )

    # Model tides for observed timesteps
    obs_tides_da = pixel_tides(
        gbox,
        time=obs_times,
        model=model,
        directory=directory,
        resample=False,
        extrapolate=extrapolate,
        cutoff=cutoff,
        **pixel_tides_kwargs,
    )

    # Model tides for all modelled timesteps
    all_tides_da = pixel_tides(
        gbox,
        time=all_times,
        model=model,
        directory=directory,
        resample=False,
        extrapolate=extrapolate,
        cutoff=cutoff,
        **pixel_tides_kwargs,
    )

    # Calculate statistics
    stats_lowres = _tide_statistics(obs_tides_da, all_tides_da, min_max_q=min_max_q)

    # Assign CRS and geobox to allow reprojection
    stats_lowres = stats_lowres.odc.assign_crs(crs=gbox.crs)

    # Reproject statistics into original high resolution grid
    if resample:
        print("Reprojecting statistics into original resolution")
        return _pixel_tides_resample(
            stats_lowres,
            gbox,
            resample_method,
            dask_chunks,
            dask_compute,
            None,
        )

    print("Returning low resolution statistics array")
    return stats_lowres


def tide_aliasing(
    satellites: list[str] | dict[str, float],
    constituents: list[str] | None = None,
    units: str = "days",
    max_inf: float | None = None,
    style: bool = True,
):
    """Calculate aliasing periods for tidal constituents given satellite revisit intervals.

    This function uses `pyTMD.arguments.aliasing_period` to calculate the
    aliasing periods between satellite overpass periods and the natural
    cycles of tidal constituents. The aliasing period describes how long
    it would take for a satellite to sample the entire tidal cycle for
    each constituent, based on the satellite's observation frequency.

    Short aliasing periods mean the satellite will observe the full range
    of tidal variation relatively quickly, reducing the risk of tide-related
    bias. Long aliasing periods indicate that it will take much longer to
    sample all tidal conditions, increasing the risk that satellite analyses
    may misrepresent tidal dynamics.

    Revisit periods are approximate and based on nominal repeat cycles at the equator.
    Actual observation frequency may vary due to latitude, cloud cover, sensor
    availability, and acquisition planning. Custom revisit intervals can be
    defined by passing a dictionary to `satellites`, e.g. `satellites={"custom-sat": 5}`.

    For more information, refer to https://pytmd.readthedocs.io/en/latest/api_reference/arguments.html#pyTMD.arguments.aliasing_period

    Parameters
    ----------
    satellites : list of str or dict
        List of satellite names to analyse, or a custom dictionary with
        satellite names as keys and revisit frequency in days as values.
        Supported satellites include:

        - Landsat (optical):
            - Two satellites combined: "landsat"
            - Individual: "landsat-5", "landsat-7", "landsat-8", "landsat-9"
        - Sentinel-2 (optical):
            - Two satellites combined: "sentinel-2"
            - Individual: "sentinel-2a", "sentinel-2b", "sentinel-2c"
        - Sentinel-1 (C-band SAR):
            - Two satellites combined: "sentinel-1"
            - Individual: "sentinel-1a", "sentinel-1b", "sentinel-1c"
        - Sentinel-3:
            - OLCI (optical): "sentinel-3a-olci", "sentinel-3b-olci", "sentinel-3c-olci"
            - SLSTR (thermal): "sentinel-3a-slstr", "sentinel-3b-slstr, "sentinel-3c-slstr"
            - SRAL (altimetry): "sentinel-3a-sral", "sentinel-3b-sral, "sentinel-3c-sral"
        - SWOT (KaRIn swath altimetry): "swot"
        - NISAR (L- and S-band SAR): "nisar"
    constituents : list of str or None, optional
        List of tidal constituents to include. If None, use a list of major
        constituents. Constituent names should be lowercase (e.g., "m2", "k1").
    units : str, optional
        Output time units for the aliasing periods. Must be one of:
        "years", "days", "hours", or "minutes". Default is "days".
    max_inf : float, optional
        Maximum aliasing period to display in seconds. Values exceeding
        this threshold are replaced with `np.inf`. Defaults to equivalent of 10
        years if no value is provided.
    style : bool, optional
        If True, returns a styled `pandas.DataFrame`. If False, returns a raw
        DataFrame. Default is True.

    Returns
    -------
    pandas.DataFrame
        A table showing aliasing periods for each tidal constituent across the given satellites.
        The result is styled with a color gradient if `style=True`, or returned as a plain DataFrame.

    Examples
    --------
    >>> eo_tide_aliasing(["sentinel-2", "landsat-8"])
    >>> eo_tide_aliasing(["swot"], constituents=["m2", "k1"], units="hours", style=False)
    >>> eo_tide_aliasing({"custom-sat": 5})

    """
    # If satellites is a dict
    if isinstance(satellites, dict):
        revisit_dict = satellites | REVISIT_DICT
        satellites = list(satellites.keys())
    else:
        revisit_dict = REVISIT_DICT

    # Validate satellite names
    invalid_sats = set(satellites) - set(revisit_dict)
    if invalid_sats:
        valid = ", ".join(sorted(revisit_dict))
        error_msg = f"Unknown satellite(s): {', '.join(invalid_sats)}. Must be one of: {valid}"
        raise ValueError(error_msg)

    # Time unit factors
    unit_factors = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400,
        "years": 31556952,
    }

    # Use default list of constituents if none provided
    if constituents is None:
        constituents = list(MAJOR_CONSTITUENTS.keys())

    # Extract frequency in radians per second for each constituent,
    # and convert to period in seconds
    omega = np.array([frequency(c)[0] for c in constituents])
    period = 2 * np.pi / omega

    # Compute aliasing period for each satellite
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in divide",
        )
        aliasing_periods = {}
        for sat in satellites:
            revisit = revisit_dict[sat]
            print(f"Using {revisit} day revisit for {sat}")
            aliasing_periods[("aliasing_period", sat)] = aliasing_period(
                constituents,
                unit_factors["days"] * revisit,
            )

    # Combine into a dataframe
    alias_df = pd.DataFrame(
        index=pd.Index(constituents, name="constituents"),
        data={
            ("period", ""): period,
            **aliasing_periods,
        },
    )

    # Raise error if unit is not supported
    if units not in unit_factors:
        error_msg = f"Unit not supported: {units}; must be one of 'years', 'days', 'hours', 'minutes'"
        raise ValueError(error_msg)

    # Set max value to infinity. If no max is provided, use 10 years
    if max_inf is None:
        max_inf = unit_factors["years"] * 10
    alias_df[alias_df > max_inf] = np.inf

    # Rescale to desired output time units
    precision = 3 if units != "years" else 4
    alias_df = (alias_df / unit_factors[units]).round(precision)

    # Add constituent name column
    alias_df.insert(0, column="name", value=[MAJOR_CONSTITUENTS.get(c) for c in constituents])

    # Style and return
    df_subset = alias_df.loc[:, "aliasing_period"]
    max_col = np.nanquantile(df_subset[np.isfinite(df_subset)].values, 0.9)

    if style:
        return alias_df.style.background_gradient(
            axis=None,
            vmin=0,
            vmax=max_col * 1.5,
            cmap="YlOrRd",
        ).format(precision=precision)
    return alias_df
