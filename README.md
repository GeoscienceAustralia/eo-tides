# `eo-tides:` Tide modelling tools for large-scale satellite earth observation analysis

[![Release](https://img.shields.io/github/v/release/GeoscienceAustralia/eo-tides)](https://pypi.org/project/eo-tides/)
[![Build status](https://img.shields.io/github/actions/workflow/status/GeoscienceAustralia/eo-tides/main.yml?branch=main)](https://github.com/GeoscienceAustralia/eo-tides/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/GeoscienceAustralia/eo-tides/branch/main/graph/badge.svg)](https://codecov.io/gh/GeoscienceAustralia/eo-tides)
[![Commit activity](https://img.shields.io/github/commit-activity/m/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/commit-activity/m/GeoscienceAustralia/eo-tides)
[![License](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)

- **Github repository**: <https://github.com/GeoscienceAustralia/eo-tides/>
- **Documentation** <https://GeoscienceAustralia.github.io/eo-tides/>

> [!CAUTION]
> This package is a work in progress, and not currently ready for operational use.

The `eo-tides` package provides tools for analysing coastal and ocean satellite earth observation data using information about ocean tides.

`eo-tides` combines advanced tide modelling functionality from the [`pyTMD`](https://pytmd.readthedocs.io/en/latest/) package and integrates it with `pandas`, `xarray` and `odc-geo` to provide a powerful set of tools for integrating satellite imagery with tide data.

Some key functionality includes the ability to:

- Model tides from multiple global ocean tide models (e.g. FES2022, FES2014, TPXO9, EOT20 and many more) in parallel, and return tide heights in standardised `pandas.DataFrame` format for further analysis
- "Tag" satellite data timeseries with tide data based on the exact moment of each satellite acquisition
- Model tides for every individual satellite pixel, producing three-dimensional "tide height" `xarray`-format datacubes that can be combined with satellite data
- Calculate statistics describing local tide dynamics, as well as biases caused by interactions between tidal processes and satellite orbits
- Validate modelled tides using measured sea levels from coastal tide gauges

These tools can be applied directly to petabytes of freely available satellite data (e.g. from Digital Earth Australia or Microsoft Planetary Computer) loaded via Open Data Cube's `odc-stac` or `datacube` packages, supporting coastal and ocean earth observation analysis for any time period or location globally.
