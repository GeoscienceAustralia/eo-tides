# eo-tides: Tide modelling tools for large-scale satellite earth observation analysis

[![Release](https://img.shields.io/github/v/release/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/v/release/GeoscienceAustralia/eo-tides)
[![Build status](https://img.shields.io/github/actions/workflow/status/GeoscienceAustralia/eo-tides/main.yml?branch=main)](https://github.com/GeoscienceAustralia/eo-tides/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/GeoscienceAustralia/eo-tides/branch/main/graph/badge.svg)](https://codecov.io/gh/GeoscienceAustralia/eo-tides)
[![Commit activity](https://img.shields.io/github/commit-activity/m/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/commit-activity/m/GeoscienceAustralia/eo-tides)
[![License](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)

- **Github repository**: <https://github.com/GeoscienceAustralia/eo-tides/>
- **Documentation** <https://GeoscienceAustralia.github.io/eo-tides/>

> [!CAUTION]
> This package is a work in progress, and not currently ready for operational use.

The `eo-tides` package contains parallelised tools for modelling ocean tides and combining them with time-series satellite earth observation data. The package builds upon tide modelling functionality from the powerful `pyTMD` package which supports many of the best-performing global tide models (e.g. FES2022, FES2014, TPXO9, EOT20 and many more). Tides can be easily modelled in parallel for multiple models, then combined with satellite data from any location on the planet using Open Data Cube's `odc-stac` or `datacube` packages.
