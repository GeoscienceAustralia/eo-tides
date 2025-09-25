# `eo-tides`: Tide modelling tools for large-scale satellite earth observation analysis

<img align="right" width="200" src="https://github.com/GeoscienceAustralia/eo-tides/blob/main/docs/assets/eo-tides-logo.gif?raw=true" alt="eo-tides logo" style="margin-right: 40px;">

[![Release](https://img.shields.io/github/v/release/GeoscienceAustralia/eo-tides)](https://pypi.org/project/eo-tides/)
[![Build status](https://img.shields.io/github/actions/workflow/status/GeoscienceAustralia/eo-tides/tests.yml?branch=main)](https://github.com/GeoscienceAustralia/eo-tides/actions/workflows/tests.yml?query=branch%3Amain)
[![Python Version from PEP 621 TOML](https://img.shields.io/pypi/pyversions/eo-tides)](https://github.com/GeoscienceAustralia/eo-tides/blob/main/pyproject.toml)
[![codecov](https://codecov.io/gh/GeoscienceAustralia/eo-tides/branch/main/graph/badge.svg)](https://codecov.io/gh/GeoscienceAustralia/eo-tides)
[![License](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)](https://img.shields.io/github/license/GeoscienceAustralia/eo-tides)
[![JOSS paper](https://joss.theoj.org/papers/b5680c39bf831c1159c41a2eb7ec9c5e/status.svg)](https://joss.theoj.org/papers/b5680c39bf831c1159c41a2eb7ec9c5e)

- ⚙️ **Github repository**: <https://github.com/GeoscienceAustralia/eo-tides/>
- 📦 **PyPI**: <https://pypi.org/project/eo-tides/>
- ℹ️ **Documentation**: <https://GeoscienceAustralia.github.io/eo-tides/>
- 📰 **Paper**: <https://doi.org/10.21105/joss.07786>

<br>

`eo-tides` provides powerful parallelized tools for integrating satellite Earth observation data with tide modelling. 🛠️🌊🛰️

`eo-tides` combines advanced tide modelling functionality from the [`pyTMD`](https://pytmd.readthedocs.io/en/latest/) package with [`pandas`](https://pandas.pydata.org/docs/index.html), [`xarray`](https://docs.xarray.dev/en/stable/) and [`odc-geo`](https://odc-geo.readthedocs.io/en/latest/), providing a suite of flexible tools for efficient analysis of coastal and ocean Earth observation data – from regional, continental, to global scale.

These tools can be applied to petabytes of freely available satellite data (e.g. from [Digital Earth Australia](https://knowledge.dea.ga.gov.au/) or [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)) loaded via Open Data Cube's [`odc-stac`](https://odc-stac.readthedocs.io/en/latest/) or [`datacube`](https://opendatacube.readthedocs.io/en/latest/) packages, supporting coastal and ocean earth observation analysis for any time period or location globally.

![eo-tides abstract showing satellite data, tide data array and tide animation](https://github.com/GeoscienceAustralia/eo-tides/blob/main/docs/assets/eo-tides-abstract.gif?raw=true)

## Highlights

- 🌊 Model tide heights and phases (e.g. high, low, ebb, flow) from multiple global ocean tide models in parallel, and return a `pandas.DataFrame` for further analysis
- 🛰️ "Tag" satellite data with tide heights based on the exact moment of image acquisition
- 🌐 Model tides for every individual satellite pixel through time, producing three-dimensional "tide height" `xarray`-format datacubes that can be integrated with satellite data
- 📈 Calculate statistics describing local tide dynamics, as well as biases caused by interactions between tidal processes and satellite orbits
- 🛠️ Validate modelled tides using measured sea levels from coastal tide gauges (e.g. [GESLA Global Extreme Sea Level Analysis](https://gesla.org/))
<!-- - 🎯 Combine multiple tide models into a single locally-optimised "ensemble" model informed by satellite altimetry and satellite-observed patterns of tidal inundation -->

## Supported tide models

`eo-tides` supports [all ocean tide models supported by `pyTMD`](https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#model-database). These include:

- [Empirical Ocean Tide model](https://doi.org/10.5194/essd-13-3869-2021) (EOT20)
- [Finite Element Solution tide models](https://doi.org/10.5194/os-2020-96) (FES2022, FES2014, FES2012)
- [TOPEX/POSEIDON global tide models](https://www.tpxo.net/global) (TPXO10, TPXO9, TPXO8)
- [Global Ocean Tide models](https://doi.org/10.1002/2016RG000546) (GOT5.6, GOT5.5, GOT4.10, GOT4.8, GOT4.7)
- [Hamburg direct data Assimilation Methods for Tides models](https://doi.org/10.1002/2013JC009766) (HAMTIDE11)
- [Technical University of Denmark tide models](https://doi.org/10.11583/DTU.23828874) (DTU23)

For instructions on how to set up these models for use in `eo-tides`, refer to [Setting up tide models](setup.md).

## Installing and setting up `eo-tides`

To get started with `eo-tides`, follow the [Installation](https://geoscienceaustralia.github.io/eo-tides/install/) and [Setting up tide models](https://geoscienceaustralia.github.io/eo-tides/setup/) guides.

## Jupyter Notebooks code examples

Interactive Jupyter Notebook usage examples and more complex coastal EO case studies can be found in the [`docs/notebooks/`](https://github.com/GeoscienceAustralia/eo-tides/tree/main/docs/notebooks) directory, or [rendered in the documentation here](https://geoscienceaustralia.github.io/eo-tides/notebooks/Model_tides/).

## Citing `eo-tides`

To cite `eo-tides` in your work, please use the following [Journal of Open Source Software](https://doi.org/10.21105/joss.07786) citation:

```
Bishop-Taylor, R., Phillips, C., Sagar, S., Newey, V., & Sutterley, T., (2025). eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis. Journal of Open Source Software, 10(109), 7786, https://doi.org/10.21105/joss.07786
```

<details>

<summary>BibTeX</summary>

```
@article{Bishop-Taylor2025,
  doi       = {10.21105/joss.07786},
  url       = {https://doi.org/10.21105/joss.07786},
  year      = {2025},
  publisher = {The Open Journal},
  volume    = {10},
  number    = {109},
  pages     = {7786},
  author    = {Robbi Bishop-Taylor and Claire Phillips and Stephen Sagar and Vanessa Newey and Tyler Sutterley},
  title     = {eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis},
  journal   = {Journal of Open Source Software}
}
```

</details>

In addition, please consider also citing the underlying [`pyTMD` Python package](https://pytmd.readthedocs.io/en/latest/) which powers the tide modelling functionality behind `eo-tides`:

```
Sutterley, T. C., Alley, K., Brunt, K., Howard, S., Padman, L., Siegfried, M. (2017) pyTMD: Python-based tidal prediction software. 10.5281/zenodo.5555395
```

## Contributing

We welcome contributions to `eo-tides`, both through posting issues (e.g. bug reports or feature suggestions), or directly via pull requests (e.g. bug fixes and new features).
Read the [Contributing guide](https://github.com/GeoscienceAustralia/eo-tides/blob/main/CONTRIBUTING.md) for details about how you can get involved.

## Acknowledgements

For a full list of acknowledgements, refer to [Citations and Credits](https://geoscienceaustralia.github.io/eo-tides/credits/).
This repository was initialised using the [`cookiecutter-uv`](https://github.com/fpgmaas/cookiecutter-uv) package.
