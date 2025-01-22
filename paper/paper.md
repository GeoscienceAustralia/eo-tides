---
title: "eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis"
tags:
  - Python
  - Earth observation
  - Tide modelling
  - Remote sensing
  - Coastal
  - Satellite data
authors:
  - name: Robbi Bishop-Taylor
    corresponding: true
    orcid: 0000-0002-1533-2599
    affiliation: 1
  - name: Claire Phillips
    affiliation: 1
    orcid: 0009-0003-9882-9131
  - name: Stephen Sagar
    affiliation: 1
    orcid: 0000-0001-9568-9661
  - name: Vanessa Newey
    affiliation: 1
  - name: Tyler Sutterley
    affiliation: 2
    orcid: 0000-0002-6964-1194
affiliations:
  - name: Geoscience Australia, Australia
    index: 1
    ror: 04ge02x20
  - name: University of Washington Applied Physics Laboratory, United States of America
    index: 2
    ror: 03d17d270
date: 14 January 2025
bibliography: paper.bib
---

# Summary

The `eo-tides` package provides powerful parallelised tools for integrating satellite Earth observation (EO) data with ocean tide modelling. The package provides a flexible Python-based toolkit for attributing modelled tide heights to a time-series of satellite images based on the spatial extent and acquisition time of each satellite observation (\autoref{fig:abstract}).

`eo-tides` leverages advanced tide modelling functionality from the `pyTMD` tide prediction software [@pytmd], combining this capability with EO spatial analysis tools from `odc-geo` [@odcgeo]. This allows tides to be modelled in parallel using over 50 supported tide models, and returned in standardised `pandas` [@reback2020pandas; @mckinney-proc-scipy-2010] and `xarray` [@Hoyer_xarray_N-D_labeled_2017] data formats for EO analysis.

`eo-tides` tools can be applied to petabytes of freely available satellite data loaded from the cloud using Open Data Cube (ODC)'s `odc-stac` or `datacube` packages (e.g. using [Digital Earth Australia](https://knowledge.dea.ga.gov.au/guides/setup/gis/stac/) or [Microsoft Planetary Computer's](https://planetarycomputer.microsoft.com/) STAC SpatioTemporal Asset Catalogues). Additional functionality allows users to assess potential satellite-tide biases and validate modelled tides with external tide gauge data — critical considerations for ensuring the reliability and accuracy of coastal EO workflows. These open-source tools support the efficient, scalable and robust analysis of coastal EO data for any time period or location globally.

![A typical `eo-tides` coastal EO workflow, with tide heights modelled into every pixel in a spatio-temporal stack of satellite data (for example, from ESA's Sentinel-2 or NASA/USGS Landsat), then combined to derive insights into dynamic coastal environments.\label{fig:abstract}](figures/joss_abstract.png)

# Statement of need

Satellite remote sensing offers an unparalleled resource for examining dynamic coastal environments through time or across large regions [@turner2021satellite; @vitousek2023future]. However, the highly variable influence of ocean tides can complicate analyses, making it difficult to separate the influence of changing tides from patterns of true coastal change [@vos2019coastsat]. This is a particularly challenging for large-scale coastal EO analyses, where failing to account for tide dynamics can lead to inaccurate or misleading insights into satellite-observed coastal processes.

Conversely, information about ocean tides can provide unique environmental insights that can significantly enhance the value of EO data. Traditionally, satellite data dimensions include the geographic "where" and temporal "when" of acquisition. Introducing tide height as an additional analysis dimension allows data to be filtered, sorted, and analysed based on tidal dynamics, offering a transformative re-imagining of traditional multi-temporal EO analysis [@sagar2017item]. For instance, satellite data can be analysed to focus on ecologically significant tidal stages (e.g., high tide, low tide, spring or neap tides) or specific tidal processes (e.g., ebb or flow tides; @sent2025time).

This concept has been used to map coastal change at continental-scale [@bishop2021mapping], map intertidal zone extent and elevation [@murray2012continental; @sagar2017item; @bishop2019NIDEM], and creating tidally-constrained coastal image composites [@sagar2018composites]. However, these methods have traditionally relied on bespoke, closed-source, or difficult-to-install tide modeling tools, limiting their reproducibility and portability. To support the next generation of coastal EO workflows, there is a pressing need for efficient open-source tools for combining satellite data with tide modeling. `eo-tides` addresses this need through functionality offered in five main analysis modules (`utils`, `model`, `eo`, `stats`, `validation`) described below.

# Features

## Setting up tide models

The [`eo_tides.utils`](https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.utils) module simplifies the setup of ocean tide models, addressing a common barrier to coastal EO workflows. Tools like `list_models` provide feedback on available and supported models (\autoref{fig:list}), while `clip_models` can improve performance by clipping large model files to smaller regions, significantly reducing processing times for high-resolution models like FES2022. Comprehensive documentation is available to [assist setting up commonly used tide models](https://geoscienceaustralia.github.io/eo-tides/setup/), including downloading and organising model files.

![An example output from `list_tides`, providing a useful summary table that clearly identifies available and supported tide models.\label{fig:list}](figures/joss_fig_list.png)

## Modelling tides

The [`eo_tides.model`](https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.model) module is powered by tide modelling functionality from the `pyTMD` Python package [@pytmd].

`pyTMD` is an open-source tidal prediction software that simplifies the calculation of ocean and earth tides. Tides are frequently decomposed into harmonic constants (or constituents) associated with the relative positions of the sun, moon and Earth. `pyTMD.io` contains routines for reading and spatially interpolating major constituent values from commonly available ocean tide models. `pyTMD.astro` contains routines for computing the positions of celestial bodies for a given time. For ocean tides, `pyTMD` computes the longitudes of the sun (S), moon (H), lunar perigree (P), ascending lunar node (N) and solar perigree (PP). `pyTMD.arguments` combines astronomical coefficients with the "Doodson number" of each constituent, and adjusts the amplitude and phase of each constituent based on their modulations over the 18.6 year nodal period. Finally, `pyTMD.predict` uses results from those underlying functions to predict tidal values at a given location and time.

The `model_tides` function from `eo_tides.model` wraps `pyTMD` functionality to return tide predictions in a standardised `pandas.DataFrame` format, enabling integration with satellite EO data and parallelised processing for improved performance. Parallelisation in `eo-tides` is automatically optimised based on available workers and requested tide models and tide modelling locations. This parallelisation can significantly improve performance, especially for large-scale analyses run on multi-core machines (\autoref{tab:benchmark}). Additional functions like `model_phases` classify tides into high/low/flow/ebb phases, critical for interpreting satellite-observed coastal processes like changing turbidity and ocean colour [@sent2025time].

Table: A [benchmark comparison](https://github.com/GeoscienceAustralia/eo-tides/blob/main/paper/benchmarking.ipynb) of tide modelling performance with parallelisation on vs. off, for a typical large-scale analysis involving a month of hourly tides modelled at 10,000 point locations using three models (FES2022, TPXO10, GOT5.6). \label{tab:benchmark}

| Cores | Parallelisation   | No parallelisation | Speedup |
| ----- | ----------------- | ------------------ | ------- |
| 8     | 2min 46s ± 663 ms | 9min 28s ± 536 ms  | 3.4x    |
| 32    | 55.9 s ± 560 ms   | 9min 24s ± 749 ms  | 10.1x   |

## Combining tides with satellite data

The [`eo_tides.eo`](https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.eo) module integrates modelled tides with `xarray`-format satellite data. `eo-tides` offers two tide attribution approaches that differ in complexity and performance: `tag_tides` assigns a single tide height per timestep for small-scale studies, while `pixel_tides` models tides spatially and temporally for larger-scale analyses, returning a unique tide height for each pixel in a dataset (\autoref{tab:tide_stats}, \autoref{fig:pixel}). These functions can be applied to satellite data for any coastal location on the planet, for example using open data loaded from the cloud using [ODC](https://www.opendatacube.org/) and STAC [@stac2024].

Table: Comparison of the `tag_tides` and `pixel_tides` functions. \label{tab:tide_stats}

| `tag_tides`                                                              | `pixel_tides`                                                                                   |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| - Assigns a single tide height to each satellite image timestep          | - Assigns a tide height to every individual pixel through time to capture spatial tide dynamics |
| - Ideal for local or site-scale analysis                                 | - Ideal for large-scale coastal product generation                                              |
| - Fast, low memory use                                                   | - Slower, higher memory use                                                                     |
| - Single tide height per image can produce artefacts and discontinuities | - Produce spatially seamless results across large regions                                       |

![An example spatial tide height output produced by the `pixel_tides` function.\label{fig:pixel}](figures/joss_fig_pixel.png)

## Calculating tide statistics and satellite biases

The [`eo_tides.stats`](https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.stats) module identifies biases caused by complex tide alaising interactions between tidal dynamics and satellite observations. These interactions can prevent satellites from observing the entire tide cycle [@eleveld2014estuarine; @sent2025time], leading coastal EO studies to produce biased or misleading results [@bishop2019NIDEM]. The `tide_stats` and `pixel_stats` functions produce a range of useful automated reports, plots and statistics that summarise how well a satellite time series captures real-world tidal conditions (\autoref{fig:stats}).

![An example of tidally-biased satellite coverage, where only ~68% of the modelled astronomical tide range is observed.\label{fig:stats}](figures/joss_fig_stats.png)

## Validating modelled tides

The [`eo_tides.validation`](https://geoscienceaustralia.github.io/eo-tides/api/#eo_tides.validation) module validates modelled tides against sea-level measurements from the GESLA Global Extreme Sea Level Analysis [@GESLAv3] archive (\autoref{fig:gesla}). It enables comparison of multiple tide models against observed data, allowing users to select optimal tide models for their specific study area or application (\autoref{fig:gesla}).

![A comparison of multiple tide models (EOT20, GOT5.5, HAMTIDE11) against observed sea level data from the Broome 62650 GESLA tide gauge.\label{fig:gesla}](figures/joss_fig_gesla.png)

# Research projects

Early versions of `eo-tides` functions have been used for continental-scale intertidal mapping [@deaintertidal], multi-decadal shoreline mapping across Australia [@bishop2021mapping] and [Africa](https://www.digitalearthafrica.org/platform-resources/services/coastlines), and for satellite-derived shoreline tide correction in the `CoastSeg` Python package [@Fitzpatrick2024].

# Acknowledgements

Functions from `eo-tides` were originally developed in the Digital Earth Australia Notebooks and Tools repository [@krause2021dea]. We thank all DEA Notebooks contributers for their invaluable assistance with code review, feature suggestions and code edits. This paper is published with the permission of the Chief Executive Officer, Geoscience Australia (copyright 2025).

# References
