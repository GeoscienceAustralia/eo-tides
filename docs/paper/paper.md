---
title: 'eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis'
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
date: 27 November 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.

---

# Summary
The `eo-tides` package provides powerful parallelized tools for integrating satellite Earth observation (EO) data with ocean tide modelling. The package provides a flexible Python-based API that facilitates the modelling and attribution of tide heights to a time-series of satellite images based on the spatial extent and acquisition time of each satellite observation.

`eo-tides` leverages advanced tide modelling functionality from the `pyTMD` [@tyler_sutterley_2024] tide prediction software, combining this fundamental tide modelling capability with EO spatial analysis tools from `odc-geo` [@odc-geo]. This allows tides to be modelled in parallel automatically using over 50 supported tide models, and returned in standardised `pandas` [@The_pandas_development_team_pandas-dev_pandas_Pandas] and `xarray` [@Hoyer_xarray_N-D_labeled_2017] data formats for further analysis.

Tools from the `eo-tides` package are designed to be applied directly to petabytes of freely available satellite data loaded from the cloud using Open Data Cube's `odc-stac` or `datacube` packages (e.g. using [Digital Earth Australia](https://knowledge.dea.ga.gov.au/guides/setup/gis/stac/) or [Microsoft Planetary Computer's](https://planetarycomputer.microsoft.com/) SpatioTemporal Asset Catalogue). Additional functionality enables evaluating potential satellite-tide biases, and validating modelled tides using external tide gauge data — both important considerations for assessing the reliability and accuracy of coastal EO workflows. In combination, these open source tools support the efficient, scalable and robust analysis of coastal EO data for any time period or location globally.

# Statement of need
Satellite remote sensing offers an unparalleled method to view and examine dynamic coastal environments over large temporal and spatial scales. However, the variable and sometimes extreme influence of ocean tides in these regions can complicate analyses, making it difficult to separate the influence of changing tides from patterns of true coastal change over time. This is a particularly significant challenge for continental- to global-scale coastal EO analyses, where failing to account for complex tide dynamics can lead to inaccurate or misleading insights into coastal processes observed by satellites.

Conversely, information about ocean tides can also provide unique environmental insights that can greatly enhance the utility of coastal EO data. Conventionally, satellite data dimensions consider the geographical "where" and the temporal "when" of data acquisition. The addition of tide height as a new analysis dimension allows data to be filtered, sorted and analysed with respect to tidal processes, delivering a powerful re-imagining of traditional multi-temporal EO data analysis. For example, satellite data can be analysed to focus on specific ecologically-significant portions of the local tide range (e.g. high or low tide), or by each image's position in the tidal cycle (e.g. spring, neap, ebb or flow tides).

This concept has been used to map tidally-corrected annual coastlines from Landsat satellite data at continental scale [@bishop2021mapping], generate digital elevation models of the intertidal zone [@bishop2019NIDEM], and create tidally-constrained imagery composites of the coastline at low and high tide [@sagar2018composites]. However, these approaches have been historically based on bespoke, closed-source tide modelling tools, limiting the reproducibility and portability of these techniques to new coastal EO applications. To support the next generation of coastal EO workflows, there is a critical need for new open-source approaches for combining satellite data with tide modelling.

The `eo-tides` package aims to address these challenges by providing a set of performant open-source Python tools for attributing satellite EO data with modelled ocean tides. This functionality is provided in four main analysis modules (`model`, `eo`, `stats` and `validation`) which are described briefly below.

## Modelling tides
The `eo_tides.model` module builds upon advanced tide modelling capability provided by the `pyTMD` tide prediction software [@tyler_sutterley_2024].

[TODO Tyler: Insert brief paragraph here about the core capability of the `pyTMD` package, with general background to the science used to predict tides and the range of supported global tide models]

[TODO Robbi: Insert brief paragraph here about how `eo-tides` wraps `pyTMD` functionality to model tides in parallel and return data in pandas/xarray format required for EO analysis]

Tide modelling functionality in the `model_tides` function is primarily intended to support more complex EO-related capability in the `eo_tides.eo` module. However it can also be used independently of EO data, for example for any application that requires a time series of modelled tide heights. In addition to modelling tide heights, the `model_phases` function allows users to calculate tidal phases at any location and time. This can be used to classify tides into high and low tide observations, or determine whether the tide was rising (i.e. flow tide) or falling (i.e. ebb tide) at any point in time.

## Combining tides with satellite data
The `eo_tides.eo` module contains the package's core functionality, focusing on tools for attributing satellite data with modelled tide heights.
For tide attribution, `eo-tides` offers two approaches that differ in complexity and performance: `tag_tides` and `pixel_tides` (Table 1).

The `tag_tides` function provides a fast and efficient method for small scale applications where tides are unlikely to vary across a study area. This approach allocates a single tide height to each satellite data timestep, based on the geographic-centroid of the dataset and the acquisition time of each image. Having tide height as a variable allows the selection and analysis of satellite data based on tides. For example, all available satellite observations for an area of interest could be sorted by tide height, or used to extract and comapre the lowest and highest tide images in the time series.

However, in reality tides vary spatially – potentially by many metres in areas of complex and extreme tidal dynamics. This means that an individual satellite image can capture a range of contrasting tide conditions. For larger scale coastal EO analysis, the `pixel_tides` function can be used to seamlessly model tides through both time and space, producing three-dimensional "tide height" datacube that can be integrated with satellite data. For efficient processing, `pixel_tides` `models tides into a customisable low resolution grid surrounding each satellite image in the time series. These modelled tides are then re-projected back into the original resolution of the input satellite image, returning a unique tide height for every individual satellite pixel through time.

Table 1: Comparison of the `tag_tides` and `pixel_tides` functions for attributing satellite EO with tide heights.

| `tag_tides`                                                                 | `pixel_tides`                                                                                              |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Assigns a single tide height to each timestep/satellite image                         | Assigns a tide height to every individual pixel through time to capture spatial tide dynamics                                               |
| Ideal for local or site-scale analysis                                      | Ideal for regional to global-scale coastal product generation                                              |
| Fast, low memory use                                                        | Slower, higher memory use                                                                                  |
| Single tide height per image can produce artefacts in complex tidal regions | Produce spatially seamless results across large extents by applying analyses at the pixel level |

## Calculating tide statistics and satellite biases
The `eo_tides.stats` module contains tools for calculating statistics describing local tide dynamics, as well as biases caused by interactions between tidal processes and satellite orbits. Complex interactions between temporal tide dynamics and the regular overpass timing of sun-synchronous satellite sensors like Landsat, Sentinel-1 and Sentinel-2 mean that satellites often do not always observe the entire tidal cycle. Biases in satellite coverage of the tidal cycle can mean that tidal extremes (e.g. the lowest or highest tides at a location) may either never be captured by satellites, or be over-represented in the satellite EO record. Local tide dynamics can cause these biases to vary greatly both through time and space [@bishop2019NIDEM], making it challenging to compare coastal processes consistently - particularly for large-scale coastal EO analyses.

To ensure that coastal EO analyses are not inadvertently affected by tide biases, it is important to understand and compare how well the tides observed by satellites match the full range of modelled tides at a location. The `tide_stats` function compares the subset of tides observed by satellite data against the full range of tides modelled at a regular interval through time across the entire time period covered by the satellite dataset. This comparison is used to calculate several useful statistics that summarise how well a satellite time series captures the full range of real-world tidal conditions [@bishop2019NIDEM]. These statistics include:

  1. Spread: The proportion of the modelled astronomical tidal range that was observed by satellites. A high value indicates good coverage of the tide range.
  2. High-tide offset: The proportion of the highest tides never observed by satellites, relative to the modelled astronomical tidal range. A high value indicates that the satellite data never captures the highest tides.
  3. Low-tide offset: The proportion of the lowest tides never observed by satellites, relative to the modelled astronomical tidal range. A high value indicates that the satellite data never captures the lowest tides.

An interrogation of satellite tide bias in any area of interest using `eo-tides` statistical functionality will return a report and plot \autoref{fig:stats}, adding insightful tide-based context to a coastal EO analysis:

<!-- ![](../assets/Sen2_tide_stats.png)
<div style="margin-left: 40px; margin-right: 40px; margin-bottom: 20px;">
<b> Figure 1 </b> In this example satellite time series, the data captured a biased proportion of the tide range: only observing ~68% of the modelled astronomical tide range, and never observing the lowest 24% of tides. The plot visually demonstrates the relationships between satellite observed tide heights (black dots) and modelled astronomical tide height (blue lines) at this location.</div> -->

![In this example satellite time series, the data captured a biased proportion of the tide range: only observing ~68% of the modelled astronomical tide range, and never observing the lowest 24% of tides. The plot visually demonstrates the relationships between satellite observed tide heights (black dots) and modelled astronomical tide height (blue lines) at this location.\label{fig:stats}](figures/joss_fig_stats.png)


## Validating modelled tide heights
The tide models supported by `eo-tides` can vary significantly in accuracy across the world's coastlines. Evaluating the accuracy of modelled tides is critical for ensuring that resulting marine or coastal EO analyses are reliable and useful.

Validation functionality in `eo-tides` provides a convenient tool for loading high-quality sea-level measurements from the GESLA Global Extreme Sea Level Analysis [@GESLAv3] archive – a global dataset of almost 90,713 years of sea level data from 5,119 records across the world. This data can be used to compare against tides modelled using `eo-tides` to calculate the accuracy of your tide modelling and identify the optimal tide models to use for your study area.

`eo-tides` has designed the loading of GESLA data to be identical in format to the tide modelling data load and correlations between GESLA data and modelled tide heights are quantified through the calculation of accuracy statistics that include the Root Mean Square Error (RMSE), Mean Absolute Error (MAE), R-squared and bias.

Furthermore, different ocean tide models perform differently in different locations. `eo-tides` supports the comparison of multiple tide models, evaluating them against GESLA data, to empower users to make informed decisions and choose the optimal tide model that best suits their location or application with confidence.

# Research projects
Early versions of functions provided in `eo-tides` has been used to support continental-scale modelling of the elevation and exposure of Australia's intertidal zone, and to support tide correction for satellite-derived shorelines as part of the `CoastSeg` Python package.

# Acknowledgements
Functions from `eo-tides` were originally developed in the Digital Earth Australia Notebooks and Tools repository. The authors would like to thank all DEA Notebooks contributers and maintainers for their invaluable assistance with code review, feature suggestions and code edits. This paper is published with the permission of the Chief Executive Officer, Geoscience Australia.

# References
