---
title: 'eo-tides: Tide modelling tools for large-scale satellite Earth observation analysis'
tags:
  - Python
  - Earth observation
  - Tide modelling
  - Parallelized
  - Scalable
authors:
  - name: Robbi Bishop-Taylor
    corresponding: true # (This is how to denote the corresponding author)
#    orcid: TBD # 0000-0000-0000-0000
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1 
  - name: Tyler Sutterley
#    equal-contrib: TBD # true 
    affiliation: 2 # (Multiple affiliations must be quoted)
  - name: Claire Phillips
#    corresponding: TBD # true 
    affiliation: 1
affiliations:
 - name: Geoscience Australia, Australia
   index: 1
#   ror: TBD # uncertain what this code is e.g. 00hx57361
# - name: TBD # Institution Name, Country
   index: 2
date: 27 November 2024
bibliography: paper.bib

# TODO add journal draft
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Satellite Earth observation offers an unparalleled method to view and examine dynamic coastal environments over large temporal and spatial scales. The variable influence of tide in these regions provides another dimension that increases the utility of coastal Earth obseration data. `eo-tides` facilitates the attribution of that tidal dimension to satellite Earth observation data, the combination of which delivers a powerful reimagining of traditional multi-temporal Earth observation data analysis. Conventionally, satellite data dimensions consider the geographical 'where' and the temporal 'when' of data acquisition. The addition of a tide height dimension allows exploration by data selection of the 'where' in the local tide range (e.g. high or low tide) and 'when' in the tidal cycle (e.g. spring, neap, ebb or flow tides) that data was collected. The concept has been used to map the mean annual position of continental coastlines [@bishop2021mapping], generate national digital elevation models [@bishop2019NIDEM] of intertidal zones and create synthetic imagery composites of coasts at low and high tide [@sagar2018composites].

# Statement of need
`eo-tides` is a powerful python based API that facilitates the modelling and attribution of global tide heights to satellite data, for improved data utility and analysis in coastal and marine research. It leverages satellite data acquisition times and a wide range of global tide models, using a python based tide prediction software, `pyTMD` [@tyler_sutterley_2024]. `eo-tides` further adopts functionality from the `pandas` [@The_pandas_development_team_pandas-dev_pandas_Pandas], `xarray` [@Hoyer_xarray_N-D_labeled_2017] and `odc-geo` [@odc-geo] packages to deliver a suite of flexible and parallelized Earth observation based tide modelling tools. Around coastlines especially, tidal dynamics can be spatially variable. `eo-tides` is applied at the pixel-scale of Earth observation datasets yet can be applied to petabytes of coastal satellite data for any time period or location globally. 

The `eo-tides` tool-suite supports a wide variety of global ocean models to "tag" satellite data with tide heights corresponding to exact moments of image acquisition. Furthermore, it supports pixel based tide modelling through time to produce three-dimensional `xarray` style datacubes that can be integrated with satellite data. Additional functionality enables parallelized modelling of tide heights and phases (e.g. high, low, ebb, flow tides), calculation of statistics describing local tide dynamics and biases caused by interactions between tidal processes and satellite orbits as well as validation of modelled tides against measured sea levels from coastal tide gauges (e.g. GESLA Global Extreme Sea Level Analysis [@GESLAv3]). `eo-tides` provides a suite of flexible, parallelized tools for efficient analysis of coastal and ocean Earth observation data – from regional to continental and the global scale.

# Functionality
## Modelling tides
The underlying `pyTMD` tide modelling capability provides the foundation of the `eo_tides` package. `eo_tides` upscales the sophisticated tide modelling in `pyTMD` to scales appropriate to Earth observation data e.g. 10m spatial pixel resolution with Sentinel-2 imagery.

[TODO: Insert text here about the core functions of the `pyTMD` package with comment on the range of supported global tide models. Segue into how `eo-tides` enables the application of `pyTMD` modelling at EO appropriate scale]

The tide modelling functionality in `eo-tides` can be used independently of Earth observation (EO) data, e.g. for any application where you need to generate a time series of tide heights. However, it also underpins the more complex EO-related functions in the `eo-tides` package. Tide modelling functionality is also provided to support modelling of tidal phase at any location and time. This can be used to classify tides into high and low tide observations, or determine whether the tide was rising (i.e. flow tide) or falling (i.e. ebb tide) at any point in time.

## Combining tides with satellite data
When combining tide heights with satellite data, `eo-tides` offers two approaches that differ in their complexity and performance. A fast and efficient method for assigning tide heights to whole timesteps is offered that best suits small scale applications where tides are unlikely to vary across a study area. In contrast, for large scale, seamless coastal EO datasets, `eo-tides` offers an approach that models tides through both time and space, returning a tide height for every satellite pixel in the dataset. However, the complexity of this approach naturally comes at the expense of performance.

Using the former approach, the tide height at the geographic-centroid of the dataset is attributed to each timestep, representing the relative trend of the tide dynamics for the area of interest at that moment in time. Having tide height as a variable allows the selection and analysis of satellite data using information about tides. For example, in any area of interest, we could sort all available timesteps by tide height, then identify and compare the lowest and highest tide images in our time series.

However, in reality, tides vary spatially – potentially by several metres in areas of complex tidal dynamics. This means that an individual satellite image can capture a range of tide conditions. The pixel-based, seamless tide height attribution approach is well suited for applications that require localised information on tides. For efficient processing, this approach first models tides into a low resolution grid surrounding each satellite image in the time series. This lower resolution data includes a buffer around the extent of the satellite data so that tides can be modelled seamlessly across analysis boundaries. Optionally, users can interpolate and re-project the low resolution tide data back into the resolution of the input satellite image, resulting in an individual tide height for every pixel in the dataset through time and space.

Further functionality in this latter approach allows users to calculate and return timesteps from the modelled tide height array that reveal important features of the input satellite time series. These include the minimum, maximum and median satellite-observed tide heights in the array and are returned by supplying the desired quantiles to the analysis parameters.

## Calculating tide statistics and satellite biases
Complex interactions between temporal tide dynamics and the regular mid-morning overpass timing of sun-synchronous sensors like Landsat or Sentinel-2 mean that satellites often do not observe the entire tidal cycle. Biases in satellite coverage of the tidal cycle can mean that tidal extremes (e.g. the lowest or highest tides at a location) may either never be captured by satellites, or be over-represented in the satellite EO record. Local tide dynamics can cause these biases to vary greatly both through time and spatially [@bishop2019NIDEM], making it challenging to consistently analyse and compare coastal processes consistently - particularly for large-scale (e.g. regional or global) analyses.

To ensure that coastal EO analyses are not inadvertently affected by tide biases, it is important to understand and compare how well the tides observed by satellites match the full range of modelled tides at a location. Statistical capabilities in `eo-tides` compare the subset of tides observed by satellite data against the full range of tides modelled at a regular interval through time across the entire time period covered by the satellite dataset. This comparison is used to calculate several useful statistics that summarise how well your satellite data capture real-world tidal conditions. These statistics include: 

  1. Spread: The proportion of the full modelled astronomical tidal range that was observed by satellites. A high value indicating good coverage of the tide range.
  2. Offset high: The proportion of the highest tides not observed by satellites at any time, as a proportion of the full modelled astronomical tidal range. A high value indicates that the satellite data is biased towards never capturing high tides.
  3. Offset low: The proportion of the lowest tides not observed by satellites at any time, as a proportion of the full modelled astronomical tidal range. A high value indicates that the satellite data is biased towards never capturing low tides.

For either an area-of-interest or pixel-based application of tide modelling using `eo-tides`, the statistical functionality can be used to summarise your dataset, adding insightful tide-based context to your coastal satellite data. For example, application of an `eo-tides` statistical function over your dataset will return a report of the tidal characteristics for your dataset with an associated plot to visualise the data (\autoref{fig:1}).

```
  Using tide modelling location: 122.21, -18.00
  Modelling tides with EOT20

  - Modelled astronomical tide range: 9.30 metres.
  - Observed tide range: 6.29 metres.

  - 68% of the modelled astronomical tide range was observed at this location.
  - The highest 8% (0.77 metres) of the tide range was never observed.
  - The lowest 24% (2.25 metres) of the tide range was never observed.

  - Mean modelled astronomical tide height: -0.00 metres.
  - Mean observed tide height: 0.69 metres.

  - The mean observed tide height was 0.69 metres higher than the mean modelled astronomical tide height.
```
[In this satellite dataset where tide modelling is applied for the geometric centroid 122.21, -18.00, using tide model EOT20,the satellite captured a biased proportion of the tide range at this location: only observing ~68% of the tide range, and never observing the lowest 24% of tides The plot visually demonstrates the relationships between satellite observed tide-height and modelled astronomical tide height for this location.\label{fig:Figure 1}]

![Test image.\label{fig:"Figure x"; label="figx"}](../assets/eo-tides-logo-128.png)

test fig text \autoref{figx} \autoref{Figure x}

<div style="margin-left: 20px;">
<b> Figure 1 </b> In this satellite dataset where tide modelling is applied for the geometric centroid 122.21, -18.00, using tide model EOT20,the satellite captured a biased proportion of the tide range at this location: only observing ~68% of the tide range, and never observing the lowest 24% of tides. The plot visually demonstrates the relationships between satellite observed tide-height and modelled astronomical tide height for this location.
</div>

Applying `eo-tides` statistical functionality at the pixel scale produces rasterised output datasets of the tidal statistics and biases represented in (\autoref{fig:1}) *Figure 1* that can be used to explore spatial relationships to satellite-tide bias in your input dataset.

## Validating modelled tide heights
The tide models supported by `eo-tides` can vary significantly in accuracy across the world's coastlines. Evaluating the accuracy of your modelled tides is critical for ensuring that resulting marine or coastal EO analyses are reliable and useful.

Validation functionality in `eo-tides` provides a convenient tool for loading high-quality sea-level measurements from the GESLA Global Extreme Sea Level Analysis (REFERENCE) archive – a global archive of almost 90,713 years of sea level data from 5,119 records across the world. This data can be used to compare against tides modelled using `eo-tides` to calculate the accuracy of your tide modelling and identify the optimal tide models to use for your study area.

To streamline validation analyses and ensure comparability with tide height datasets, `eo-tides` has designed the loading of GESLA data to be identical in format to the tide modelling data load. Correlations between GESLA data and modelled tide heights are quantified in `eo-tides` with the calculation of accuracy statistics that include the Root Mean Square Error (RMSE), Mean Absolute Error (MAE), R-squared and bias.

Furthermore, different ocean tide models perform differently in different locations, meaning it can be valuable to compare the accuracy of multiple different models against measured gauge data. This can help users make an informed decision about the best model to use for a given application or study area. `eo-tides` allows users to compare tide models and evaluate them against GESLA data, empowering users to consider the accuracy metrics and choose the optimal tide model that best suits their location with confidence.

# References
A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.
https://joss.readthedocs.io/en/latest/paper.html#internal-references
https://joss.readthedocs.io/en/latest/paper.html#citations

# Research projects
Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

# Acknowledgements
Acknowledgement of any financial support.

---
Check that the paper complies with Pandoc compilation into PDF
https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles
---
