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
#  - name: TBD # PyTMD author
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
# bibliography: TBD # paper.bib

# TODO add journal draft
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Satellite Earth observation offers an unparalleled method to view and examine dynamic coastal environments over large temporal and spatial scales. The variable influence of tide in these regions provides an added dimension that influences the utility of coastal Earth obseration data. `eo-tides` facilitates the addition of that tidal dimension to satellite Earth observation data, the combination of which delivers a powerful reimagining of traditional multi-temporal Earth observation data analysis. Conventionally, satellite data dimensions consider the geographical 'where' and the temporal 'when' of data acquisition. The addition of a tide height dimension allows exploration by data selection of the 'where' in the local tide range (e.g. high or low tide) and 'when' in the tidal cycle (e.g. spring, neap, ebb or flow tides) that data was collected. The concept has been used to map the mean annual position of continental [coastlines](https://doi.org/10.1016/j.rse.2021.112734), generate national [digital elevation models](https://doi.org/10.1016/j.ecss.2019.03.006) of intertidal zones and create synthetic imagery composites of coasts at [low and high tide](https://doi.org/10.3390/rs10030480).

# Statement of need
`eo-tides` is a powerful python based API that facilitates the modelling and attribution of global tide heights to satellite data, for improved data utility and analysis in coastal and marine research. It leverages satellite data acquisition times and a wide range of global tide models, using a python based tide prediction software, [`pyTMD`](https://pytmd.readthedocs.io/en/latest/). `eo-tides` further adopts functionality from the [`pandas`](https://pandas.pydata.org/docs/index.html), [`xarray`](https://docs.xarray.dev/en/stable/) and [`odc-geo`](https://odc-geo.readthedocs.io/en/latest/) packages to deliver a suite of flexible and parallelized Earth observation based tide modelling tools. Around coastlines especially, tidal dynamics can be spatially variable. `eo-tides` is applied at the pixel-scale of Earth observation datasets yet can be applied to petabytes of coastal satellite data for any time period or location globally. 

The `eo-tides` tool-suite supports a wide variety of global ocean models to "tag" satellite data with tide heights corresponding to exact moments of image acquisition. Furthermore, it supports pixel based tide modelling through time to produce three-dimensional `xarray` style datacubes that can be integrated with satellite data. Additional functionality enables parallelized modelling of tide heights and phases (e.g. high, low, ebb, flow tides), calculation of statistics describing local tide dynamics and biases caused by interactions between tidal processes and satellite orbits as well as validation of modelled tides using measured sea levels from coastal tide gauges (e.g. [GESLA Global Extreme Sea Level Analysis](https://gesla.org/)). `eo-tides` provides a suite of flexible, parallelized tools for efficient analysis of coastal and ocean Earth observation data – from regional to continental and the global scale.

# Functionality
## Modelling tides
The underlying `pyTMD` tide modelling capability provides the foundation of the `eo_tides` package. `eo_tides` in turn upscales the application of the sophisticated `pyTMD` modelling to scales appropriate to Earth observation data e.g. 10m spatial pixel resolution with Sentinel-2 imagery.

[TODO: Insert text here about the core functions of the `pyTMD` package with comment on the range of supported global tide models. Segue into how `eo-tides` enables the application of `pyTMD` modelling at EO appropriate scale]

The tide modelling functionality in `eo-tides` can be used independently of Earth observation (EO) data, e.g. for any application where you need to generate a time series of tide heights. However, it also underpins the more complex EO-related functions in the `eo-tides` package. Tide modelling functionality is also provided to support modelling of tidal phase at any location and time. This can be used to classify tides into high and low tide observations, or determine whether the tide was rising (i.e. flow tide) or falling (i.e. ebb tide) at any point in time.

## Combining tides with satellite data

## Calculating tide statistics and satellite biases

## Validating modelled tide heights

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
