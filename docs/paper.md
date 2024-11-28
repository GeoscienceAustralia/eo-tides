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
A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

The combination of satellite Earth observation data with tide height in coastal and marine research delivers a powerful new analysis dimension that enables a reimagining of traditional multi-temporal Earth observation data analysis. Conventionally, satellite data dimensions consider the geographical 'where' and the temporal 'when' of data acquisition. The addition of a tide height dimension allows the exploration by data selection of 'where' in the local tide range (e.g. high or low tide) and 'when' in the tidal cycle data was collected (e.g. spring, neap, ebb or flow tides). The concept has been used to map the mean annual position of the [coastline](https://doi.org/10.1016/j.rse.2021.112734), generate [digital elevation models](https://doi.org/10.1016/j.ecss.2019.03.006) of the intertidal zone and create synthetic imagery composites of the coastal zone at [low and high tide](https://doi.org/10.3390/rs10030480).

A wide range of global tide models have been developed that can predict tide heights for given locations and dates/times. [`pyTMD`](https://pytmd.readthedocs.io/en/latest/) is a python based software package that implements many of these tide models for anywhere on Earth. `eo-tides`, using functionality from [`pandas`](https://pandas.pydata.org/docs/index.html), [`xarray`](https://docs.xarray.dev/en/stable/) and [`odc-geo`](https://odc-geo.readthedocs.io/en/latest/), employs a suite of parallelized tools that integrates `pyTMD` modelled tide heights into satellite Earth observation datasets. Acknowledging that tidal dynamics have great spatial variation, `eo-tides` enables tide heights to be attributed to Earth observation data at the pixel scale and provides a flexible suite of tools that can be applied to petabytes of coastal satellite data for any time period or location globally. 

The `eo-tides` suite of tools support a wide variety of global ocean models and includes the ability to "tag" satellite data with tide heights corresponding to exact moments of image acquisition and integrates pixel based tide modelling through time. Additional functionality enables the modelling of tide heights and phases (e.g. high, low, ebb, flow tides) in parallel, calculation of statistics describing local tide dynamics and biases caused by interactions between tidal processes and satellite orbits and validation of modelled tides using measured sea levels from coastal tide gauges (e.g. [GESLA Global Extreme Sea Level Analysis](https://gesla.org/)).

# Statement of need
Satellite Earth observation offers an unparalleled method to view and examine dynamic coastal environments over large temporal and spatial scales. The variable influence of tide in these regions provides an added dimension that influences the utility of coastal Earth obseration data. Numerous global tide models exist that can be used to estimate tide height for any given location, date and time and the [`pyTMD`](https://pytmd.readthedocs.io/en/latest/) package is a python-based prediction software that enables tidal estimation from a variety of these models. `eo-tides` combines advanced tide modelling functionality from [`pyTMD`](https://pytmd.readthedocs.io/en/latest/) with [`pandas`](https://pandas.pydata.org/docs/index.html), [`xarray`](https://docs.xarray.dev/en/stable/) and [`odc-geo`](https://odc-geo.readthedocs.io/en/latest/), enabling pixel based tide height attribution to Earth obseravtion data. Tide-height attributed Earth observation data has been used to map the mean annual position of the [coastline](https://doi.org/10.1016/j.rse.2021.112734), generate [digital elevation models](https://doi.org/10.1016/j.ecss.2019.03.006) of the intertidal zone and create synthetic imagery composites of the coastal zone at [low and high tide](https://doi.org/10.3390/rs10030480). `eo-tides` provides a suite of flexible, parallelized tools for efficient analysis of coastal and ocean Earth observation data – from regional to continental and the global scale.

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
