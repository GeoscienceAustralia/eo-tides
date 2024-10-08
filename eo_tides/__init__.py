"""
eo_tides
========

Tide modelling tools for large-scale satellite earth observation analysis.

`eo-tides` provides powerful parallelized tools for integrating satellite
Earth observation data with tide modelling. `eo-tides` combines advanced
tide modelling functionality from the `pyTMD` package with `pandas`,
`xarray` and `odc-geo`, providing a suite of flexible tools for efficient
analysis of coastal and ocean Earth observation data – from regional,
continental, to global scale.

These tools can be applied to petabytes of freely available satellite
data (e.g. from Digital Earth Australia or Microsoft Planetary Computer)
loaded via Open Data Cube's `odc-stac` or `datacube` packages, supporting
coastal and ocean earth observation analysis for any time period or
location globally.

Modules
-------
model : Core tide modelling functionality
eo : Combine satellite EO data with tide modelling
stats : Calculate local tide dynamics and satellite bias statistics
utils : Utility functions and helper tools
validation : Load observed tide gauge data to validate modelled tides
"""

# Import commonly used functions for convenience
from .eo import pixel_tides, tag_tides
from .model import list_models, model_tides
from .stats import tide_stats
from .utils import idw
from .validation import eval_metrics, load_gauge_gesla

# Define what should be imported with "from eo_tides import *"
__all__ = [
    "list_models",
    "model_tides",
    "tag_tides",
    "pixel_tides",
    "tide_stats",
    "idw",
    "eval_metrics",
    "load_gauge_gesla",
]
