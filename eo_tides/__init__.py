"""eo_tides: Tide modelling tools for large-scale satellite earth observation analysis.

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
eo : Tools for integrating satellite EO data with tide modelling
stats : Tools for analysing local tide dynamics and satellite biases
utils : General-purpose utilities for tide model setup and data processing
validation : Validation tools for comparing modelled tides to observed tide gauge data.
"""

from importlib.metadata import version

# Import commonly used functions for convenience
from .eo import pixel_tides, tag_tides
from .model import ensemble_tides, model_phases, model_tides
from .stats import pixel_stats, tide_aliasing, tide_stats
from .utils import clip_models, idw, list_models
from .validation import eval_metrics, load_gauge_gesla

# Define what should be imported with "from eo_tides import *"
__all__ = [
    "clip_models",
    "ensemble_tides",
    "eval_metrics",
    "idw",
    "list_models",
    "load_gauge_gesla",
    "model_phases",
    "model_tides",
    "pixel_stats",
    "pixel_tides",
    "tag_tides",
    "tide_aliasing",
    "tide_stats",
]

# Add version metadata to package
__version__ = version("eo-tides")
