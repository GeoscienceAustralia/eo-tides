# Setting up tide models

!!! important

    `eo-tides` provides tools for modelling tides using global ocean tide models but does not host or maintain the models themselves. **Users are responsible for accessing, using, and citing ocean tide models in compliance with each model's licensing terms.**

Once you [have installed `eo-tides`](install.md), we need to download and set up the external global ocean tide models required for `eo-tides` to work.
The following documentation provides instructions for getting started with several common global ocean tide models.

!!! tip

    Please refer to the [`pyTMD` documentation](https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories) for additional instructions covering all other supported tide models.

## Setting up a tide model directory

As a first step, we need to create a directory that will contain our tide model data.
This directory will be accessed by all `eo-tides` functions.
For example, we might want to store our tide models in a directory called `tide_models/`:

```
tide_models/
```

!!! tip

    This directory doesn't need to be called `tide_models`; use any name and/or location that is convenient to you and accessible from your Python environment. Please refer to [the documentation below](#configuring-eo-tides-to-use-tide-model-directory) for further details on configuring `eo-tides` to use this directory.

## Downloading tide model data

Now we need to download some data from one or more models, and save this into our tide model directory.
Follow the guides below for some of the most commonly used global ocean tide models:

??? note "EOT20 Empirical Ocean Tide model (default)"

    ### EOT20 Empirical Ocean Tide model (default)

    1. Visit [EOT20 - A global Empirical Ocean Tide model from multi-mission satellite altimetry](https://doi.org/10.17882/79489)
    2. Under `Data`, click `Download`:

        ![image](assets/eot20_download.jpg)

    3. Create a new directory inside your [tide model directory](#setting-up-a-tide-model-directory) called `EOT20/` to store the EOT20 model files.

    4. Extract the `85762.zip` and then `ocean_tides.zip` into this new directory. You should end up with the following directory structure containing the extracted NetCDF files:

        ```
        tide_models/EOT20/ocean_tides/
           |- 2N2_ocean_eot20.nc
           |- ...
           |- T2_ocean_eot20.nc
        ```

??? note "FES2022 Finite Element Solution tide models"

    ### FES2022 Finite Element Solution tide models

    1. [Register with AVISO+](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html), and select `FES (Finite Element Solution - Oceanic Tides Heights)` from the `Licence Agreement and product selection` section:

        ![image](assets/fes_productselection.jpg)

    2. Once your registration is complete, login to [MY AVISO+](https://www.aviso.altimetry.fr/en/my-aviso-plus.html).
    3. Once logged in, select [My products](https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html) in the left-hand menu:

        ![image](https://user-images.githubusercontent.com/17680388/160057999-381fb818-e379-46cb-a3c4-a836308a96d8.png)

    4. `FES (Finite Element Solution - Oceanic Tides Heights)` should appear under `Your current subscriptions.` Right click on `Ftp`, and copy the FTP address.

        ![image](https://user-images.githubusercontent.com/17680388/160058064-77430ddf-1939-449d-86e7-f05b27ca768a.png)

    5. Using an FTP client like FileZilla, log in to the FTP using your AVISO+ username and password:

        ![image](https://user-images.githubusercontent.com/17680388/160058263-b0b1da72-e5ac-47ca-b1d0-544569d3f06a.png)

    6. Navigate to `/auxiliary/tide_model/`, and download the contents of one or more of the following directories:

        - `fes2022b/ocean_tide/`
        - `fes2022b/ocean_tide_extrapolated/`

        !!! tip

            The "extrapolated" version of FES models have been extended inland using a simple "nearest" extrapolation method to ensure data coverage across the entire coastal zone. This can be useful for ensuring you always return a modelled tide, but can also introduce uncertainty into your modelling (particularly in complex regions such as narrow peninsulas or inlets/embayments).

    7. Create new nested directories inside your [tide model directory](#setting-up-a-tide-model-directory) called `fes2022b/ocean_tide/` (if using standard model data) or `fes2022b/ocean_tide_extrapolated/` (if using extrapolated model data) to store the FES2022 model files.

    8. Extract your `...nc.xz` files into this directory (e.g. `tar -xf m2_fes2022.nc.xz`). You should end up with the following directory structure containing the extracted NetCDF files:

        ```
        tide_models/fes2022b/ocean_tide/
           |- 2n2_fes2022.nc
           |- ...
           |- t2_fes2022.nc
        ```
        Or:
        ```
        tide_models/fes2022b/ocean_tide_extrapolated/
           |- 2n2_fes2022.nc
           |- ...
           |- t2_fes2022.nc
        ```

??? note "FES2014 Finite Element Solution tide models"

    ### FES2014 Finite Element Solution tide models

    1. [Register with AVISO+](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html), and select `FES (Finite Element Solution - Oceanic Tides Heights)` from the `Licence Agreement and product selection` section:

        ![image](assets/fes_productselection.jpg)

    2. Once your registration is complete, login to [MY AVISO+](https://www.aviso.altimetry.fr/en/my-aviso-plus.html).
    3. Once logged in, select [My products](https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html) in the left-hand menu:

        ![image](https://user-images.githubusercontent.com/17680388/160057999-381fb818-e379-46cb-a3c4-a836308a96d8.png)

    4. `FES (Finite Element Solution - Oceanic Tides Heights)` should appear under `Your current subscriptions.` Right click on `Ftp`, and copy the FTP address.

        ![image](https://user-images.githubusercontent.com/17680388/160058064-77430ddf-1939-449d-86e7-f05b27ca768a.png)

    5. Using an FTP client like FileZilla, log in to the FTP using your AVISO+ username and password:

        ![image](https://user-images.githubusercontent.com/17680388/160058263-b0b1da72-e5ac-47ca-b1d0-544569d3f06a.png)

    6. Navigate to `/auxiliary/tide_model/`, and download the contents of one or more of the following directories:

        - `fes2014_elevations_and_load/fes2014b_elevations/`
        - `fes2014_elevations_and_load/fes2014b_elevations_extrapolated/`

        !!! tip

            The "extrapolated" version of FES have been extended inland using a simple "nearest" extrapolation method to ensure data coverage across the entire coastal zone. This can be useful for ensuring you always return a modelled tide, but can also introduce uncertainty into your modelling (particularly in complex regions such as narrow peninsulas or inlets/embayments).

    7. Create a new directory inside your [tide model directory](#setting-up-a-tide-model-directory) called `fes2014/` to store the FES2014 model files.

    8. Extract `ocean_tide.tar.xz` or `ocean_tide_extrapolated.tar.xz` into this directory (e.g. `tar -xf ocean_tide.tar.xz`). You should end up with the following directory structure containing the extracted NetCDF files:

        ```
        tide_models/fes2014/ocean_tide/
           |- 2n2.nc
           |- ...
           |- t2.nc
        ```
        Or:
        ```
        tide_models/fes2014/ocean_tide_extrapolated/
           |- 2n2.nc
           |- ...
           |- t2.nc
        ```

??? note "GOT Global Ocean Tide models"

    ### GOT Global Ocean Tide models

    1. Visit [Ocean tide models](https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models)
    2. Under `Short-period (diurnal/semidiurnal) tides`, click choose your desired GOT model:

        ![image](assets/got_download.jpg)

    3. Create a new directory inside your [tide model directory](#setting-up-a-tide-model-directory) called either `GOT4.7/`, `got4.8/`, `GOT4.10c/`, `GOT5.5/` or `GOT5.6/` to store the GOT model files.

    4. Extract your downloaded `.tar.gz` file into this new directory. You should end up with the following directory structure containing the extracted NetCDF files:

        ```
        tide_models/GOT5.6/ocean_tides/
           |- ...
        ```
        Or:
        ```
        tide_models/GOT5.5/ocean_tides/
           |- ...
        ```
        !!! important

            Note that GOT5.6 requires that both GOT5.6 and GOT5.5 model files are downloaded and extracted.

        Or:
        ```
        tide_models/GOT4.10c/grids_oceantide/
           |- ...
        ```
        Or:
        ```
        tide_models/got4.8/grids_oceantide/
           |- ...
        ```
        Or:
        ```
        tide_models/GOT4.7/grids_oceantide/
           |- ...
        ```

## Configuring `eo-tides` to use tide model directory

`eo-tides` can be pointed to the location of your [tide model directory](#setting-up-a-tide-model-directory) and your downloaded tide model data in two ways:

### Using the `directory` function parameter

All tide modelling functions from `eo-tides` provide a `directory` parameter that can be used to specify the location of your tide model directory.
For example, using `model_tides` from the `eo_tides.model` module:

```py hl_lines="8"
import pandas as pd
from eo_tides.model import model_tides

model_tides(
        x=155,
        y=-35,
        time=pd.date_range("2022-01-01", "2022-01-04", freq="1D"),
        directory="tide_models/"
)
```

### Setting the `EO_TIDES_TIDE_MODELS` environmental variable

For more advanced usage, you can set the path to your [tide model directory](#setting-up-a-tide-model-directory) by setting the `EO_TIDES_TIDE_MODELS` environment variable:

```py hl_lines="2"
import os
os.environ["EO_TIDES_TIDE_MODELS"] = "tide_models/"
```

All tide modelling functions from `eo-tides` will check for the presence of the `EO_TIDES_TIDE_MODELS` environment variable, and use it as the default `directory` path if available (the `EO_TIDES_TIDE_MODELS` environment variable will be overuled by the `directory` parameter if provided).

!!! tip

    Setting the `EO_TIDES_TIDE_MODELS` environment variable can be useful when the location of your tide model directory might change between different environments, and you want to avoid hard-coding a single location via the `directory` parameter.

## Verifying available and supported models

You can check what tide models have been correctly set up for use by `eo-tides` using the [`eo_tides.model.list_models`](api.md#eo_tides.model.list_models) function:

```py
from eo_tides.model import list_models

available_models, supported_models = list_models(directory="tide_models/")
```

This will print out a useful summary, with available models marked with a ✅:

```
─────────────────────────────────────────────────────────────────────────────
 󠀠🌊  | Model        | Expected path
─────────────────────────────────────────────────────────────────────────────
 ✅  │ EOT20        │ tests/data/tide_models_tests/EOT20/ocean_tides
 ❌  │ FES2014      │ tests/data/tide_models_tests/fes2014/ocean_tide
 ✅  │ HAMTIDE11    │ tests/data/tide_models_tests/hamtide
 ❌  │ TPXO9.1      │ tests/data/tide_models_tests/TPXO9.1/DATA
 ...   ...            ...
─────────────────────────────────────────────────────────────────────────────

Summary:
Available models: 2/50
```

## Next steps

Now that you have [installed `eo-tides`](install.md) and set up some tide models, you can learn how to use `eo-tides` for [modelling tides and analysing satellite data!](notebooks/Model_tides.ipynb)
