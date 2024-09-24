# Setting up tide models

## Getting started with the [FES2014 Finite Element Solution tide model](https://www.aviso.altimetry.fr/es/data/products/auxiliary-products/global-tide-fes/description-fes2014.html)

1. [Register with AVISO+](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html), and select `FES2014 / FES2012 (Oceanic Tides Heights)` from the `PRODUCT SELECTION` section:

   ![image](https://user-images.githubusercontent.com/17680388/160057710-dbb0c8b9-56e9-451a-91c3-d90e503d8512.png)

2. Once your registration is complete, login to [MY AVISO+](https://www.aviso.altimetry.fr/en/my-aviso-plus.html).
3. Once logged in, select [My products](https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html) in the left-hand menu:

   ![image](https://user-images.githubusercontent.com/17680388/160057999-381fb818-e379-46cb-a3c4-a836308a96d8.png)

4. `FES2014 / FES2012 (Oceanic Tides Heights)` should appear under `Your current subscriptions.` Right click on `Ftp`, and copy the FTP address.

   ![image](https://user-images.githubusercontent.com/17680388/160058064-77430ddf-1939-449d-86e7-f05b27ca768a.png)

5. Using an FTP client like FileZilla, log in to the FTP using your AVISO+ username and password:

   ![image](https://user-images.githubusercontent.com/17680388/160058263-b0b1da72-e5ac-47ca-b1d0-544569d3f06a.png)

6. Navigate to `/auxiliary/tide_model/fes2014_elevations_and_load/`, and download the following files:

- `fes2014a_loadtide/load_tide.tar.xz`
- `fes2014b_elevations/ocean_tide.tar.xz` (_or_ `fes2014b_elevations_extrapolated/ocean_tide_extrapolated.tar.xz`; this extrapolated version includes additional coverage of the coastal zone, which can be useful for coastal applications)

7. Create a new folder (i.e. `tide_models/fes2014/`) to store the model files in a location that will be accessible to the DEA Coastlines code. Extract `load_tide.tar.xz` and `ocean_tide.tar.xz` into this folder (e.g. `tar -xf load_tide.tar.xz`). You should end up with the following directory structure containing the extracted NetCDF files:

```
tide_models/fes2014/load_tide/
    |- 2n2.nc
    |- ...
    |- t2.nc
tide_models/fes2014/ocean_tide/
    |- 2n2.nc
    |- ...
    |- t2.nc
```

> **Note**: If you downloaded `ocean_tide_extrapolated.tar.xz`, you will need to rename the extracted `ocean_tide_extrapolated` folder to `ocean_tide`.

### Modelling tides

If the `pyTMD` Python package is not installed on your system, this can be installed using:

```
pip install pytmd
```

Tides can now be modelled using the `model_tides` function from the `dea_tools.coastal` package:

```
import pandas as pd
from dea_tools.coastal import model_tides

lons=[155, 160]
lats=[-35, -36]
example_times = pd.date_range("2022-01-01", "2022-01-04", freq="1D")

model_tides(
        x=lons,
        y=lats,
        time=example_times.values,
        directory='tide_models'
)
```

In the DEA Coastlines code, this function is used in the `raster` module as part of the annual water index composite generation process.

Depending on where you created the `tide_models` directory, you may need to update the `directory` parameter of the `model_tide_points` function to point to the location of the FES2014 model files.
