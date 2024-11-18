# Installing `eo-tides`

## Stable version

[![Release](https://img.shields.io/github/v/release/GeoscienceAustralia/eo-tides)](https://pypi.org/project/eo-tides/)
[![Python Version from PEP 621 TOML](https://img.shields.io/pypi/pyversions/eo-tides)](https://github.com/GeoscienceAustralia/eo-tides/blob/main/pyproject.toml)

The [latest stable release](https://pypi.org/project/eo-tides/) of `eo-tides` can be installed into your Python environment from [PyPI](https://pypi.org/project/eo-tides/) using `pip`:

```console
python3 -m pip install eo-tides
```

By default, only essential package dependencies are installed. To install all packages required for running the included Jupyter Notebook examples (including `odc-stac` and `pystac-client` for loading freely available satellite data), run:

```console
python3 -m pip install eo-tides[notebooks]
```

## Unstable development version

To install the latest unstable development version of `eo-tides` directly from Github, run:

```console
python3 -m pip install git+https://github.com/GeoscienceAustralia/eo-tides.git
```

!!! important

    Unstable development versions of `eo-tides` may contain bugs and untested new features. Unless you need access to a specific unpublished feature, we recommend installing the latest stable version instead.

### Cloning locally

To clone the `eo-tides` repository locally:

```console
git clone https://github.com/GeoscienceAustralia/eo-tides.git
```

Navigate to the project directory and install in editable mode from your local copy of the repository:

```console
cd eo-tides
python3 -m pip install -e .
```

## Next steps

Once you have installed `eo-tides`, you will need to [download and set up at least one tide model](setup.md) before you can model tides.
