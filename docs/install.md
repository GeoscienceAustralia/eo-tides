# Installing `eo-tides`

## Stable version

The [latest stable release](https://github.com/GeoscienceAustralia/eo-tides/releases) of `eo-tides` can be installed into your Python environment from [PyPI](https://pypi.org/project/eo-tides/) using `pip`:

```console
pip install eo-tides
```

By default, only essential package dependencies are installed. To install all packages required for running the included Jupyter Notebook examples (including `odc-stac` and `pystac-client` for loading freely available satellite data), run:

```console
pip install eo-tides[notebooks]
```

## Unstable development version

To install the latest unstable development version of `eo-tides`, run:

```console
pip install git+https://github.com/GeoscienceAustralia/eo-tides.git
```

!!! important

    The unstable development version of `eo-tides` may contain bugs and untested new features. Unless you need access to a specific unpublished feature, we recommend installing the latest stable version instead.

## Next steps

Once you have installed `eo-tides`, you will need to [download and set up at least one tide model](setup.md) before you can model tides.
