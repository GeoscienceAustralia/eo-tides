# Installing `eo-tides`

`eo-tides` can be installed into your Python environment from [PyPI](https://pypi.org/project/eo-tides/) using `pip`:

```
pip install eo-tides
```

By default, only essential package dependencies are installed. To install all packages required for running the included Jupyter Notebook examples (including `odc-stac` and `pystac-client` for loading freely available satellite data):

```
pip install eo-tides[notebooks]
```

Once you have installed `eo-tides`, you will need to [download and set up at least one tide model](../setup/) before you can model tides.
