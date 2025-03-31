# Installing `eo-tides`

## Stable version

The latest stable release of `eo-tides` is:

[![PyPI](https://img.shields.io/pypi/v/eo-tides)](https://pypi.org/project/eo-tides/)

It is compatible with the following Python versions:

[![Python Version from PEP 621 TOML](https://img.shields.io/pypi/pyversions/eo-tides)](https://github.com/GeoscienceAustralia/eo-tides/blob/main/pyproject.toml)

`eo-tides` can be installed into your Python environment using either `pip` (the standard Python package manager), or `uv` ([a fast Python package manager](https://docs.astral.sh/uv/) written in Rust).

!!! tip

    We recommend installing with `uv` as this makes it easy to set up an isolated environment containing compatible Python packages.

=== "Installing with `pip`"

    Install `eo-tides`:
    ```console
    python3 -m pip install eo-tides[notebooks]
    ```

    The code above will also install [additional package dependencies](https://github.com/GeoscienceAustralia/eo-tides/blob/main/pyproject.toml#L62-L69) required for running [Jupyter Notebook code examples](notebooks/Model_tides.ipynb) (including `odc-stac` and `pystac-client` for loading satellite data).

    To install `eo-tides` with only essential package dependencies:

    ```console
    python3 -m pip install eo-tides
    ```

=== "Installing with `uv`"

    First, [install `uv` using the method of your choice](https://docs.astral.sh/uv/getting-started/installation/). Then:

    Create a new virtual environment with your chosen version of Python:
    ```console
    uv venv --python 3.13
    ```

    Activate your virtual environment:
    === "Linux and macOS"
        ```console
        source .venv/bin/activate
        ```
    === "Windows"
        ```console
        .venv\Scripts\activate
        ```

    Install `eo-tides` into your virtual environment:

    ```console
    uv pip install eo-tides[notebooks]
    ```

    The code above will also install [additional package dependencies](https://github.com/GeoscienceAustralia/eo-tides/blob/main/pyproject.toml#L62-L69) required for running [Jupyter Notebook code examples](notebooks/Model_tides.ipynb) (including `odc-stac` and `pystac-client` for loading satellite data).

    To install `eo-tides` with only essential package dependencies:
    ```console
    uv pip install eo-tides
    ```

    You are now ready to run code using `eo-tides`! For example, you could launch your virtual environment with JupyterLab to run the [example Jupyter Notebooks](notebooks/Model_tides.ipynb):
    ```console
    uv run --with jupyter jupyter lab
    ```

## Unstable development pre-releases

Unstable development pre-releases of `eo-tides` are also available:

[![PyPI](https://img.shields.io/badge/pypi-releases-f48041)](https://pypi.org/project/eo-tides/#history)

=== "Installing with `pip`"

    Install an example `eo-tides` pre-release (see [full list here](https://pypi.org/project/eo-tides/#history)), including additional `notebooks` dependencies:
    ```console
    python3 -m pip install eo-tides[notebooks]==0.6.3.dev5
    ```

=== "Installing with `uv`"

    First, [install `uv` using the method of your choice](https://docs.astral.sh/uv/getting-started/installation/). Then:

    Create a new virtual environment with your chosen version of Python:
    ```console
    uv venv --python 3.13
    ```

    Activate your virtual environment:
    === "Linux and macOS"
        ```console
        source .venv/bin/activate
        ```
    === "Windows"
        ```console
        .venv\Scripts\activate
        ```

    Install an example `eo-tides` pre-release (see [full list here](https://pypi.org/project/eo-tides/#history)), including additional `notebooks` dependencies:
    ```console
    uv pip install eo-tides[notebooks]==0.6.3.dev5
    ```

!!! important

    Unstable development pre-releases may contain bugs and untested new features. Unless you need access to a specific unpublished feature, we recommend installing the latest [stable version](#stable-version) instead.

### Advanced: Developing locally

To work on `eo-tides` locally, we recommend using `uv`.

First, [install `uv` using the method of your choice](https://docs.astral.sh/uv/getting-started/installation/).

Clone the `eo-tides` repository:

```console
git clone https://github.com/GeoscienceAustralia/eo-tides
cd eo-tides/
```

Use `uv sync` to populate an environment with package dependencies from the `eo-tides` [lockfile](https://github.com/GeoscienceAustralia/eo-tides/blob/main/uv.lock):

```console
uv sync --all-extras
```

Set up pre-commit hooks:

```console
uv run pre-commit install
```

We provide a [pre-prepared Makefile](https://github.com/GeoscienceAustralia/eo-tides/blob/main/Makefile) that you can use to easily run common tasks:

```console
# Run code quality checks
make check

# Run pytest tests
make test

# Run Jupyter notebook tests
make test-notebooks

# Build and preview documentation
make docs
```

## Next steps

Once you have installed `eo-tides`, you will need to [download and set up at least one tide model](setup.md) before you can model tides.
