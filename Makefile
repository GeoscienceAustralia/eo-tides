# This will install eo-tides and all required dependencies specified in
# `uv.lock`. To install optional dependencies, run `uv sync --all-extras`
# If running locally, can set `export VIRTUAL_ENV="path"` to control
# location of resulting virtual environment
# Optionally run --resolution lowest-direct
.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --all-extras
	@uv run pre-commit install

# Install lowest dependencies of eo-tides
.PHONY: install-lowest
install-lowest: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --all-extras --resolution lowest-direct
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry . docs/notebooks/

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
	uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml --verbose
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
    uv run python -m pytest --nbval-lax docs/notebooks/ --verbose

.PHONY: test-model
test-model: ## Test model module with pytest
	@echo "🚀 Testing model module: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
	uv run python -m pytest tests/test_model.py --verbose

.PHONY: test-eo
test-eo: ## Test eo module with pytest
	@echo "🚀 Testing eo module: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
	uv run python -m pytest tests/test_eo.py --verbose

.PHONY: test-stats
test-stats: ## Test stats module with pytest
	@echo "🚀 Testing stats module: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
	uv run python -m pytest tests/test_stats.py --verbose

.PHONY: test-utils
test-utils: ## Test utils module with pytest
	@echo "🚀 Testing utils module: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
	uv run python -m pytest tests/test_utils.py --verbose

.PHONY: test-notebooks
test-notebooks: ## Test notebooks with pytest
	@echo "🚀 Testing notebooks: Running pytest"
	@tar --skip-old-files -xzf ./tests/data/tide_models.tar.gz -C ./tests/data
	@export EO_TIDES_TIDE_MODELS=./tests/data/tide_models && \
    uv run python -m pytest --nbval-lax docs/notebooks/ --verbose

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

# On Sandbox: uv run mkdocs serve -a localhost:8000
# https://app.sandbox.dea.ga.gov.au/user/robbi.bishoptaylor@ga.gov.au/proxy/8000/eo-tides/
.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
