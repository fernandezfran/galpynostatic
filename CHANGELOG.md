# Changelog of galpynostatic

## Version 0.1

First object-oriented version:

- `galpynostatic.datasets` with discrete surface data from a computational physics continuum model for different geometries.
- `galpynostatic.preprocessing.get_discharge_capacities` to obtain discharge capacities from galvanostatic profiles.
- `galpynostatic.model.GalvanostaticRegressor` to fit experimental data to the model, predict and plot.
- examples of use with common and state-of-the-art research lithium battery systems.
- software quality:
    - runs on Ubuntu with Python 3.8+
    - documentation with installation guide, tutorial and API in readthedocs.
    - multiple unit tests.
    - 100% coverage.
    - PEP8 style code assured with flake8 and extensions.
    - Continuous integration in GitHub Actions.
    - MIT LICENSE
    - available as a package in PyPI
