# Changelog of galpynostatic

## Version 0.1

First object-oriented version:

- `galpynostatic.datasets` with discrete surface data from a computational 
physics continuum model for different geometries.
- `galpynostatic.preprocessing.GetDischargeCapacities` to obtain discharge 
capacities from galvanostatic profiles in a `sklearn.base.TransformerMixin` way.
- `galpynostatic.model.GalvanostaticRegressor` to fit experimental data to the 
model, predict and plot through an accessor to 
`galpynostatic.plot.GalvanostaticPlotter`.
- `galpynostatic.predict.t_minutes_length` to estimate the characteristic 
diffusion length to charge the electrode material in t minutes.
- examples of use with common and state-of-the-art research lithium battery systems.
- software quality:
    - runs on Ubuntu with Python 3.8+
    - documentation with installation guide, tutorial and API in readthedocs.
    - multiple unit tests.
    - 100% coverage.
    - PEP8 style code assured with flake8 and extensions.
    - CI/CD on GitHub Actions.
    - MIT LICENSE.
    - available as a package in PyPI.
