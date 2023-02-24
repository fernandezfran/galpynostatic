# Changelog of galpynostatic

## Version 0.1

This first object-oriented version contains the 
`galpynostatic.model.GalvanostaticRegressor` to fit SOC versus C-rates 
experimental data to the physics-based heuristic model, with the surface data 
available in `galpynostatic.datasets`. Allows visualization through an accessor 
to `galpynostatic.plot.GalvanostaticPlotter` and make predictions on the optimal 
size characteristic of the fifteen-minute charging electrode material with
`galpynostatic.size.predict_length`. It also offers the 
`galpynostatic.preprocessing.GetDischargeCapacities` class to obtain discharge 
capacities from galvanostatic profiles.

The software quality is the following:
    - runs on Ubuntu with Python 3.8+.
    - documentation with installation guide, tutorials with examples of use with 
    common and state-of-the-art research lithium battery systems, and API in 
    readthedocs.
    - multiple unit tests.
    - 100% coverage.
    - PEP8 style code assured with flake8 and extensions.
    - CI/CD on GitHub Actions.
    - MIT LICENSE.
    - available as a package in PyPI.
