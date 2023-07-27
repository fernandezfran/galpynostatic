# Changelog of galpynostatic

# v0.1.0 (2023-07-27)

This is the first Python object-oriented version of galpynostatic.

## Features

- A galvanostatic regressor to fit maximum State-of-Charge (SOC) values versus C-rates experimental data with the physics-based heuristic model implemented here. 
- Visualization in different formats through a plotter.
- Make predictions of the optimal particle size for the fifteen-minute charging electrode material. 
- A preprocessing tool to obtain discharge capacities from galvanostatic profiles, useful to define the maximum SOC values.
- Surface datasets of the continuous computational physics previous model for different single-particle geometries. 

## Software quality assurance

- Runs on Ubuntu with Python 3.8+.
- Documentation available in readthedocs with installation guide, tutorials and API reference.
- Multiple unit tests.
- 100% coverage.
- PEP8 code style assured with flake8 and extensions.
- CI/CD on GitHub Actions.
- MIT LICENSE, encouraging its use in both academic and commercial settings.
- PyPI package distribution.
