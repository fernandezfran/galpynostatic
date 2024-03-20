# Changelog of galpynostatic

## v0.4.0 (2023-03-20)

- Change the name of the metric `umbem` (thesis version) to `bmxfc` (paper version).


## v0.3.3 (2024-02-05)

- Fix a bug in the shape of the return of `GetDischargeCapacities` in `preprocessing` module.
- Change the `if/else` condition in the `datasets` submodule by a `try/except` block.


## v0.3.2 (2024-01-30)

- `base` and `utils` modules are now tested directly.
- Better docs of `datasets` submodule and the three functions replaced in only one. For each assert now there is a test.


## v0.3.1 (2024-01-29)

- Include the optimal C-rate uncertainty calculation like in particle size.
- Changed the inner workings of functions in the `make_prediction` module to use `GalvanostaticRegressor` model methods instead of geometric reordering to find the optimal point.
- Add `**kwargs` to `scipy.optimize.newton` in `make_prediction` module.


## v0.3.0 (2024-01-25)

- Add a new function in the `make_prediction` module to predict the optimal C-rate to reach a desired SOC.
- Explicitly use Newton optimization method in `make_prediction` module.
- Change the return of the transform `GetDischargeCapacities` in `preprocessing` module to the shape required for the model fitting.
- Improved self-consistency and grammar of documentation.


## v0.2.3 (2023-12-28)

- Create the `base` module with the `MapSpline` class.
- Change the project description.


## v0.2.2 (2023-12-27)

- Change the name of the `bmx_fc` metric to `umbem`, which is the name of the metric in the cited PhD thesis.
- Replace the `test_metric` to use `pytest.mark.parametrize` and have a test for each value insted of the dataframe all togheter.


## v0.2.1 (2023-12-26)

- Allow the modification of C-rate with minutes parameter in `bmx_fc` of `metric` module.
- Citation of theoretical framework in `CITATION.bib` file.


## v0.2.0 (2023-12-12)

- An implementation of a new module with two metrics for benchmarking an extreme fast-charging of battery electrode materials.
- Fixed test errors in `make_prediction` module due to uncertaintes calculations.


## v0.1.1 (2023-09-25)

- Fixed the citation link, the BibTeX file and the doi.
- The test of the plots with Python3.9+ instead of Python3.8
- Fixed the uncertainties calculations.


## v0.1.0 (2023-07-27)

This is the first Python object-oriented version of galpynostatic.

### Features

- A galvanostatic regressor to fit maximum State-of-Charge (SOC) values versus C-rates experimental data with the physics-based heuristic model implemented here. 
- Visualization in different formats through a plotter.
- Make predictions of the optimal particle size for the fifteen-minute charging electrode material. 
- A preprocessing tool to obtain discharge capacities from galvanostatic profiles, useful to define the maximum SOC values.
- Surface datasets of the continuous computational physics previous model for different single-particle geometries. 

### Software quality assurance

- Runs on Ubuntu with Python 3.8+.
- Documentation available in readthedocs with installation guide, tutorials and API reference.
- Multiple unit tests.
- 100% coverage.
- PEP8 code style assured with flake8 and extensions.
- CI/CD on GitHub Actions.
- MIT LICENSE, encouraging its use in both academic and commercial settings.
- PyPI package distribution.
