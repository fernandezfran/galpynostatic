# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![pypi downloads](https://img.shields.io/pypi/dw/galpynostatic?label=PyPI%20Downloads)](https://pypistats.org/packages/galpynostatic)
[![python version](https://img.shields.io/badge/python-3.8%2B-77b7fe)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-fcf695)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/doi-TODO-b19cd9)](https://www.doi.org/)

**galpynostatic** is a Python package with a physics-based heuristic model for 
identifying the five minutes charging electrode material.


## Requirements

You need Python 3.8+ to run galpynostatic. All other dependencies, which are the 
usual ones of the scientific computing stack
([matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/), 
[pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) 
and [SciPy](https://scipy.org/)), are installed automatically.


## Installation

You can install the most recent stable release of galpynostatic with 
[pip](https://pip.pypa.io/en/latest/)

```
pip install galpynostatic
```

or you can install the development version directly from this
[repo](https://github.com/fernandezfran/galpynostatic)
```
pip install git+https://github.com/fernandezfran/galpynostatic
```


## Quickstart

```python
import galpynostatic

# experimental data definition (with numpy and pandas)
C_rates, dataframes, eq_pot, xmax, d = ...

# obtain discharge capacities
xmaxs = galpynostatic.preprocessing.get_discharge_capacities(dataframes, eq_pot)

# xmaxs normalization by a maximum value
xmaxs = xmaxs / xmax

# fit the model
dataset = galpynostatic.datasets.load_spherical()
greg = galpynostatic.model.GalvanostaticRegressor(dataset, d, 3)
greg.fit(C_rates, xmaxs)

# get the diffusion coefficient and the kinetic rate constant
dcoeff = greg.dcoeff_
k0 = greg.k0_

# and an estimation of the characteristic diffusion length to charge the 80%
# of the electrode in 5 minutes
new_d = greg.t_minutes_lenght()
```

The `greg` object also allows to obtain the values predicted by the model, plot 
these predictions next to the experimental data and plot the points on the 
surface on which they were fitted. For a broader view, please refer to the 
[documentation tutorials](https://galpynostatic.readthedocs.io/en/latest/tutorial/index.html)
or the [examples](https://github.com/fernandezfran/galpynostatic/tree/main/examples).



## License

galpynostatic is under 
[MIT License](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE).


## Citation

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following article:

> Autores. (Año). Título. _Revista_. DOI

Bibtex entry:

```bibtex
    @article{
        ...
    }
```


## Contact

For bugs or questions, please send an email to <fernandezfrancisco2195@gmail.com>
