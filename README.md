# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![pypi downloads](https://img.shields.io/pypi/dw/galpynostatic?label=PyPI%20Downloads)](https://pypistats.org/packages/galpynostatic)
[![python version](https://img.shields.io/badge/python-3.8%2B-77b7fe)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-fcf695)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/doi-TODO-b19cd9)](https://www.doi.org/)

**galpynostatic** is a Python package with a physics-based heuristic model
towards the fast-charging of lithium-ion batteries electrode material.

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


## Usage

To learn how to use galpynostatic you can start by following the 
[tutorial](https://galpynostatic.readthedocs.io/en/latest/tutorial/index.html)
and then take a look at the [API](https://galpynostatic.readthedocs.io/en/latest/api.html).


## License

galpynostatic is under 
[MIT License](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE).


## Citation

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following article:

> Fernandez, Francisco, et al. "Towards the fast-charging of LIBs electrode 
> materials: a heuristic model based on galvanostatic simulations" (2023). 
> _arXiv_. DOI

BibTeX entry:

```bibtex
@article{fernandez2023towards,
  title={Towards the fast-charging of LIBs electrode materials: a heuristic model based on galvanostatic simulations},
  author={Fernandez, Francisco and Gavilán, Maximiliano and Visintín, Aldo and Barraco, Daniel and Ein-Eli, Yair and Leiva, Ezequiel},
  year={2023}
}
```

We will publish this arXiv work in a refereed journal that will supersede this citation.
