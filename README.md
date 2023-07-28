# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![coverage status](https://coveralls.io/repos/github/fernandezfran/galpynostatic/badge.svg)](https://coveralls.io/github/fernandezfran/galpynostatic)
[![documentation status](https://readthedocs.org/projects/galpynostatic/badge/?version=latest)](https://galpynostatic.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/galpynostatic)](https://pypi.org/project/galpynostatic/)
[![python version](https://img.shields.io/badge/python-3.8%2B-4584b6)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-ffde57)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8)](https://doi.org/10.1016/j.electacta.2023.142951)

**galpynostatic** is a Python package with a physics-based heuristic model to 
predict the optimal electrode particle size for a fast-charging of lithium-ion
batteries.


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
python -m pip install -U pip
python -m pip install -U galpynostatic
```


## Usage

To learn how to use galpynostatic you can start by following the 
[tutorials](https://galpynostatic.readthedocs.io/en/latest/tutorials/index.html)
and then read the [API](https://galpynostatic.readthedocs.io/en/latest/api.html).

Also, you can read the Jupyter Notebook pipeline in the
[paper folder](https://github.com/fernandezfran/galpynostatic/tree/main/paper) 
to reproduce the results of the published article.


## License

galpynostatic is under 
[MIT License](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE).


## Citation

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following 
[article](https://www.doi.org/10.1016/j.electacta.2023.142951)

> F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli, 
> E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
> heuristic model based on galvanostatic simulations" (2023). _Electrochimica 
> Acta_

BibTeX entry:

```bibtex
@article{fernandez2023towards,
    title = {Towards a fast-charging of LIBs electrode materials: a heuristic model based on galvanostatic simulations},
    journal = {Electrochimica Acta},
    pages = {142951},
    year = {2023},
    issn = {0013-4686},
    doi = {https://doi.org/10.1016/j.electacta.2023.142951},
    url = {https://www.sciencedirect.com/science/article/pii/S001346862301126X},
    author = {F. Fernandez and E.M. Gavilán-Arriazu and D.E. Barraco and A. Visintin and Y. Ein-Eli and E.P.M. Leiva},
    keywords = {Fast-charging, Lithium-Ion Battery, Heuristic Model, Galvanostatic charge},
    abstract = {Fast charging is one of the most important features to be accomplished for the improvement of electric vehicles. In the search for optimal use of active materials for this aim, we present a recipe to find the conditions for fast charging, fifteen minutes for 80 % of the State-of-Charge, of lithium-ion battery's single particle electrodes, thus taking advantage of the maximum possible capacity. A guide based on a general model that considers diffusion and charge transfer limitations under constant current is proposed. This guide was constructed on the basis of our previous theoretical development. A Python free and user-friendly package is provided to handle all experimental data processing and estimations.}
}
```


## Contact

You can contact me if you have any questions at <ffernandev@gmail.com>
