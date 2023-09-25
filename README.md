# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![documentation status](https://readthedocs.org/projects/galpynostatic/badge/?version=latest)](https://galpynostatic.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/galpynostatic)](https://pypi.org/project/galpynostatic/)
[![python version](https://img.shields.io/badge/python-3.9%2B-4584b6)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-ffde57)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8)](https://doi.org/10.1016/j.electacta.2023.142951)

**galpynostatic** is a Python package with a physics-based heuristic model to 
predict the optimal electrode particle size for a fast-charging of lithium-ion
batteries.


## Requirements

You need Python 3.9+ to run galpynostatic. All other dependencies, which are the 
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
and then read the 
[API](https://galpynostatic.readthedocs.io/en/latest/api/index.html).

Also, you can read the Jupyter Notebook pipeline in the
[paper folder](https://github.com/fernandezfran/galpynostatic/tree/main/paper) 
to reproduce the results of the published article.


## License

galpynostatic is under 
[MIT License](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE).


## Citation

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following 
[article](https://doi.org/10.1016/j.electacta.2023.142951)

> F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli and 
> E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
> heuristic model based on galvanostatic simulations." _Electrochimica Acta 464_
> (2023): 142951.

BibTeX entry:

```bibtex
@article{fernandez2023towards,
  title={Towards a fast-charging of LIBs electrode materials: a heuristic model based on galvanostatic simulations},
  author={Fernandez, F and Gavil{\'a}n-Arriazu, EM and Barraco, DE and Visintin, A and Ein-Eli, Y and Leiva, EPM},
  journal={Electrochimica Acta},
  volume={464},
  pages={142951},
  year={2023},
  publisher={Elsevier}
}
```


## Contact

You can contact me if you have any questions at <ffernandev@gmail.com>
