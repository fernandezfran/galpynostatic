# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![documentation status](https://readthedocs.org/projects/galpynostatic/badge/?version=latest)](https://galpynostatic.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/galpynostatic)](https://pypi.org/project/galpynostatic/)
[![python version](https://img.shields.io/badge/python-3.12%2B-4584b6)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-ffde57)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)
[![doi](https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8)](https://doi.org/10.1016/j.electacta.2023.142951)

**galpynostatic** is a Python/C++ package with physics-based and data-driven 
models to predict optimal conditions for fast-charging lithium-ion batteries.


## Contact

If you have any questions, you can contact me at <ffernandev@gmail.com>


## Requirements

You need Python 3.12+ to run galpynostatic. All other dependencies, which are the 
usual ones of the scientific computing stack
([matplotlib](https://matplotlib.org/), [NumPy](https://numpy.org/), 
[pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) 
and [SciPy](https://scipy.org/)), are installed automatically.


## Installation

You can install the latest stable release of galpynostatic with 
[pip](https://pip.pypa.io/en/latest/)

```
python -m pip install --upgrade pip
python -m pip install --upgrade galpynostatic
```


## Usage

To learn how to use galpynostatic you can start by following the 
[tutorials](https://galpynostatic.readthedocs.io/en/latest/tutorials/index.html)
and then read the 
[API](https://galpynostatic.readthedocs.io/en/latest/api/index.html).


## License

galpynostatic is licensed under the 
[MIT License](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE).


## Citations

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the main article of the package:

> F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli and 
> E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
> heuristic model based on galvanostatic simulations." _Electrochimica Acta 464_
> (2023): 142951. DOI: https://doi.org/10.1016/j.electacta.2023.142951

For certain modules of the code, please refer to other works:
- `galpynostatic.metric`: TODO DOI
- `galpynostatic.datasets`: https://doi.org/10.1002/cphc.202200665

BibTeX entries can be found in the 
[CITATIONS.bib](https://github.com/fernandezfran/galpynostatic/blob/main/CITATIONS.bib)
file.
