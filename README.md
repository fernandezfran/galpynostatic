# galpynostatic

[![galpynostatics CI](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg)](https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml)
[![documentation status](https://readthedocs.org/projects/galpynostatic/badge/?version=latest)](https://galpynostatic.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/galpynostatic)](https://pypi.org/project/galpynostatic/)
[![python version](https://img.shields.io/badge/python-3.9%2B-4584b6)](https://www.python.org/)
[![mit license](https://img.shields.io/badge/License-MIT-ffde57)](https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE)

**galpynostatic** is a Python package with physics-based models to predict 
optimal conditions for fast-charging lithium-ion batteries.


## Requirements

You need Python 3.9+ to run galpynostatic. All other dependencies, which are the 
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
you could cite the following article:

[![doi](https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8)](https://doi.org/10.1016/j.electacta.2023.142951)

> F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli and 
> E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
> heuristic model based on galvanostatic simulations." _Electrochimica Acta 464_
> (2023): 142951. 

For the theoretical framework and the universal map datasets 
(`galpynostatic.datasets` submodule) refer to:

[![datasets](https://img.shields.io/badge/doi-10.1002/cphc.202200665-2f4995)](https://doi.org/10.1002/cphc.202200665)

> E. M. Gavilán‐Arriazu, D. E. Barraco, Y. Ein‐Eli and E. P. M. Leiva. 
> "Galvanostatic Fast Charging of Alkali‐Ion Battery Materials at the 
> Single‐Particle Level: A Map‐Driven Diagnosis". _ChemPhysChem, 24_.6 (2023): 
> e202200665. 

BibTeX entries can be found in the 
[CITATIONS.bib](https://github.com/fernandezfran/galpynostatic/blob/main/CITATIONS.bib)
file.


## Contact

If you have any questions, you can contact me at <ffernandev@gmail.com>
