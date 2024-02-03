.. galpynostatic documentation master file, created by
   sphinx-quickstart on Wed Dec 14 16:21:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
galpynostatic
=============

.. image:: https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml/badge.svg
   :target: https://github.com/fernandezfran/galpynostatic/actions/workflows/CI.yml
   :alt: galpynostatic CI

.. image:: https://readthedocs.org/projects/galpynostatic/badge/?version=latest
   :target: https://galpynostatic.readthedocs.io/
   :alt: ReadTheDocs

.. image:: https://img.shields.io/pypi/v/galpynostatic
   :target: https://pypi.org/project/galpynostatic/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/python-3.9%2B-4584b6
   :target: https://www.python.org/
   :alt: python version

.. image:: https://img.shields.io/badge/License-MIT-ffde57
   :target: https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE
   :alt: mit license


**galpynostatic** is a Python package with physics-based models to predict
optimal conditions for fast-charging lithium-ion batteries


Requirements
------------

You need Python 3.9+ to run galpynostatic. All other dependencies, which are the 
usual ones of the scientific computing stack
(`matplotlib <https://matplotlib.org/>`__, `NumPy <https://numpy.org/>`__, 
`pandas <https://pandas.pydata.org/>`__, `scikit-learn <https://scikit-learn.org>`__
and `SciPy <https://scipy.org/>`__), are installed automatically.


Code Repository
---------------

https://github.com/fernandezfran/galpynostatic/


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   install
   tutorials/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/api


Citations
---------

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following article:

.. image:: https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8
   :target: https://doi.org/10.1016/j.electacta.2023.142951
   :alt: doi

.. pull-quote::

   F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli 
   and E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
   heuristic model based on galvanostatic simulations." `Electrochimica Acta 464`
   (2023): 142951.

For the theoretical framework and the universal map datasets 
(``galpynostatic.datasets`` submodule) refer to:

.. image:: https://img.shields.io/badge/doi-10.1002/cphc.202200665-2f4995
   :target: https://doi.org/10.1002/cphc.202200665
   :alt: datasets

.. pull-quote::

   E. M. Gavilán‐Arriazu, D. E. Barraco, Y. Ein‐Eli and E. P. M. Leiva. 
   "Galvanostatic Fast Charging of Alkali‐Ion Battery Materials at the 
   Single‐Particle Level: A Map‐Driven Diagnosis". `ChemPhysChem, 24`.6 (2023): 
   e202200665.


BibTeX entries can be found in the 
`CITATIONS.bib <https://github.com/fernandezfran/galpynostatic/blob/main/CITATIONS.bib>`__ 
file.

Contact
-------

If you have any questions, you can contact me at ffernandev@gmail.com


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
