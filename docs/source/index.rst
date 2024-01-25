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

.. image:: https://img.shields.io/badge/doi-10.1016/j.electacta.2023.142951-36abe8
   :target: https://doi.org/10.1016/j.electacta.2023.142951
   :alt: doi


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


Citation
--------

If you use galpynostatic in a scientific publication, we would appreciate it if 
you could cite the following article:

.. pull-quote::

   F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y. Ein-Eli 
   and E. P. M. Leiva. "Towards a fast-charging of LIBs electrode materials: a 
   heuristic model based on galvanostatic simulations." `Electrochimica Acta 464`
   (2023): 142951.

BibTeX entry:

.. code-block:: bibtex

   @article{fernandez2023towards,
     title={Towards a fast-charging of LIBs electrode materials: a heuristic model based on galvanostatic simulations},
     author={Fernandez, F and Gavil{\'a}n-Arriazu, EM and Barraco, DE and Visintin, A and Ein-Eli, Y and Leiva, EPM},
     journal={Electrochimica Acta},
     volume={464},
     pages={142951},
     year={2023},
     publisher={Elsevier}
   }

Other related citations can be found in the 
`CITATION.bib <https://github.com/fernandezfran/galpynostatic/blob/main/CITATION.bib>`__ 
file.

Contact
-------

If you have any questions, you can contact me at ffernandev@gmail.com


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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
