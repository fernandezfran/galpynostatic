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

.. image:: https://img.shields.io/pypi/dw/galpynostatic?label=PyPI%20Downloads
   :target: https://pypistats.org/packages/galpynostatic
   :alt: pypi downloads

.. image:: https://img.shields.io/badge/python-3.8%2B-77b7fe
   :target: https://www.python.org/
   :alt: python version

.. image:: https://img.shields.io/badge/License-MIT-fcf695
   :target: https://github.com/fernandezfran/galpynostatic/blob/main/LICENSE
   :alt: mit license

.. image:: https://img.shields.io/badge/doi-TODO-b19cd9
   :target: https://www.doi.org/
   :alt: doi


**galpynostatic** is a Python package with a physics-based heuristic model to 
predict the optimal electrode particle size for a fast-charging of lithium-ion
batteries.


Requirements
------------

You need Python 3.8+ to run galpynostatic. All other dependencies, which are the 
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

   Fernandez, Francisco, et al. "Towards a fast-charging of LIBs electrode 
   materials: a heuristic model based on galvanostatic simulations" (2023). 
   _TODO_. DOI

BibTeX entry:

.. code-block:: bibtex

   @article{fernandez2023towards,
       title={Towards a fast-charging of LIBs electrode materials: a heuristic model based on galvanostatic simulations},
       author={Fernandez, Francisco and Gavilán, Maximiliano and Barraco, Daniel and Visintín, Aldo and Ein-Eli, Yair and Leiva, Ezequiel},
       year={2023}
   }


Contact
-------

You can contact me if you have any questions at ffernandev@gmail.com


.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   tutorials/index
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
