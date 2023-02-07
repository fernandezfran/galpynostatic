#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Private module with the dataset surface spline."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

import scipy.interpolate

# ============================================================================
# CLASSES
# ============================================================================


class SurfaceSpline:
    r"""Spline of the `dataset` discrete surface.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with a map of State of Charge (SOC) as function of :math:`\ell`
        and :math:`\Xi` parameters, this can be loaded using load functions in
        :ref:`galpynostatic.datasets`.

    Attributes
    ----------
    ells : numpy.ndarray
        Unique :math:`\ell` possible values in the dataset.

    xis : numpy.ndarray
        Unique :math:`\Xi` possible values in the dataset.

    spline : scipy.interpolate.RectBivariateSpline
        Bivariate spline approximation over the discrete dataset.
    """

    def __init__(self, dataset):
        self.ells = np.unique(dataset.l)
        self.xis = np.unique(dataset.chi)

        k, socs = 0, []
        for logell, logxi in it.product(self.ells, self.xis[::-1]):
            soc = 0
            try:
                if logell == dataset.l[k] and logxi == dataset.chi[k]:
                    soc = dataset.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                socs.append(soc)

        self.spline = scipy.interpolate.RectBivariateSpline(
            self.ells,
            self.xis,
            np.asarray(socs).reshape(self.ells.size, self.xis.size)[:, ::-1],
        )
