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

"""Module with the spline to the diagram surface in the dataset."""

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
    r"""Spline of the dataset discrete diagram.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with the diagram of the maximum SOC values as function of the
        internal parameters :math:`\log(\ell)` and :math:`\log(\Xi)`, this can
        be loaded using the functions of the :ref:`galpynostatic.datasets`.

    Attributes
    ----------
    logells : numpy.ndarray
        Unique :math:`\ell` possible values in the dataset.

    logxis : numpy.ndarray
        Unique :math:`\Xi` possible values in the dataset.

    spline : scipy.interpolate.RectBivariateSpline
        Bivariate spline approximation over the discrete dataset.
    """

    def __init__(self, dataset):
        self.logells = np.unique(dataset.l)
        self.logxis = np.unique(dataset.chi)

        k, socs = 0, []
        for logell, logxi in it.product(self.logells, self.logxis[::-1]):
            soc = 0
            try:
                if logell == dataset.l[k] and logxi == dataset.chi[k]:
                    soc = dataset.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                socs.append(soc)
        socs = np.array(socs)

        self.spline = scipy.interpolate.RectBivariateSpline(
            self.logells,
            self.logxis,
            socs.reshape(self.logells.size, self.logxis.size)[:, ::-1],
        )

    def _mask_logell(self, logell):
        """Mask the value between the extrems of the interval."""
        return np.logical_and(
            np.greater_equal(logell, self.logells.min()),
            np.less_equal(logell, self.logells.max()),
        )

    def _mask_logxi(self, logxi):
        """Mask the value between the extrems of the interval."""
        return np.logical_and(
            np.greater_equal(logxi, self.logxis.min()),
            np.less_equal(logxi, self.logxis.max()),
        )

    def soc(self, logell, logxi):
        r"""Predicts the maximum values of the SOC with the spline.

        This is a linear function of the spline bounded in [0, 1], values
        exceeding this range are taken to the corresponding endpoint.

        Parameters
        ----------
        logell : numpy.ndarray
            Log 10 value of :math:`\ell` parameter.

        logxi : numpy.ndarray
            Log 10 value of :math:`\Xi` parameter.

        Returns
        -------
        soc : numpy.ndarray
            The corresponding maximum SOC values in the surface spline.
        """
        return np.clip(self.spline(logell, logxi, grid=False), 0, 1)
