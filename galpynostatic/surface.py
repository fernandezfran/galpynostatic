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
    r"""Spline of the `dataset` discrete surface.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with a diagram of SOC as function of :math:`\ell` and
        :math:`\Xi` parameters, this can be loaded using the functions in
        :ref:`galpynostatic.datasets`.

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

    def soc(self, logell, logxi):
        r"""Get the SOC value given the surface spline.

        This is a linear function bounded in [0, 1], values exceeding this
        range are taken to the corresponding end point.

        Parameters
        ----------
        logell : float
            Log 10 value of :math:`\ell` parameter.

        logxi : float
            Log 10 value of :math:`\Xi` parameter.

        Returns
        -------
        soc : float
            The corresponding soc value in the surface spline.
        """
        return np.clip(self.spline(logell, logxi)[0][0], 0, 1)
