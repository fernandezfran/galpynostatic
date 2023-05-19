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
        See the Notes in :ref:`galpynostatic.model` to know the restrictions
        of this dataframe.

    Attributes
    ----------
    logells_ : numpy.ndarray
        Unique :math:`\ell` possible values in the dataset.

    logxis_ : numpy.ndarray
        Unique :math:`\Xi` possible values in the dataset.

    spline_ : scipy.interpolate.RectBivariateSpline
        Bivariate spline approximation over the discrete dataset.
    """

    def __init__(self, dataset):
        self.logells_ = np.unique(dataset.l)
        self.logxis_ = np.unique(dataset.xi)

        socs = dataset.xmax.to_numpy().reshape(
            self.logells_.size, self.logxis_.size
        )[:, ::-1]

        self.spline_ = scipy.interpolate.RectBivariateSpline(
            self.logells_, self.logxis_, socs
        )

    def _mask_logell(self, logell):
        """Mask the value between the extrems of the interval."""
        return np.logical_and(
            np.greater_equal(logell, self.logells_.min()),
            np.less_equal(logell, self.logells_.max()),
        )

    def _mask_logxi(self, logxi):
        """Mask the value between the extrems of the interval."""
        return np.logical_and(
            np.greater_equal(logxi, self.logxis_.min()),
            np.less_equal(logxi, self.logxis_.max()),
        )

    def soc(self, logell, logxi, grid=False):
        r"""Predicts the maximum values of the SOC with the spline.

        This is a linear function of the spline bounded in [0, 1], values
        exceeding this range are taken to the corresponding endpoint.

        Parameters
        ----------
        logell : numpy.ndarray
            Log 10 value of :math:`\ell` parameter.

        logxi : numpy.ndarray
            Log 10 value of :math:`\Xi` parameter.

        grid : bool, default=False
            Whether to evaluate the results on a grid spanned by the input
            arrays, or at points specified by the input arrays, i.e. True
            is a grid and False only the ordered pairs.

        Returns
        -------
        soc : numpy.ndarray
            The corresponding maximum SOC values in the surface spline.
        """
        return np.clip(self.spline_(logell, logxi, grid=grid), 0, 1)
