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
    """Spline of the `dataset` discrete surface.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with a map of State of Charge (SOC) as function of l and chi
        parameters, this can be loaded using :ref:`galpynostatic.datasets`
        load functions.

    Attributes
    ----------
    ls : ndarray
        Unique `l` possible values in the dataset.

    chis : ndarray
        Unique `chi` possible values in the dataset.

    spline : scipy.interpolate.RectBivariateSpline
        Bivariate spline approximation over the discrete dataset.
    """

    def __init__(self, dataset):
        self.ls = np.unique(dataset.l)
        self.chis = np.unique(dataset.chi)

        k, socs = 0, []
        for logl, logchi in it.product(self.ls, self.chis[::-1]):
            soc = 0
            try:
                if logl == dataset.l[k] and logchi == dataset.chi[k]:
                    soc = dataset.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                socs.append(soc)

        self.spline = scipy.interpolate.RectBivariateSpline(
            self.ls,
            self.chis,
            np.asarray(socs).reshape(self.ls.size, self.chis.size)[:, ::-1],
        )
