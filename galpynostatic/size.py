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

"""Predict optimal size characteristics of the charging electrode material."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

from .utils import flogxi

# ============================================================================
# FUNCTIONS
# ============================================================================


def predict_length(greg, minutes=15, loaded=0.8, dlogell=0.01, cm_to=10000):
    r"""Predict the characteristic diffusion length to charge in certain time.

    Once a galvanostatic model was fitted, the kinetic information in :math:`D`
    and :math:`k^0` can be fixed and leave the characteristic diffusion length,
    `d`, free. This new free parameter only apears in :math:`\ell`, so by
    setting the value of :math:`\Xi` one can predict the SOC for a range of
    :math:`\ell` values and obtain the optimal one to get certain SOC.

    The default values of this function defines the criteria of wanting the
    80% of the electrode to be charged in 15 minutes, this is translated as a
    SOC of 0.8 and a C-rate of 4C.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted galvanostatic model.

    minutes : int or float, default=15
        Desired minutes to reach the established load.

    loaded : float, default=0.8
        Desired SOC, between 0 and 1.

    dlogell : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\ell`
        evaluation between the minimum and the maximum in the diagram.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in the defualt case to
        micrometers.

    Returns
    -------
    length : float
        The characteristic length necessary to charge the electrode to the
        desired SOC and in the desired time.
    """
    c_rate = 60.0 / minutes

    logxi = flogxi(c_rate, greg.dcoeff_, greg.k0_)
    logell_range = np.arange(
        greg._surface.ells.min(), greg._surface.ells.max(), dlogell
    )

    socs = np.array([greg._soc(logell, logxi) for logell in logell_range])

    optimal_logell = scipy.interpolate.InterpolatedUnivariateSpline(
        logell_range, socs - 0.8
    ).roots()[0]

    return cm_to * np.sqrt(
        (greg.z * greg.t_h * greg.dcoeff_ * 10.0**optimal_logell) / c_rate
    )
