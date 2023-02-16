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

"""A module with the function to identify the t minutes charging length."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

# ============================================================================
# FUNCTIONS
# ============================================================================


def t_minutes_length(greg, minutes=15, loaded=0.8, dlogell=0.01, cm_to=10000):
    r"""Obtain the characteristic diffusion length to charge in t minutes.

    Once a galvanostatic model was fitted, the :math:`D_0` and :math:`k_0`
    parameters can be fixed and leave the characteristic diffusion length free,
    d, which only apears in the :math:`\ell` parameters, so by setting the
    value of :math:`\Xi` one can decrease the value of :math:`\ell` until it
    reaches a SOC that is greater than a certain desired value for a particular
    C-rate.

    The default values of this function defines the criteria of wanting the
    80% of the electrode to be charged in 5 minutes, this is translated as a
    SOC of 0.8 and a C-rate of 4C.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted galvanostatic model.

    minutes : int or float, default=5
        Desired minutes to reach the established load.

    loaded : float, default=0.8
        Desired State of Charge (SOC), between 0 and 1.

    dlogell : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\ell`
        evaluation.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in this case to
        micrometers.

    Returns
    -------
    length : float
        The characteristic length necessary to charge the battery to the
        desired percentage and in the desired time.
    """
    c_rate = 60.0 / minutes

    logxi = greg._logxi(c_rate)
    logell_rng = np.arange(-4, 2, dlogell)

    socs = np.array([greg._soc_approx(logell, logxi) for logell in logell_rng])

    optlogell = scipy.interpolate.InterpolatedUnivariateSpline(
        logell_rng, socs - 0.8
    ).roots()[0]

    length = cm_to * np.sqrt(
        (greg.z * greg.t_h * greg.dcoeff_ * 10.0**optlogell) / c_rate
    )

    return length
