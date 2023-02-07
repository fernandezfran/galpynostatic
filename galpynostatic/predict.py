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

# ============================================================================
# FUNCTIONS
# ============================================================================


def t_minutes_length(greg, minutes=5, loaded=0.8, dlogell=0.01, cm_to=10000):
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
        The delta for the decrease of the logarithm value in base 10 of the
        :math:`\ell` value.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in this case to
        micrometers.

    Returns
    -------
    length : float
        The characteristic length necessary to charge the battery to the
        desired percentage and in the desired time.

    Raises
    ------
    ValueError
        If the SOC was not found to be greater than loaded and the value of the
        logarithm in base 10 of :math:`\ell` is less than the minimum at which
        the spline was fitted.
    """
    c_rate = 60.0 / minutes

    logxi = greg._logxi(c_rate)

    optlogell, soc = greg._logell(c_rate), 0.0
    while soc < loaded:
        optlogell -= dlogell
        soc = greg._soc_approx(optlogell, logxi)
        if optlogell < np.min(greg._surface.ells):
            raise ValueError(
                "It was not possible to find the optimum value for the "
                "length given the established conditions and the range of "
                "the surface used."
            )

    length = cm_to * np.sqrt(
        (greg.z * greg.t_h * greg.dcoeff_ * 10.0**optlogell) / c_rate
    )

    return length
