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


def t_minutes_lenght(greg, minutes=5, loaded=0.8, dlogl=0.01, cm_to=10000):
    """Obtain the characteristic diffusion length to charge in t minutes.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted galvanostatic model.

    minutes : int or float, default=5
        Desired minutes to reach the established load.

    loaded : float, default=0.8
        Desired charge percentage between 0 and 1.

    dlogl : float, default=0.01
        The delta for the decrease of the logarithm value in base 10 of the l
        value.

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
        If the normalized discharge capacity was not found to be greater than
        loaded and the value of the logarithm in base 10 of l is less than the
        minimum at which the spline was fitted.
    """
    c_rate = 60.0 / minutes

    logchi = greg._logchi(c_rate)

    optlogl, xmax = greg._logl(c_rate), 0.0
    while xmax < loaded:
        optlogl -= dlogl
        xmax = greg._xmax_in_surface(optlogl, logchi)
        if optlogl < np.min(greg._ls):
            raise ValueError(
                "It was not possible to find the optimum value for the "
                "length given the established conditions and the range of "
                "the surface used."
            )

    length = cm_to * np.sqrt(
        (greg.z * greg.t_h * greg.dcoeff_ * 10.0**optlogl) / c_rate
    )

    return length
