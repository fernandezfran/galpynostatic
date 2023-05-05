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

"""Make predictions using the physics-based heurist model."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

from .utils import logxi

# ============================================================================
# FUNCTIONS
# ============================================================================


def optimal_particle_size(
    greg,
    minutes=15,
    loaded=0.8,
    cm_to=10000,
    dlogell=0.01,
):
    r"""Predict the optimal electrode particle size to charge in certain time.

    Once the physics-based heuristic model is fitted, the diffusion
    coefficient, :math:`D`, and the kinetic-rate constant, :math:`k^0`,
    parameters of the active material in the electrode remain fixed. The other
    two parameters of the model, the characteristic diffusion lenght,
    :math:`d`, (i.e. particle size) and the C-rate can vary. With this in mind,
    the model can be used to predict the particle size at a given C-rate to
    obtain a desired maximum State-of-Charge (SOC) value.

    The default parameters of this function define the criteria of reaching 80%
    of the electrode charge in 15 minutes, which translates into a maximum SOC
    value of 0.8 and a C-rate of 4C, which is the USABC (`United States
    Advanced Battery Consortium`) standard for fast charging.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        A heuristic model already fitted.

    minutes : int or float, default=15
        Desired minutes to reach the established load.

    loaded : float, default=0.8
        Desired maximum SOC value, between 0 and 1.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in the default case to
        micrometers.

    dlogell : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\ell`
        evaluation between the minimum and the maximum in the diagram.

    Returns
    -------
    particle_size : float
        The optimal particle size to charge the electrode to the desired
        maximum SOC value in the desired time.
    """
    c_rate = 60.0 / minutes

    logxi_value = logxi(c_rate, greg.dcoeff_, greg.k0_)
    logell_range = np.arange(
        greg._surface.logells.min(), greg._surface.logells.max(), dlogell
    )

    socs = greg._surface.soc(logell_range, logxi_value)

    optimal_logell = scipy.interpolate.InterpolatedUnivariateSpline(
        logell_range, socs - 0.8
    ).roots()[0]

    return cm_to * np.sqrt(
        (3600 * greg.z * greg.dcoeff_ * 10.0**optimal_logell) / c_rate
    )
