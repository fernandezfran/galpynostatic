#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Make predictions using the physics-based heuristic model.

The physics-based heuristic model [1]_ presented in this package allows to fit
State-of-Charge (SOC) battery data as a function of galvanostatic charging
rate (C-rate). Once this model has been fitted, the diffusion coefficient,
:math:`D`, and the kinetic rate constant, :math:`k^0`, parameters of the active
material in the electrode remain fixed. The other two parameters of the model,
the characteristic diffusion length, :math:`d`, (particle size) and the C-rate
can be varied. In this way, the model can be used to predict both the optimum
particle size for a given C-rate and the optimum charging rate for a given
particle size to achieve a desired maximum SOC.

References
----------
.. [1] F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y.
   Ein-Eli and E. P. M. Leiva. "Towards a fast-charging of LIBs electrode
   materials: a heuristic model based on galvanostatic simulations."
   `Electrochimica Acta 464` (2023): 142951.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.optimize

from .utils import logell, logxi

# ============================================================================
# CONSTANTS
# ============================================================================

VALUE_ERROR_MESSAGE = (
    "This system fails to find the desired point given conditions that may be "
    "due to map constraints or input parameters (such as `loaded`, initial "
    "estimate or keyword arguments passed to `scipy.optimize.newton`)."
)

# ============================================================================
# FUNCTIONS
# ============================================================================


def optimal_charging_rate(greg, c0=1.0, loaded=0.8, **kwargs):
    r"""Predict the optimal C-rate to reach a desired SOC.

    The default parameters of this function predict the C-rate required to
    reach the 80% of the electrode charge.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted GalvanostaticRegressor.

    c0 : float, default=4.0
        An initial estimate of the optimal charging rate that should be
        somewhere near the actual prediction.

    loaded : float, default=0.8
        Desired maximum SOC, between 0 and 1.

    **kwargs
        Additional keyword arguments that are passed and are documented in
        ``scipy.optimize.newton``.

    Returns
    -------
    c_rate : float
        The optimal galvanostatic charging rate to charge the electrode to the
        desired maximum SOC value.

    c_rate_err : float, optional
        The uncertainty is only returned if `greg.dcoeff_err_` and
        `greg.k0_err_` are both defined.

    Raises
    ------
    ValueError
        If the material does not meet the defined criterion given the input
        parameters or map constraints.
    """

    def objfunc(cr, greg, loaded):
        return greg.predict(np.reshape([cr], (-1, 1)))[0] - loaded

    try:
        c_rate = scipy.optimize.newton(
            objfunc, c0, args=(greg, loaded), **kwargs
        )
    except (RuntimeError, ValueError):
        raise ValueError(VALUE_ERROR_MESSAGE)

    if greg.dcoeff_err_ is None or greg.k0_err_ is None:
        return c_rate

    else:
        optimal_xi = 10.0 ** logxi(c_rate, greg.dcoeff_, greg.k0_)
        frac = greg.k0_ / greg.dcoeff_
        c_rate_err = (60.0 * np.sqrt(frac) / optimal_xi) * np.hypot(
            frac * greg.dcoeff_err_, 2 * greg.k0_err_
        )

        return (c_rate, c_rate_err)


def optimal_particle_size(
    greg, d0=1e-4, loaded=0.8, c_rate=4.0, cm_to=10_000, **kwargs
):
    r"""Predict the optimal electrode particle size to charge in certain time.

    The default parameters of this function define the criteria of reaching 80%
    of the electrode charge in 15 minutes, which corresponds to a maximum SOC
    of 0.8 and a C-rate of 4C, which is the USABC (`United States
    Advanced Battery Consortium`) standard for fast charging.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        A GalvanostaticRegressor already fitted.

    d0 : float, default=1e-4
        An initial estimate of the optimal particle size rate that should be
        somewhere near the actual prediction.

    loaded : float, default=0.8
        Desired maximum SOC value, between 0 and 1.

    c_rate : int or float, default=4.0
        Desired C-rate to reach the established SOC.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in the default case to
        microns.

    **kwargs
        Additional keyword arguments that are passed and are documented in
        ``scipy.optimize.newton``.

    Returns
    -------
    particle_size : float
        The optimal particle size to charge the electrode to the desired
        maximum SOC value in the desired time.

    particle_size_err : float, optional
        The uncertainty is only returned if `greg.dcoeff_err_` is defined.

    Raises
    ------
    ValueError
        If the material does not meet the defined criterion given the input
        parameters or map constraints.
    """

    def objfunc(d, greg, c_rate, loaded):
        greg.d = d
        return greg.predict(np.reshape([c_rate], (-1, 1)))[0] - loaded

    try:
        particle_size = np.abs(
            scipy.optimize.newton(
                objfunc, d0, args=(greg, c_rate, loaded), **kwargs
            )
        )
    except (RuntimeError, ValueError):
        raise ValueError(VALUE_ERROR_MESSAGE)

    if greg.dcoeff_err_ is None:
        return cm_to * particle_size

    else:
        optimal_ell = 10.0 ** logell(
            c_rate, particle_size, greg.z, greg.dcoeff_
        )
        factor = np.sqrt((3600 * greg.z * optimal_ell) / c_rate)
        particle_size_err = (
            factor * greg.dcoeff_err_ / (2 * np.sqrt(greg.dcoeff_))
        )

        return (cm_to * particle_size, cm_to * particle_size_err)
