#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Make predictions using the physics-based heuristic model.

Once the physics-based heuristic model [1]_ has been fitted, the diffusion
coefficient, :math:`D`, and the kinetic rate constant, :math:`k^0`, parameters
of the active material in the electrode remain fixed. The other two parameters
of the model, the characteristic diffusion length, :math:`d`, (particle size)
and the galvanostatic charging rate (C-rate) can be varied. In this way, the
model can be used to predict both the optimum particle size for a given C-rate
and the optimum charging rate for a given particle size to achieve a desired
maximum State-of-Charge (SOC).

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

import scipy.interpolate
import scipy.optimize

from .utils import logxi

# ============================================================================
# FUNCTIONS
# ============================================================================


def optimal_charging_rate(
    greg, loaded=0.8, unit="C-rate", dlogell=0.01, dlogxi=0.01
):
    r"""Predict the optimal C-rate to reach a desired SOC.

    The default parameters of this function predict the C-rate required to
    reach the 80% of the electrode charge.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted GalvanostaticRegressor.

    loaded : float, default=0.8
        Desired maximum SOC value, between 0 and 1.

    unit : str, default='C-rate'
        The desired unit of the return value, it can be `"C-rate"` or
        `"minutes"`.

    dlogxi : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\Xi`
        parameter evaluation between the minimum and the maximum in the map.

    dlogell : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\ell`
        parameter evaluation between the minimum and the maximum in the map.

    Returns
    -------
    float
        The optimal galvanostatic charging rate in C-rate or minutes units.

    Raises
    ------
    ValueError
        If the material does not meet the defined criterion given the map
        constraints.
    """
    intercept = np.log10(
        (greg.k0_ * greg.d) / (greg.dcoeff_ * np.sqrt(greg.z))
    )

    logell_min, logell_max = greg._map.logells_.min(), greg._map.logells_.max()
    logell_range = np.arange(logell_min, logell_max, dlogell)

    logxi_range = intercept - 0.5 * logell_range
    logxi_min, logxi_max = logxi_range.min(), logxi_range.max()

    socs = greg._map.soc(logell_range, logxi_range) - loaded

    dell = logell_max - logell_min
    dxi = logxi_max - logxi_min
    angle = np.arctan(dxi / dell)
    hypot = np.hypot(dell, dxi)

    h_range = np.linspace(0, hypot, socs.size)

    spline = scipy.interpolate.InterpolatedUnivariateSpline(h_range, socs)

    try:
        optimal_h = scipy.optimize.newton(lambda h: spline(h), hypot / 2)
    except RuntimeError:
        raise ValueError(
            "This material does not reach the desired SOC for a C-rate that "
            "is between the map constaints."
        )

    optimal_logell = logell_min + optimal_h * np.cos(angle)
    optimal_logxi = logxi_max - optimal_h * np.sin(angle)

    c1 = (3600 * (greg.k0_) ** 2) / (greg.dcoeff_ * 10 ** (2 * optimal_logxi))
    c2 = (3600 * greg.dcoeff_ * greg.z * 10**optimal_logell) / (greg.d**2)

    c_rate = np.mean([c1, c2])

    return c_rate if unit == "C-rate" else 60 / c_rate


def optimal_particle_size(
    greg,
    minutes=15,
    loaded=0.8,
    cm_to=10_000,
    dlogell=0.01,
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

    minutes : int or float, default=15
        Desired minutes to reach the established SOC.

    loaded : float, default=0.8
        Desired maximum SOC value, between 0 and 1.

    cm_to : float, default=10000
        A factor to convert from cm to another unit, in the default case to
        microns.

    dlogell : float, default=0.01
        The delta for the logarithm value in base 10 of the :math:`\ell`
        parameter evaluation between the minimum and the maximum in the map.

    Returns
    -------
    particle_size : float
        The optimal particle size to charge the electrode to the desired
        maximum SOC value in the desired time.

    particle_size_err : float
        The uncertainty is only returned if `greg.dcoeff_err_` is defined.

    Raises
    ------
    ValueError
        If the material does not meet the defined criterion given the map
        constraints.
    """
    c_rate = 60.0 / minutes

    logxi_value = logxi(c_rate, greg.dcoeff_, greg.k0_)
    logell_range = np.arange(
        greg._map.logells_.min(), greg._map.logells_.max(), dlogell
    )

    socs = greg._map.soc(logell_range, logxi_value) - loaded

    spline = scipy.interpolate.InterpolatedUnivariateSpline(logell_range, socs)

    try:
        optimal_logell = scipy.optimize.newton(lambda lr: spline(lr), -0.5)

        if not greg._map._mask_logell(optimal_logell):
            raise RuntimeError

    except RuntimeError:
        raise ValueError(
            "This material does not meet the defined criterion given the "
            "map constaints."
        )

    factor = np.sqrt((3600 * greg.z * 10.0**optimal_logell) / c_rate)

    sqd = np.sqrt(greg.dcoeff_)
    particle_size = cm_to * factor * sqd

    return (
        particle_size
        if greg.dcoeff_err_ is None
        else (particle_size, cm_to * factor * greg.dcoeff_err_ / (2 * sqd))
    )
