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

"""Parameters functions."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

# ============================================================================
# FUNCTIONS
# ============================================================================


def flogell(c_rate, d, z, dcoeff, t_h=3600):
    r"""Obtain log value in base 10 of :math:`\ell` parameter.

    Where :math:`\ell = d z \left( \frac{C_{rate}}{D t_h} \right)`.

    Parameters
    ----------
    c_rate : float or int
        C-rate single value

    d : float
        Characteristic diffusion length.

    z : int
        Geometric factor: 1 for planar, 2 for cylinder and 3 for sphere.

    dcoeff : float
        Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    k0 : float
        Kinetic rate constant, :math:`k^0`, in :math:`cm/s`.

    t_h : int or float, default=3600
        Time equivalent to one hour in suitable time units, by default in
        seconds.

    Returns
    -------
    logell : float
        The log 10 value of :math:`\ell` parameter.
    """
    return np.log10((c_rate * d**2) / (z * t_h * dcoeff))


def flogxi(c_rate, dcoeff, k0, t_h=3600):
    r"""Obtain log value in base 10 of :math:`\Xi` parameter.

    Where :math:`\Xi = k^0 \sqrt{\frac{t_h}{C_{rate} D}}`.

    Parameters
    ----------
    c_rate : float or int
        C-rate single value

    dcoeff : float
        Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    k0 : float
        Kinetic rate constant, :math:`k^0`, in :math:`cm/s`.

    t_h : int or float, default=3600
        Time equivalent to one hour in suitable time units, by default in
        seconds.

    Returns
    -------
    logxi : float
        The log 10 value of :math:`\Xi` parameter.
    """
    return np.log10(k0 * np.sqrt(t_h / (c_rate * dcoeff)))
