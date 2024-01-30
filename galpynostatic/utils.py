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

"""Internal unitless parameters functions."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

# ============================================================================
# FUNCTIONS
# ============================================================================


def logell(c_rate, d, z, dcoeff):
    r"""Obtain log value in base 10 of :math:`\ell` internal parameter.

    :math:`\ell = \frac{d^2 C_{rate}}{z t_h D}` where :math:`d` is the
    particle size, :math:`C_{rate}` is the galvanostatic charging rate,
    :math:`z` is a geometrical factor, :math:`D` is the diffusion coefficient
    and :math:`t_h` is the time equivalent to one hour in suitable time units,
    here 3600 seconds.

    Parameters
    ----------
    c_rate : float or int or array-like
        C-rate values.

    d : float
        Characteristic diffusion length (particle size) in :math:`cm`.

    z : int
        Geometric factor: 1 for planar, 2 for cylinder and 3 for sphere.

    dcoeff : float
        Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    Returns
    -------
    logell : float or array-like
        The log 10 value of :math:`\ell` internal parameter.
    """
    return np.log10((np.asarray(c_rate) * d**2) / (3600 * z * dcoeff))


def logxi(c_rate, dcoeff, k0):
    r"""Obtain log value in base 10 of :math:`\Xi` internal parameter.

    :math:`\Xi = k^0 \sqrt{\frac{t_h}{C_{rate} D}}` where :math:`k^0` is the
    kinetic rate constant, :math:`t_h` is the time equivalent to one hour in
    suitable time units, here 3600 seconds, :math:`C_{rate}` is the
    galvanostatic charging rate and :math:`D` is the diffusion coefficient.

    Parameters
    ----------
    c_rate : float or int or array-like
        C-rate values.

    dcoeff : float
        Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    k0 : float
        Kinetic rate constant, :math:`k^0`, in :math:`cm/s`.

    Returns
    -------
    logxi : float or array-like
        The log 10 value of :math:`\Xi` internal parameter.
    """
    return np.log10(k0 * np.sqrt(3600 / (np.asarray(c_rate) * dcoeff)))
