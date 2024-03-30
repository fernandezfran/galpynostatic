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

"""Metrics for benchmarking fast charging battery electrode materials."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

from .base import MapSpline
from .model import GalvanostaticRegressor

# ============================================================================
# FUNCTIONS
# ============================================================================


def bmxfc(greg, c_rate=4, loaded=0.8, full_output=False, **kwargs):
    r"""Metric for benchmarking an extreme fast charging of Li-ion materials.

    This universal metric for Benchmarking battery electrode Materials for an
    eXtreme Fast Charging (BMXFC) is defined as the maximum State-of-Charge
    (SOC) retained when a material is charged for 15 minutes under constant
    current conditions [2]_. The evaluation of the BMXFC is performed using the
    model in this package, which accounts for finite diffusion, charge
    transfer, particle size and the total charging rate.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor or dict
        An already fitted GalvanostaticRegressor model or a dict containing
        the following keys with float values definened: `d` in :math:`cm`
        (particle size), `dcoeff_` in :math:`cm^2/s` (diffusion coefficient)
        and `k0_` in :math:`cm/s` (kinetic rate constant).

    c_rate : int or float, default=4
        Galvanostatic charging rate (:math:`60 minutes / 15 minutes`, for
        example, for the default case).

    loaded : float, default=0.8
        Criteria for considering the electrode material with fast charging
        capabilities.

    full_output : bool, default=False
        If `full_output` is `False` (default case), the SOC value is returned.
        If it is `True` a dict is returned with the keys `soc`, `criteria` and
        `greg`, where the first one correspond with the SOC value, the second
        is a boolean with True if the electrode material is classified as a
        fast charging one and False if is not and `greg` with the
        `galpynostatic.model.GalvanostaticRegregssor` to allow further
        predictions and plots.

    **kwargs
        Additional keyword arguments that are passed and are documented in
        `galpynostatic.model.GalvanostaticRegressor`.

    Returns
    -------
    soc : float
        BMXFC value.

    res : dict, optional
        A dict present if `full_output=True` and described there.

    References
    ----------
    .. [2] F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, Y. Ein-Eli and
       E. P. M. Leiva. "A metric for benchmarking an extreme fast-charging of
       Li-ion battery electrode materials." `Journal TODO`.
    """
    if isinstance(greg, dict):
        greg_ = GalvanostaticRegressor(d=greg["d"], **kwargs)
        greg_._validate_geometry()
        greg_._map = MapSpline(greg_.dataset)
        greg_.dcoeff_, greg_.k0_ = greg["dcoeff_"], greg["k0_"]
        greg_.dcoeff_err_ = None
    else:
        greg_ = greg

    soc = greg_.predict(np.reshape([c_rate], (-1, 1)))[0]

    return (
        soc
        if not full_output
        else {"soc": soc, "criteria": soc >= loaded, "greg": greg_}
    )


def fom(d, dcoeff):
    r"""Figure-of-Merit (FOM) for fast charging comparisons.

    This metric was proposed by Xia et al. [3]_ and combines the diffusion
    coefficient, :math:`D`, and the geometric size, :math:`d`, to define the
    characteristic time of diffusion, :math:`\tau`.

    Parameters
    ----------
    d : float
        Geometric size in :math:`cm`.

    dceoff : float
        Diffusion coefficient in :math:`cm^2/s`.

    Returns
    -------
    float
        The FOM characteristic diffusion time (:math:`\tau`) value.

    References
    ----------
    .. [3] H. Xia, W. Zhang, S. Cao and X. Chen. "A figure of merit for
       fast-charging Li-ion battery materials." `ACS Nano, 16` (2022):
       8525-8530.
    """
    return d**2 / dcoeff
