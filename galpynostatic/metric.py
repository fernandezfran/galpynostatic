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

"""Metrics to compare fast-charging battery electrode materials."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

from .model import GalvanostaticRegressor

# ============================================================================
# FUNCTIONS
# ============================================================================


def bmx_fc(greg, minutes=15, loaded=0.8, full_output=False):
    r"""Universal metric for benchmarking fast-charging electrode materials.

    This universal metric for Benchmarking electrode Materials for an eXtreme
    fast charging (BMX-FC) is defined as the maximum State-of-Charge retained
    when a material is charged for 15 minutes under constant current
    conditions. The evaluation of the BMX-FC is performed with a generic model
    that considers finite diffusion, charge transfer, particle size and the
    overall charging rate.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor or dict
        An already fitted GalvanostaticRegressor model or a dict containing
        the following keys with float values definened: `d` in :math:`cm`,
        `dcoeff_` in :math:`cm^2/s` and `k0_` in :math:`cm/s`.

    minutes : int or float, default=15
        Minutes of charging.

    loaded : float, default=0.8
        Criteria to consider the electrode material with fast-charging
        capabilities.

    full_output : bool, default=False
        If `full_output` is False (default), the SOC value is returned. If
        True a dict is returned with the keys `soc`, `criteria` and `greg`,
        where the first one correspond with the SOC value, the second is a
        boolean with True if the electrode material is classified as a
        fast-charging one and False if is not and `greg` with the
        `galpynostatic.model.GalvanostaticRegregssor` to allow further
        predictions and plots.

    Returns
    -------
    soc : float
        BMX-FC value.

    res : dict, optional
        A dict present if `full_output=True` and described there.
    """
    if isinstance(greg, dict):
        dcoeff_log10 = np.log10(greg.dcoeff_)
        k0_log10 = np.log10(greg.k0_)

        greg = GalvanostaticRegressor(
            d=greg.d,
            dcoeff_lle=dcoeff_log10,
            dcoeff_ule=dcoeff_log10,
            dcoeff_num=1,
            k0_lle=k0_log10,
            k0_ule=k0_log10,
            k0_num=1,
        )

        X = np.logspace(-1, 0, 10).reshape(-1, 1)
        y = 1 - 2 * np.arctan(np.logspace(-1, 1, 10)) / np.pi

        greg.fit(X, y)

    soc = greg.predict(np.array([[60.0 / minutes]]))[0]

    criteria = soc >= loaded

    return (
        soc
        if not full_output
        else {"soc": soc, "criteria": criteria, "greg": greg}
    )


def fom(dcoeff, d):
    r"""Figure-of-Merit (FOM) for fast-charging comparisons.

    This metric was proposed by Xia et al. [1]_ and combines the diffusion
    coefficient and the geometric size to define th characteristic time
    of diffusion.

    Parameters
    ----------
    dceoff : float
        Diffusion coefficient in :math:`cm^2/s`.

    d : float
        Geometric size in :math:`cm`.

    Returns
    -------
    float
        The FOM characteristic time of diffusion value.

    References
    ----------
    .. [1] H. Xia, W. Zhang, S. Cao and X. Chen. "A figure of merit for
       fast-charging Li-ion battery materials." `ACS Nano, 16` (2022):
       8525-8530.
    """
    return d**2 / dcoeff
