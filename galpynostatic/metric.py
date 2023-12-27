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

from .datasets.map import MapSpline
from .model import GalvanostaticRegressor

# ============================================================================
# FUNCTIONS
# ============================================================================


def umbem(greg, minutes=15, loaded=0.8, full_output=False, **kwargs):
    r"""Universal metric for benchmarking fast-charging electrode materials.

    This Universal Metric for Benchmarking fast-charging Electrode Materials
    is defined as the maximum State-of-Charge retained when a material is
    charged for 15 minutes under constant current conditions [2]_. The
    evaluation of the UMBEM is performed with a generic model that considers
    finite diffusion, charge transfer, particle size and the
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

    **kwargs
        Additional keyword arguments that are passed and are documented in
        `galpynostatic.model.GalvanostaticRegressor`.

    Returns
    -------
    soc : float
        UMBEM value.

    res : dict, optional
        A dict present if `full_output=True` and described there.

    References
    ----------
    .. [2] Fernandez, Francisco. `Modelado computacional para el desarrollo de
       electrodos de baterías de ion-litio de próxima generación`. PhD thesis,
       Universidad Nacional de Córdoba, 2024.
    """
    if isinstance(greg, dict):
        greg_ = GalvanostaticRegressor(d=greg["d"], **kwargs)
        greg_._validate_geometry()
        greg_._map = MapSpline(greg_.dataset)
        greg_.dcoeff_, greg_.k0_ = greg["dcoeff_"], greg["k0_"]
        greg_.dcoeff_err_ = None
    else:
        greg_ = greg

    soc = greg_.predict(np.array([[60.0 / minutes]]))[0]

    criteria = soc >= loaded

    return (
        soc
        if not full_output
        else {"soc": soc, "criteria": criteria, "greg": greg_}
    )


def fom(d, dcoeff):
    r"""Figure-of-Merit (FOM) for fast-charging comparisons.

    This metric was proposed by Xia et al. [3]_ and combines the diffusion
    coefficient and the geometric size to define th characteristic time
    of diffusion.

    Parameters
    ----------
    d : float
        Geometric size in :math:`cm`.

    dceoff : float
        Diffusion coefficient in :math:`cm^2/s`.

    Returns
    -------
    float
        The FOM characteristic time of diffusion value.

    References
    ----------
    .. [3] H. Xia, W. Zhang, S. Cao and X. Chen. "A figure of merit for
       fast-charging Li-ion battery materials." `ACS Nano, 16` (2022):
       8525-8530.
    """
    return d**2 / dcoeff
