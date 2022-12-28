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

"""Utilities for experimental data preprocessing."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

# ============================================================================
# FUNCTIONS
# ============================================================================


def get_discharge_capacities(dataframes, eq_pot, vcut=0.15, **kwargs):
    """Obtain the discharge capacities at a given cut-off potential.

    It subtract from all curves the equilibrium potential to find the capacity
    at which the potential is cut off 150mV below.

    Parameters
    ----------
    dataframes : `list` of `pd.DataFrame`
        having only two columns, where the first one is the capacity and the
        second one the voltage

    eq_pot : float
        equilibrium potential in V

    vcut : float, default=0.15
        cut-off potential in V, the default value corresponds to 150 mV, which
        is the one defined by the data of the distributed maps

    **kwargs
        additional keyword arguments that are passed and are documented in
        ``scipy.interpolate.InterpolatedUnivariateSpline``

    Returns
    -------
    np.array
        discharge capacities in the same order as the pd.DataFrame in the
        list

    Raises
    ------
    ValueError
        When one of the galvanostatic profiles passed in `dataframes` does not
        intersect the cut-off potential below the equilibrium potential
    """
    try:
        roots = [
            scipy.interpolate.InterpolatedUnivariateSpline(
                df.iloc[:, 0], df.iloc[:, 1] - eq_pot + vcut, **kwargs
            ).roots()[0]
            for df in dataframes
        ]

    except IndexError:
        raise ValueError(
            "A galvanostatic profile does not intersect the cut-off potential "
            "from the equlibrium potential."
        )

    return np.array(roots, dtype=np.float32)
