#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================================
# DOCS
# ============================================================================

"""Common utilities for experimental data processing."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

# ============================================================================
# FUNCTIONS
# ============================================================================


def get_discharge_capacities(dfs, eq_pot, vcut=0.15, **kwargs):
    """Obtain the discharge capacities at a given cut-off potential.

    It subtract from all curves the equilibrium potential to find the capacity
    at which the potential is cut off 150mV below.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        having only two columns, where the first one is the capacity and the
        second one the voltage

    eq_pot : float
        equilibrium potential in V

    vcut : float, default=0.15
        cut-off potential in V

    **kwargs
        additional keyword arguments that are passed and are documented in
        `scipy.interpolate.InterpolatedUnivariateSpline`

    Returns
    -------
    np.array
        discharge capacities in the same order as the pd.DataFrame in the
        list
    """
    return np.asarray(
        [
            scipy.interpolate.InterpolatedUnivariateSpline(
                df.iloc[:, 0], df.iloc[:, 1] - eq_pot + vcut, **kwargs
            ).roots()[0]
            for df in dfs
        ],
        dtype=np.float32,
    )
