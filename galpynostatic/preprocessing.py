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

"""Module for preprocessing of experimental data."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.interpolate

from sklearn.base import TransformerMixin

# ============================================================================
# CLASSES
# ============================================================================


class GetDischargeCapacities(TransformerMixin):
    """Obtain the discharge capacities at a given cut-off potential.

    It subtract from all curves the equilibrium potential to find the capacity
    at which the potential is cut off below `vcut`.

    Parameters
    ----------
    eq_pot : float
        The equilibrium potential in Volts (V).

    vcut : float, default=0.15
        The cut-off potential in V, the default value corresponds to 150 mV,
        which is the one defined by the data of the distributed maps.
    """

    def __init__(self, eq_pot, vcut=0.15):
        self.eq_pot = eq_pot
        self.vcut = vcut

    def fit(self, X, y=None, **fit_params):
        """Define the fit parameters.

        Parameters
        ----------
        X : Ignored
            Not used, presented for API consistency.

        y : Ignored
            Not used, presented for sklearn API consistency by convention.

        **fit_params
            Additional keyword arguments that are passed and are documented in
            ``scipy.interpolate.InterpolatedUnivariateSpline``.
        """
        self.fit_params = fit_params
        return self

    def transform(self, X):
        """Perform the substraction of the X curves.

        Parameters
        ----------
        X : `list of pd.DataFrame`
            Dataframes having only two columns, where the first one is the
            capacity and the second one is the voltage.

        Returns
        -------
        X_new : ndarray
            Discharge capacities in the same order as the pd.DataFrame in the
            list.

        Raises
        ------
        ValueError
            When one of the galvanostatic profiles passed in `X` does not
            intersect the cut-off potential below the equilibrium potential.
        """
        try:
            X_new = [
                scipy.interpolate.InterpolatedUnivariateSpline(
                    df.iloc[:, 0],
                    df.iloc[:, 1] - self.eq_pot + self.vcut,
                    **self.fit_params,
                ).roots()[0]
                for df in X
            ]

        except IndexError:
            raise ValueError(
                "A galvanostatic profile does not intersect the cut-off "
                "potential from the equlibrium potential."
            )

        return np.array(X_new, dtype=np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        """Transform the X curves with optional parameters `fit_params`.

        Parameters
        ----------
        X : `list of pd.DataFrame`
            Dataframes having only two columns, where the first one is the
            capacity and the second one is the voltage.

        y : Ignored
            Not used, presented for sklearn API consistency by convention.

        **fit_params
            Additional keyword arguments that are passed and are documented in
            ``scipy.interpolate.InterpolatedUnivariateSpline``.

        Returns
        -------
        X_new : ndarray
            Discharge capacities in the same order as the pd.DataFrame in the
            list.

        Raises
        ------
        ValueError
            When one of the galvanostatic profiles passed in `X` does not
            intersect the cut-off potential below the equilibrium potential.
        """
        return super(GetDischargeCapacities, self).fit_transform(
            X, y, **fit_params
        )
