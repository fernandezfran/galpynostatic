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

"""Module to handle experimental data preprocessing."""

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

    This Transformer subtracts the equilibrium potential, `eq_pot`,
    from all galvanostatic profiles the to find the discharge capacity for
    each curve where the potential is cut off by `vcut`.

    Parameters
    ----------
    eq_pot : float, default=0.0
        The equilibrium potential in volts (V).

    vcut : float, default=0.15
        The cut-off potential in V. The default value is 150 mV, which is the
        value defined by the distributed map data in the
        :ref:`galpynostatic.datasets`.

    Notes
    -----
    Discharge capacities are useful to define the maximum value of the
    State-of-Charge (SOC) for a given galvanostatic charging rate (C-rate),
    which is the appropriate way to have the data for the
    :ref:`galpynostatic.model`. Our suggestion for determining the maximum
    value of the SOC is to take the maximum value of the discharging
    capacities corresponding to the value of the C-rate to which the curve
    already converges with respect to the previous one. In this case, all the
    values obtained for the discharge capacities are divided by this value and
    the maximum SOC values are obtained.
    """

    def __init__(self, eq_pot=0.0, vcut=0.15):
        self.eq_pot = eq_pot
        self.vcut = vcut

    def fit(self, X, y=None, **fit_params):
        """Define the fit parameters.

        Parameters
        ----------
        X : Ignored
            Not used, presented for API consistency.

        y : Ignored
            Not used, presented by convention for sklearn API consistency.

        **fit_params
            Additional keyword arguments that are passed and are documented in
            ``scipy.interpolate.InterpolatedUnivariateSpline``.
        """
        self.fit_params = fit_params
        return self

    def transform(self, X):
        """Transform the curves to single values of discharge capacities.

        Parameters
        ----------
        X : list of pandas.DataFrame
            DataFrames having only two columns, the first being the capacity
            and the second being the voltage.

        Returns
        -------
        X_new : array-like of shape (n_measurements,)
            Discharge capacities in the same order as the ``pandas.DataFrame``
            in the input list, but reshaped to fit.
        """
        X_new = np.zeros(len(X))

        for k, df in enumerate(X):
            capacity = df.iloc[:, 0]
            voltage = df.iloc[:, 1] - self.eq_pot + self.vcut

            spline = scipy.interpolate.InterpolatedUnivariateSpline(
                capacity, voltage, **self.fit_params
            )

            try:
                X_new[k] = spline.roots()[0]
            except IndexError:
                ...

        return X_new

    def fit_transform(self, X, y=None, **fit_params):
        """Transform the curves to discharge capacities with optional params.

        Parameters
        ----------
        X : list of pandas.DataFrame
            DataFrames having only two columns, the first being the capacity
            and the second being the voltage.

        y : Ignored
            Not used, presented for sklearn API consistency by convention.

        **fit_params
            Additional keyword arguments that are passed and are documented in
            ``scipy.interpolate.InterpolatedUnivariateSpline``.

        Returns
        -------
        X_new : array-like of shape (n_measurements,)
            Discharge capacities in the same order as the ``pandas.DataFrame``
            in the input list, but reshaped to fit.
        """
        return super(GetDischargeCapacities, self).fit_transform(
            X, y, **fit_params
        )
