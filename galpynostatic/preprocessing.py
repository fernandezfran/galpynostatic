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

    This Transformer will subtract from all galvanostatic profiles the
    equilibrium potential, `eq_pot`, to find the discharge capacity for each
    curve where the potential is cut off by `vcut`.

    Parameters
    ----------
    eq_pot : float, default=0.0
        The equilibrium potential in Volts (V).

    vcut : float, default=0.15
        The cut-off potential in V, the default value corresponds to 150 mV,
        which is the one defined by the data of the distributed maps.

    Notes
    -----
    Discharge capacities are useful to define the maximum value of SOC for a
    given C-rate, which is the appropiate way to have the data for the
    :ref:`galpynostatic.model`. Our suggestion for determining the maximum
    value of SOC is to take the maximum value for the discharge capacities
    that corresponds with the value of the C-rate to which the curve already
    converges with respect to the previous one. In this case, all the values
    obtained for the discharge capacities are divided by this one and the
    maximum SOC values are obtained.
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
            Not used, presented for sklearn API consistency by convention.

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
            Dataframes that have only two columns, where the first is the
            capacity and the second is the voltage.

        Returns
        -------
        X_new : array-like of shape (n_measurement,)
            Discharge capacities in the same order as ``pandas.DataFrame`` in
            the input list.
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
            Dataframes that have only two columns, where the first is the
            capacity and the second is the voltage.

        y : Ignored
            Not used, presented for sklearn API consistency by convention.

        **fit_params
            Additional keyword arguments that are passed and are documented in
            ``scipy.interpolate.InterpolatedUnivariateSpline``.

        Returns
        -------
        X_new : array-like of shape (n_measurement,)
            Discharge capacities in the same order as ``pandas.DataFrame`` in
            the input list.
        """
        return super(GetDischargeCapacities, self).fit_transform(
            X, y, **fit_params
        )
