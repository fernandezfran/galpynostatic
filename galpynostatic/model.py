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

"""Module with the galvanostatic regression model."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

import pandas as pd

import sklearn.metrics
from sklearn.base import BaseEstimator, RegressorMixin

from .plot import GalvanostaticPlotter
from .surface import SurfaceSpline
from .utils import flogell, flogxi

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(BaseEstimator, RegressorMixin):
    r"""An heuristic regressor for galvanostatic data.

    This physics-based heuristic model [1]_ uses the diagram in the `dataset`
    (:ref:`galpynostatic.datasets`) to perform a grid search by taking
    different combinations of the diffusion coefficient, :math:`D`, and the
    kinetic rate constant, :math:`k^0`, to fit experimental data of the
    State-of-Charge (SOC) of the electrode material as a function of the
    C-rates. This is done considering invariant all the other experimental
    values involved in the continuum galvanostatic model :math:`\Xi` and
    :math:`\ell` parameters, such as the characteristic diffusion length,
    :math:`d`, and the geometrical factor, :math:`z`.

    Each time a set of parameters :math:`D` and :math:`k^0` is taken, the
    SOC values are predicted and the mean square error (MSE) is computed. Then,
    the set of parameters that minimizes the MSE are obtained, thus
    yielding fundamental parameters of the system that together with the
    diagram allows to make predictions.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with the maximum SOC values diagram as function of
        :math:`\log(\ell)` and :math:`\log(\Xi)` internal parameters, this can
        be loaded using the functions in :ref:`galpynostatic.datasets`.

    d : float
        Characteristic diffusion length.

    z : int
        Geometric factor (1 for planar, 2 for cylinder and 3 for sphere).

    Attributes
    ----------
    dcoeff_ : float
        Predicted diffusion coefficient in :math:`cm^2/s`.

    k0_ : float
        Predicted kinetic rate constant in :math:`cm/s`.

    mse_ : float
        Mean squared error of the best fitted model.

    Notes
    -----
    By default the grid search is performed on the values of
    ``numpy.logspace(-15, -6, num=100)`` and
    ``numpy.logspace(-14, -5, num=100)`` for :math:`D` and :math:`k^0`,
    respectively. Their range and precision can be modified through the
    properties ``dcoeffs`` and ``k0s``.

    References
    ----------
    .. [1] Fernandez, F., Gavilán-Arriazu, E.M., Barraco, D., Visintín, A.,
       Ein-Eli, Y. and Leiva, E., 2023. Towards a fast-charging of LIBs
       electrode materials: a heuristic model based on galvanostatic
       simulations. TODO.
    """

    def __init__(self, dataset, d, z):
        self.d = d
        self.z = z

        self.dcoeff_, self.k0_, self.mse_ = None, None, None

        self._dcoeffs = np.logspace(-15, -6, num=100)
        self._k0s = np.logspace(-14, -5, num=100)

        self._surface = SurfaceSpline(dataset)

    def _logell(self, c_rate):
        r"""Logarithm value in base 10 of :math:`\ell` parameter."""
        return flogell(c_rate, self.d, self.z, self.dcoeff_)

    def _logxi(self, c_rate):
        r"""Logarithm value in base 10 of :math:`\Xi` parameter."""
        return flogxi(c_rate, self.dcoeff_, self.k0_)

    def _soc(self, logell, logxi):
        """Find a single value of the SOC given the surface spline."""
        return self._surface.soc(logell, logxi)[0][0]

    @property
    def dcoeffs(self):
        """Diffusion coefficients to evaluate in model training."""
        return self._dcoeffs

    @dcoeffs.setter
    def dcoeffs(self, dcoeffs):
        """Diffusion coefficients to evaluate in model training setter."""
        self._dcoeffs = dcoeffs

    @property
    def k0s(self):
        """Kinetic rate constants to evaluate in model training."""
        return self._k0s

    @k0s.setter
    def k0s(self, k0s):
        """Kinetic rate constants to evaluate in model training setter."""
        self._k0s = k0s

    def fit(self, X, y):
        """Fit the galvanostatic regressor model.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Target maximum SOC values.

        Returns
        -------
        self : object
            Fitted model.
        """
        params = np.array(list(it.product(self._dcoeffs, self._k0s)))
        mse = np.full(params.shape[0], np.inf)

        for k, (self.dcoeff_, self.k0_) in enumerate(params):
            pred = self.predict(X)
            try:
                mse[k] = sklearn.metrics.mean_squared_error(y, pred)
            except ValueError:
                mse[k] = np.inf

        idx = np.argmin(mse)

        self.dcoeff_, self.k0_ = params[idx]
        self.mse_ = mse[idx]

        return self

    def predict(self, X):
        """Predict using the galvanostatic model in the range of the surface.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        Returns
        -------
        y : array-like of shape (n_measurements,)
            The predicted maximum SOC values for the C-rates inputs.
        """
        y = np.full(X.size, np.nan)
        for k, x in enumerate(X):
            logell = self._logell(x[0])
            logxi = self._logxi(x[0])

            mask_logell = self._surface._mask_logell(logell)
            mask_logxi = self._surface._mask_logxi(logxi)
            if mask_logell and mask_logxi:
                y[k] = self._soc(logell, logxi)

        return y

    def score(self, X, y, sample_weight=None):
        r"""Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input C-rates, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Experimental maximum SOC values.

        sample_weight : Ignored
            Not used, presented for sklearn API consistency by convention.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        """
        return super(GalvanostaticRegressor, self).score(X, y, sample_weight)

    @property
    def plot(self):
        """Plot accessor to :ref:`galpynostatic.plot`."""
        return GalvanostaticPlotter(self)

    def to_dataframe(self, X, y=None):
        """Convert the train, the evaluation or both sets to a dataframe.

        Obtain a dataframe with two or three columns (`C_rates`, `SOC_true` &
        `SOC_pred`), depending if you passed the `y` values or not.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates.

        y : array-like of shape (n_measurements,), default=None
            maximum SOC values.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe with the train or the evaluation set.
        """
        dict_ = {"C_rates": X.ravel()}

        if y is not None:
            dict_["SOC_true"] = y

        dict_["SOC_pred"] = self.predict(X)

        return pd.DataFrame(dict_, dtype=np.float32)
