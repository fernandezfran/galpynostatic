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

"""Module with the galvanostatic model."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

import pandas as pd

import sklearn.metrics
from sklearn.base import RegressorMixin

from ._surface import SurfaceSpline
from .plot import GalvanostaticPlotter

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(RegressorMixin):
    """An heuristic regressor for galvanostatic data.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with a map of State of Charge (SOC) as function of l and chi
        parameters, this can be loaded using :ref:`galpynostatic.datasets`
        load functions.

    d : float
        Characteristic diffusion length.

    z : int
        Geometric factor: 1 for planar, 2 for cylinder and 3 for sphere.

    t_h : int or float, default=3600
        Time equivalent to one hour in suitable time units, by default in
        seconds.

    Attributes
    ----------
    dcoeff_ : float
        Estimated diffusion coefficient.

    k0_ : float
        Estimated kinetic rate constant.

    mse_ : float
        Mean squared error of the fitted model.

    Notes
    -----
    By default the grid search is performed on the values of
    ``numpy.logspace(-15, -6, num=100)`` and
    ``numpy.logspace(-14, -5, num=100)`` for the coefficients D and k,
    respectively. Their range and precision can be modified through the
    properties ``dcoeffs`` and ``k0s``, respectively.
    """

    def __init__(self, dataset, d, z, t_h=3600):
        self.dataset = dataset
        self.d = d
        self.z = z

        self.t_h = t_h

        self.dcoeff_, self.k0_, self.mse_ = None, None, None

        self._dcoeffs = np.logspace(-15, -6, num=100)
        self._k0s = np.logspace(-14, -5, num=100)

        self._surface = SurfaceSpline(dataset)

    def _logl(self, cr):
        """Logarithm value in base 10 of l parameter."""
        return np.log10(
            (cr * self.d**2) / (self.z * self.t_h * self.dcoeff_)
        )

    def _logchi(self, cr):
        """Logarithm value in base 10 of chi parameter."""
        return np.log10(self.k0_ * np.sqrt(self.t_h / (cr * self.dcoeff_)))

    def _soc_approx(self, logl, logchi):
        """Find the value of soc given the surface spline.

        This is a linear function bounded in [0, 1], values exceeding this
        range are taken to the corresponding end point.
        """
        return max(0, min(1, self._surface.spline(logl, logchi)[0][0]))

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
        """Fit the galvanostatic model.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates measurements.

        y : array-like of shape (n_measurements,)
            Target State of Charge (SOC).

        Returns
        -------
        self : object
            Fitted model.
        """
        dks = np.array(list(it.product(self._dcoeffs, self._k0s)))
        mse = np.full(dks.shape[0], np.inf)

        for k, (self.dcoeff_, self.k0_) in enumerate(dks):
            pred = self.predict(X)
            if None not in pred:
                mse[k] = sklearn.metrics.mean_squared_error(y, pred)

        idx = np.argmin(mse)

        self.dcoeff_, self.k0_ = dks[idx]
        self.mse_ = mse[idx]

        return self

    def predict(self, X):
        """Predict using the galvanostatic model in the range of the surface.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates measurements.

        Returns
        -------
        y : array-like of shape (n_measurements,)
            The predicted SOC for the C rates inputs.
        """
        y = np.full(X.size, None)
        for k, x in enumerate(X):
            logl = self._logl(x[0])
            logchi = self._logchi(x[0])

            if (self._surface.ls.min() <= logl <= self._surface.ls.max()) and (
                self._surface.chis.min() <= logchi <= self._surface.chis.max()
            ):
                y[k] = self._soc_approx(logl, logchi)

        return y

    def score(self, X, y, sample_weight=None):
        r"""Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates measurements.

        y : array-like of shape (n_measurements,)
            True SOC.

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
        """Plot accessor."""
        return GalvanostaticPlotter(self)

    def to_dataframe(self, X, y=None):
        """Convert the train or the evaluation set to a dataframe.

        You can transform the training dataset, in case you pass in the y
        values, you will have a dataframe with three columns: `C_rates`,
        `SOC_true` & `SOC_pred`.

        In the default case, in which `y` is `None`, you can pass any value of
        `X` with physical meaning and predict on it, in that case the dataframe
        will have only two columns: `C_rates` & `SOC_pred`.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates.

        y : array-like of shape (n_measurements,), default=None
            SOC.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe with the train or the evaluation set values.
        """
        dict_ = {"C_rates": X.ravel()}

        if y is not None:
            dict_["SOC_true"] = y

        dict_["SOC_pred"] = self.predict(X)

        return pd.DataFrame(dict_, dtype=np.float32)
