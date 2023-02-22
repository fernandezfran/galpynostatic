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

from .plot import GalvanostaticPlotter
from .surface import SurfaceSpline
from .utils import flogell, flogxi

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(RegressorMixin):
    r"""An heuristic regressor for galvanostatic data.

    The physics-based model uses the diagrams from the datasets
    (:ref:`galpynostatic.datasets`) to perform a grid search of the :math:`\Xi`
    and :math:`\ell` simulation parameters. The grid search consists of taking
    experimental measurements of the State-of-Charge (SOC) of the electrode as
    a function of the C-rates and trying different possible combinations of the
    diffusion coefficient (:math:`D`) and the kinetic constant (:math:`k^0`).
    This is done considering invariant the other parameters involved in
    :math:`\Xi` and :math:`\ell`, such as the characteristic diffusion length
    (:math:`d`) and the geometrical factor (:math:`z`). Each time a set of
    parameters :math:`D` and :math:`k^0` is taken, the values that would be
    obtained for the SOC in the diagram are predicted and the mean square error
    (MSE) is calculated. After an exhaustive exploration, the set of parameters
    that minimizes the MSE are obtained, thus yielding fundamental parameters
    of the system that together with the diagram allows to make predictions.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset with a map of State of Charge (SOC) as function of :math:`\ell`
        and :math:`\Xi` parameters, this can be loaded using the load functions
        in :ref:`galpynostatic.datasets`

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
        Estimated diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    k0_ : float
        Estimated kinetic rate constant, :math:`k^0`, in :math:`cm/s`.

    mse_ : float
        Mean squared error of the fitted model.

    Notes
    -----
    By default the grid search is performed on the values of
    ``numpy.logspace(-15, -6, num=100)`` and
    ``numpy.logspace(-14, -5, num=100)`` for the coefficients :math:`D` and
    :math:`k^0`, respectively. Their range and precision can be modified
    through the properties ``dcoeffs`` and ``k0s``, respectively.
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

    def _logell(self, c_rate):
        r"""Logarithm value in base 10 of :math:`\ell` parameter."""
        return flogell(c_rate, self.d, self.z, self.dcoeff_, t_h=self.t_h)

    def _logxi(self, c_rate):
        r"""Logarithm value in base 10 of :math:`\Xi` parameter."""
        return flogxi(c_rate, self.dcoeff_, self.k0_, t_h=self.t_h)

    def _soc(self, logell, logxi):
        """Find the value of SOC given the surface spline."""
        return self._surface.soc(logell, logxi)

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
            C-rates measurements.

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
            C-rates measurements.

        Returns
        -------
        y : array-like of shape (n_measurements,)
            The predicted SOC for the C-rates inputs.
        """
        y = np.full(X.size, None)
        for k, x in enumerate(X):
            logell = self._logell(x[0])
            logxi = self._logxi(x[0])

            if (
                self._surface.ells.min() <= logell <= self._surface.ells.max()
            ) and (
                self._surface.xis.min() <= logxi <= self._surface.xis.max()
            ):
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
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates measurements.

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
            C-rates.

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
