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

"""Module with the galvanostatic regression model."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

import pandas as pd

import sklearn.metrics
from sklearn.base import BaseEstimator, RegressorMixin

from .datasets import load_cylindrical, load_planar, load_spherical
from .datasets.map import MapSpline
from .plot import GalvanostaticPlotter
from .utils import logell, logxi

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(BaseEstimator, RegressorMixin):
    r"""A heuristic regressor for SOC versus C-rates galvanostatic data.

    This physics-based heuristic model [1]_ uses the maps in
    :ref:`galpynostatic.datasets` to perform a grid search by taking different
    combinations of the diffusion coefficient, :math:`D`, and the
    kinetic-rate constant, :math:`k^0`, to fit experimental data of the
    State-of-Charge (SOC) of the electrode material as a function of the
    C-rates. This is done considering invariant all the other experimental
    values involved in the parameters :math:`\Xi` and :math:`\ell` of the
    maps of the continuous galvanostatic model [1]_, such as the
    characteristic diffusion length, :math:`d`, and the geometrical factor,
    :math:`z` (see :ref:`galpynostatic.utils`).

    Each time a set of parameters :math:`D` and :math:`k^0` is taken, the
    SOC values are predicted and the mean square error (MSE) is calculated.
    Then, the set of parameters that minimizes the MSE is obtained, thus
    providing fundamental parameters of the system.

    Parameters
    ----------
    dataset : str or pandas.DataFrame
        A str indicating the particle geometry (planar, cylindrical or
        spherical) to use the datasets distributed in this package which can
        also be loaded using the functions of the
        :ref:`galpynostatic.datasets` to give it as a ``pandas.DataFrame`` with
        the map of the maximum SOC values as function of the internal
        parameters :math:`\log(\ell)` and :math:`\log(\Xi)`.

    d : float
        Characteristic diffusion length (particle size) in cm.

    z : int
        Geometric factor (1 for planar, 2 for cylinder and 3 for sphere).

    Raises
    ------
    ValueError
        When the dataset passed is a str but is not a valid geometry (planar,
        cylindrical or spherical).

    Notes
    -----
    By default the grid search is performed on the values of
    ``numpy.logspace(-15, -6, num=100)`` and
    ``numpy.logspace(-14, -5, num=100)`` for :math:`D` and :math:`k^0`,
    respectively. Their range and number of samples to be evaluated can be
    modified through the properties ``dcoeffs`` and ``k0s``.

    You can also give your own dataset to another potential cut-off in the
    same format as the distributed ones and as ``pandas.DataFrame``, i.e. in
    the column of :math:`\ell` the different values have to be grouped in
    ascending order and for each of these groups the :math:`\Xi` have to be in
    decreasing order and respecting that for each group of :math:`\ell` the
    same values are simulated (this is a restriction to perform the
    ``scipy.interpolate.RectBivariateSpline``, since `x` and `y` have to be
    strictly in a special order, which is handled internally by the
    :ref:`galpynostatic.map`).

    References
    ----------
    .. [1] F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin,
       Y. Ein-Eli, E. P. M. Leiva, 2023. Towards a fast-charging of LIBs
       electrode materials: a heuristic model based on galvanostatic
       simulations. _TODO_.


    Attributes
    ----------
    dcoeff_ : float
        Predicted diffusion coefficient in :math:`cm^2/s`.

    dcoeff_err_ : float
        Uncertainty in the predicted diffusion coefficient.

    k0_ : float
        Predicted kinetic rate constant in :math:`cm/s`.

    k0_err_ : float
        Uncertainty in the predicted kinetic rate constant.

    mse_ : float
        Mean squared error of the best fitted model.
    """

    def __init__(self, dataset, d, z):
        self.dataset = dataset
        self.d = d
        self.z = z

        self.dcoeff_, self.k0_, self.mse_ = None, None, None
        self.dcoeff_err_, self.k0_err_ = None, None

        self._dcoeffs = np.logspace(-15, -6, num=100)
        self._k0s = np.logspace(-14, -5, num=100)

        load_geometry = {
            "planar": load_planar,
            "cylindrical": load_cylindrical,
            "spherical": load_spherical,
        }
        if isinstance(self.dataset, str):
            if self.dataset in load_geometry:
                self.dataset = load_geometry[self.dataset]()
            else:
                raise ValueError(f"{self.dataset} is not a valid geometry.")
        self._map = MapSpline(self.dataset)

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
        """Fit the heuristic galvanostatic regressor model.

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
        params = np.array(tuple(it.product(self._dcoeffs, self._k0s)))

        mse = np.full(params.shape[0], np.inf)
        for k, (self.dcoeff_, self.k0_) in enumerate(params):
            pred = self.predict(X)
            try:
                mse[k] = sklearn.metrics.mean_squared_error(y, pred)
            except ValueError:
                ...

        idx = np.argmin(mse)

        self.dcoeff_, self.k0_ = params[idx]
        self.mse_ = mse[idx]

        self.dcoeff_err_, self.k0_err_ = _estimate_uncertainties(
            self, X, y, ("dcoeff_", "k0_"), 1e-6
        )

        return self

    def predict(self, X):
        """Predict using the heuristic model within the range of the map.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        Returns
        -------
        y : array-like of shape (n_measurements,)
            The predicted maximum SOC values for the C-rates inputs.
        """
        logells = logell(X.ravel(), self.d, self.z, self.dcoeff_)
        logxis = logxi(X.ravel(), self.dcoeff_, self.k0_)

        mask_logell = self._map._mask_logell(logells)
        mask_logxi = self._map._mask_logxi(logxis)

        return np.where(
            mask_logell & mask_logxi, self._map.soc(logells, logxis), np.nan
        )

    def score(self, X, y, sample_weight=None):
        r"""Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse).

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

    def to_dataframe(self, X, y=None):
        """Convert the train, the evaluation or both sets into a dataframe.

        Get a dataframe with two or three columns (`C_rates`, `SOC_true` and
        `SOC_pred`), depending on whether you have passed the `y` values or
        not.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates.

        y : array-like of shape (n_measurements,), default=None
            maximum SOC values.

        Returns
        -------
        df : pandas.DataFrame
            A ``pandas.DataFrame`` with the train, the evaluation or both sets.
        """
        df = pd.DataFrame({"C_rates": X.ravel()})

        if y is not None:
            df["SOC_true"] = y

        df["SOC_pred"] = self.predict(X)

        return df

    @property
    def plot(self):
        """Plot accessor to :ref:`galpynostatic.plot`."""
        return GalvanostaticPlotter(self)


# ============================================================================
# FUNCTIONS
# ============================================================================


def _estimate_uncertainties(greg, X, y, attrs, delta):
    """Uncertainties of `attrs` estimations."""
    residuals = y - greg.predict(X)

    sigmas = np.zeros(len(attrs))
    for i, attr in enumerate(attrs):
        param = greg.__dict__[attr]

        greg.__dict__[attr] = (1 + delta) * param
        upper = greg.predict(X)

        greg.__dict__[attr] = (1 - delta) * param
        lower = greg.predict(X)

        diff = upper - lower
        mask = diff != 0

        derivative = diff[mask] / (2 * delta * param)

        ws = residuals[mask] / 2
        norm = np.sum(ws**2) / (len(ws) - 1)
        sigmas[i] = norm * np.sum((ws / derivative) ** 2)

    return np.sqrt(sigmas) / len(residuals)
