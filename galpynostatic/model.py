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

"""Module with the galvanostatic regression model."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error as _skl_mse
from sklearn.utils import validation as _skl_validation

from .base import MapSpline
from .datasets import load_dataset
from .plot import GalvanostaticPlotter
from .utils import logell, logxi

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(BaseEstimator, RegressorMixin):
    r"""A heuristic regressor for State-of-Charge (SOC) versus C-rates data.

    This physics-based heuristic model [4]_ uses the maps in the
    :ref:`galpynostatic.datasets` to perform a grid search, using different
    combinations of the diffusion coefficient, :math:`D`, and the
    kinetic rate constant, :math:`k^0`, to fit experimental data of the
    State-of-Charge (SOC) of the electrode material as a function of the
    galvanostatic charging rate (C-rate). This is done while keeping invariant
    all the other experimental descriptors involved in the parameters
    :math:`\Xi` and :math:`\ell` of the continuum model [4]_, such
    as the characteristic diffusion length, :math:`d`, and the geometric
    factor, :math:`z` (see :ref:`galpynostatic.utils`).

    Each time a set of parameters :math:`D` and :math:`k^0` are taken, the
    SOC values are predicted and the mean square error (MSE) is calculated.
    The set of parameters that minimises the MSE is then obtained, providing
    a fundamental description of the system.

    Parameters
    ----------
    dataset : str or pandas.DataFrame, default="spherical"
        A str specifying the particle geometry (`"planar"`, `"cylindrical"` or
        `"spherical"`) to use the datasets distributed in this package, which
        can also be loaded using the functions of the
        :ref:`galpynostatic.datasets` to get it as a ``pandas.DataFrame`` with
        the mapping of maximum SOC values as a function of the internal
        parameters :math:`\log(\ell)` and :math:`\log(\Xi)`.

    d : float, default=1e-4
        Characteristic diffusion length (particle size) in cm.

    z : int, default=3
        Geometric factor (`1` for planar, `2` for cylinder and `3` for sphere).

    dcoeff_lle : int, default=-15
        The lower limit exponent of the diffusion coefficient line used to
        generate the grid.

    dcoeff_ule : int, default=-6
        The upper limit exponent of the diffusion coefficient line used to
        generate the grid.

    dcoeff_num : int, default=100
        Number of samples of diffusion coefficients to generate between the
        lower and the upper limit exponents.

    k0_lle : int, default=-14
        The lower limit exponent of the kinetic rate constant line used to
        generate the grid.

    k0_ule : int, default=-5
        The upper limit exponent of the kinetic rate constant line used to
        generate the grid.

    k0_num : int, default=100
        Number of samples of kinetic rate constants to generate between the
        lower and the upper limit exponents.

    Notes
    -----
    You can also give your own dataset to another potential cut-off in the
    same format as the distributed ones and as ``pandas.DataFrame``, i.e. in
    the column of :math:`\ell` the different values have to be grouped in
    ascending order and for each of these groups the :math:`\Xi` have to be in
    descending order and respecting that for each group of :math:`\ell` the
    same values are simulated (this is a restriction to perform the
    ``scipy.interpolate.RectBivariateSpline``, since `x` and `y` must be
    strictly in a special order, which is handled internally by the
    :ref:`galpynostatic.base`).

    References
    ----------
    .. [4] F. Fernandez, E. M. Gavilán-Arriazu, D. E. Barraco, A. Visintin, Y.
       Ein-Eli and E. P. M. Leiva. "Towards a fast-charging of LIBs electrode
       materials: a heuristic model based on galvanostatic simulations."
       `Electrochimica Acta 464` (2023): 142951.

    Attributes
    ----------
    dcoeff_ : float
        Predicted diffusion coefficient in :math:`cm^2/s`.

    dcoeff_err_ : float
        Uncertainty of the predicted diffusion coefficient.

    k0_ : float
        Predicted kinetic rate constant in :math:`cm/s`.

    k0_err_ : float
        Uncertainty of the predicted kinetic rate constant.

    mse_ : float
        Mean squared error of the best fitted model.
    """

    def __init__(
        self,
        dataset="spherical",
        d=1e-4,
        z=3,
        dcoeff_lle=-15,
        dcoeff_ule=-6,
        dcoeff_num=100,
        k0_lle=-14,
        k0_ule=-5,
        k0_num=100,
    ):
        self.dataset = dataset
        self.d = d
        self.z = z
        self.dcoeff_lle = dcoeff_lle
        self.dcoeff_ule = dcoeff_ule
        self.dcoeff_num = dcoeff_num
        self.k0_lle = k0_lle
        self.k0_ule = k0_ule
        self.k0_num = k0_num

    def _validate_geometry(self):
        """Validate geometry (when dataset is a string)."""
        if isinstance(self.dataset, str):
            self.dataset = load_dataset(geometry=self.dataset)

    def _grid_points(self):
        """Grid points (D, k0) to evaluate in the grid search."""
        dcoeffs = np.logspace(
            self.dcoeff_lle, self.dcoeff_ule, num=self.dcoeff_num
        )
        k0s = np.logspace(self.k0_lle, self.k0_ule, num=self.k0_num)
        return np.array(tuple(it.product(dcoeffs, k0s)))

    def _calculate_uncertainties(self, X, y, attrs, delta):
        """Uncertainties of `attrs` calculation.

        The uncertainties are computed as the root squared values of the
        diagonal in the covariance matrix, which is approximated with the
        inverse of the Hessian matrix, calculated as the product of the
        Jacobian matrix with its transpose.
        """
        jacobian = np.zeros((len(attrs), len(y)))
        for i, attr in enumerate(attrs):
            param = self.__dict__[attr]

            self.__dict__[attr] = (1 + delta) * param
            upper = self.predict(X)

            self.__dict__[attr] = (1 - delta) * param
            lower = self.predict(X)

            jacobian[i] = (upper - lower) / (2 * delta * param)

        hessian = np.dot(jacobian, jacobian.T)
        covariance = np.linalg.inv(hessian)

        stdsq = np.sum((y - self.predict(X)) ** 2) / (len(y) - len(attrs))

        return stdsq * np.sqrt(np.diag(covariance))

    def fit(self, X, y, sample_weight=None):
        """Fit the heuristic galvanostatic regressor model.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates data of the experiments.

        y : array-like of shape (n_measurements,)
            Target maximum SOC values, between 0 and 1.

        sample_weight : array-like of shape(n_measurements,), default=None
            Individual weights of each data point.

        Returns
        -------
        self : object
            Fitted model.

        Raises
        ------
        ValueError
            If the instantiated dataset is an str but is not a valid geometry
            (`"planar"`, `"cylindrical"` or `"spherical"`).
        """
        X, y = _skl_validation.check_X_y(X, y)
        self._validate_geometry()

        self._map = MapSpline(self.dataset)

        params = self._grid_points()

        mse = np.full(params.shape[0], np.inf)
        for k, (self.dcoeff_, self.k0_) in enumerate(params):
            try:
                mse[k] = _skl_mse(
                    y, self.predict(X), sample_weight=sample_weight
                )
            except ValueError:
                ...

        idx = np.argmin(mse)
        self.mse_ = mse[idx]

        self.dcoeff_, self.k0_ = params[idx]

        self.dcoeff_err_, self.k0_err_ = self._calculate_uncertainties(
            X, y, ("dcoeff_", "k0_"), np.cbrt(np.finfo(float).eps)
        )

        return self

    def predict(self, X):
        """Predict using the heuristic model within the map constraints.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates input points.

        Returns
        -------
        y : array-like of shape (n_measurements,)
            The predicted maximum SOC values for the C-rate inputs.
        """
        _skl_validation.check_is_fitted(self)
        X = _skl_validation.check_array(X)

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
        sum of squares ``((y_experimental - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares
        ``((y_experimental - y_experimental.mean()) ** 2).sum()``. The best
        possible score is 1.0 and it can be negative (because the model can be
        arbitrarily worse).

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates data of the experiments.

        y : array-like of shape (n_measurements,)
            Maximum SOC values of the experiments.

        sample_weight : array-like of shape(n_measurements,), default=None
            Individual weights of each data point.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        """
        return super(GalvanostaticRegressor, self).score(
            X, y, sample_weight=sample_weight
        )

    def to_dataframe(self, X, y=None):
        """Convert the train, the evaluation or both sets into a dataframe.

        Get a dataframe with two or three columns (`C_rates`, `SOC_exp` and
        `SOC_pred`), depending on whether you have passed the `y`-values or
        not.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rate points.

        y : array-like of shape (n_measurements,), default=None
            Maximum SOC values.

        Returns
        -------
        df : pandas.DataFrame
            A ``pandas.DataFrame`` containing the train, the evaluation or both
            sets.
        """
        df = pd.DataFrame({"C_rates": X.ravel()})

        if y is not None:
            df["SOC_exp"] = y

        df["SOC_pred"] = self.predict(X)

        return df

    @property
    def plot(self):
        """Plot accessor for the :ref:`galpynostatic.plot`."""
        return GalvanostaticPlotter(self)
