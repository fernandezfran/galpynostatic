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

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error as _skl_mse
from sklearn.utils import validation as _skl_validation

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
    dataset : str or pandas.DataFrame, default="spherical"
        A str indicating the particle geometry (planar, cylindrical or
        spherical) to use the datasets distributed in this package which can
        also be loaded using the functions of the
        :ref:`galpynostatic.datasets` to give it as a ``pandas.DataFrame`` with
        the map of the maximum SOC values as function of the internal
        parameters :math:`\log(\ell)` and :math:`\log(\Xi)`.

    d : float, default=1e-4
        Characteristic diffusion length (particle size) in cm.

    z : integer, default=3
        Geometric factor (1 for planar, 2 for cylinder and 3 for sphere).

    dcoeff_lle : integer, default=-15
        The lower limit exponent of the diffusion coefficient line to generate
        the grid.

    dcoeff_ule : integer, default=-6
        The upper limit exponent of the diffusion coefficient line to generate
        the grid.

    dcoeff_num : integer, default=100
        Number of samples of diffusion coefficients to generate between the
        lower and the upper limit exponent.

    k0_lle : integer, default=-14
        The lower limit exponent of the kinetic rate constant line to generate
        the grid.

    k0_ule : integer, default=-5
        The upper limit exponent of the kinetic rate constant line to generate
        the grid.

    k0_num : integer
        Number of samples of kinetic rate constants to generate between the
        lower and the upper limit exponent.

    Notes
    -----
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
            load_geometry = {
                "planar": load_planar,
                "cylindrical": load_cylindrical,
                "spherical": load_spherical,
            }

            if self.dataset in load_geometry:
                self.dataset = load_geometry[self.dataset]()
            else:
                raise ValueError(f"{self.dataset} is not a valid geometry.")

    def _grid_points(self):
        """Grid points (D, k0) to evaluate in the grid search."""
        dcoeffs = np.logspace(
            self.dcoeff_lle, self.dcoeff_ule, self.dcoeff_num
        )
        k0s = np.logspace(self.k0_lle, self.k0_ule, self.k0_num)
        return np.array(tuple(it.product(dcoeffs, k0s)))

    def _calculate_uncertainties(self, X, y, attrs, delta):
        """Uncertainties of `attrs` calculation.

        The uncertainties are computed as the root squared values of the
        diagonal in the covariance matrix, which is approximated with the
        inverse of the Hessian matrix, calculated as the product of the
        Jacobian matrix with its transpose.
        """
        residuals = y - self.predict(X)
        dfree = len(y) - len(attrs)

        jacobian = np.zeros((len(attrs), len(y)))
        for i, attr in enumerate(attrs):
            param = self.__dict__[attr]

            self.__dict__[attr] = (1 + delta) * param
            upper = self.predict(X)

            self.__dict__[attr] = (1 - delta) * param
            lower = self.predict(X)

            jacobian[i] = (upper - lower) / (2 * delta * param)

        hessian = np.dot(jacobian, jacobian.T)

        covariance = np.var(residuals) * np.linalg.inv(hessian)

        chisq = np.sum(residuals**2) / dfree

        return np.sqrt(np.diag(chisq * covariance))

    def fit(self, X, y, sample_weight=None):
        """Fit the heuristic galvanostatic regressor model.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Target maximum SOC values.

        sample_weight : array-like of shape(n_measurments,), default=None
            Individual weights of each data point.

        Returns
        -------
        self : object
            Fitted model.

        Raises
        ------
        ValueError
            When the dataset instantiated is a str but is not a valid geometry
            (planar, cylindrical or spherical).
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
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Experimental maximum SOC values.

        sample_weight : array-like of shape(n_measurments,), default=None
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
            df["SOC_exp"] = y

        df["SOC_pred"] = self.predict(X)

        return df

    @property
    def plot(self):
        """Plot accessor to :ref:`galpynostatic.plot`."""
        return GalvanostaticPlotter(self)
