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

import matplotlib.pyplot as plt

import numpy as np

import scipy.interpolate

import sklearn.metrics
from sklearn.base import RegressorMixin

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor(RegressorMixin):
    """An heuristic regressor for galvanostatic data.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with a map of xmax as function of l and chi parameters, this
        can be loaded using ``galpynostatic.dataset`` load functions.

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
    ``np.logspace(-15, -6, num=100)`` and ``np.logspace(-14, -5, num=100)`` for
    the coefficients D and k, respectively. Their range and precision can be
    modified through the properties ``dcoeffs`` and ``k0s``, respectively.
    """

    def __init__(self, dataset, d, z, t_h=3600):
        self.dataset = dataset
        self.d = d
        self.z = z

        self.t_h = t_h

        self.dcoeff_, self.k0_, self.mse_ = None, None, None

        self._dcoeffs = np.logspace(-15, -6, num=100)
        self._k0s = np.logspace(-14, -5, num=100)

    def _logl(self, cr):
        """Logarithm value of l parameter in base 10."""
        return np.log10(
            (cr * self.d**2) / (self.z * self.t_h * self.dcoeff_)
        )

    def _logchi(self, cr):
        """Logarithm value of chi parameter in base 10."""
        return np.log10(self.k0_ * np.sqrt(self.t_h / (cr * self.dcoeff_)))

    def _xmax_in_surface(self, logl, logchi):
        """Find the value of xmax given the surface spline."""
        return max(0, min(1, self._surf_spl(logl, logchi)[0][0]))

    def _surface(self):
        """Surface spline."""
        self._ls = np.unique(self.dataset.l)
        self._chis = np.unique(self.dataset.chi)

        k, xmaxs = 0, []
        for logl, logchi in it.product(self._ls, self._chis[::-1]):
            xmax = 0
            try:
                if logl == self.dataset.l[k] and logchi == self.dataset.chi[k]:
                    xmax = self.dataset.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                xmaxs.append(xmax)

        self._surf_spl = scipy.interpolate.RectBivariateSpline(
            self._ls,
            self._chis,
            np.asarray(xmaxs).reshape(self._ls.size, self._chis.size)[:, ::-1],
        )

    def _plot_surface(self, ax=None):
        """Plot 2D surface."""
        ax = plt.gca() if ax is None else ax

        leval = np.linspace(np.min(self._ls), np.max(self._ls), num=1000)
        chieval = np.linspace(np.min(self._chis), np.max(self._chis), num=1000)

        z = self._surf_spl(leval, chieval)
        z[z > 1] = 1.0
        z[z < 0] = 0.0

        im = ax.imshow(
            z.T,
            extent=[leval.min(), leval.max(), chieval.min(), chieval.max()],
            origin="lower",
        )
        clb = plt.colorbar(im)
        clb.ax.set_ylabel(r"x$_{max}$")
        clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        return ax

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
            C rates samples.

        y : array-like
            Target normalized discharge capacities.

        Returns
        -------
        self : object
            Fitted model.
        """
        self._surface()

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
            C rates samples.

        Returns
        -------
        y : ndarray
            The predicted normalized discharge capacities
        """
        y = np.full(X.size, None)
        for k, x in enumerate(X):
            logl = self._logl(x[0])
            logchi = self._logchi(x[0])

            if (self._ls.min() <= logl <= self._ls.max()) and (
                self._chis.min() <= logchi <= self._chis.max()
            ):
                y[k] = self._xmax_in_surface(logl, logchi)

        return y

    def score(self, X, y, sample_weight=None):
        r"""Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates samples.

        y : array-like
            True normalized discharge capacities.

        sample_weight : Ignored
            Not used, presented for sklearn API consistency by convention.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        """
        return super(GalvanostaticRegressor, self).score(X, y, sample_weight)

    def t_minutes_lenght(
        self, minutes=5, load_percentage=0.8, dlogl=0.01, cm_to=10000
    ):
        """Obtain the characteristic diffusion length to charge in t minutes.

        Parameters
        ----------
        minutes : int or float, default=5
            Desired minutes to reach the established load.

        load_percentage : float, default=0.8
            Desired charge percentage between 0 and 1.

        dlogl : float, default=0.01
            The delta for the decrease of the logarithm value in base 10 of the
            l value.

        cm_to : float, default=10000
            A factor to convert from cm to another unit, in this case to
            micrometers.

        Returns
        -------
        length : float
            The characteristic length necessary to charge the battery to the
            desired percentage and in the desired time.

        Raises
        ------
        ValueError
            If the normalized discharge capacity was not found to be greater
            than load_percentage and the value of the logarithm in base 10 of
            l is less than the minimum at which the spline was fitted.
        """
        c_rate = 60.0 / minutes

        logchi = self._logchi(c_rate)

        optlogl, xmax = self._logl(c_rate), 0.0
        while xmax < load_percentage:
            optlogl -= dlogl
            xmax = self._xmax_in_surface(optlogl, logchi)
            if optlogl < np.min(self._ls):
                raise ValueError(
                    "It was not possible to find the optimum value for the "
                    "length given the established conditions and the range of "
                    "the surface used."
                )

        length = cm_to * np.sqrt(
            (self.z * self.t_h * self.dcoeff_ * 10.0**optlogl) / c_rate
        )

        return length

    def plot_vs_data(self, X, y, ax=None, data_kws=None, pred_kws=None):
        """Plot predictions against data.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates samples.

        y : array-like
            Target normalized discharge capacities.

        ax : matplotlib.axes.Axes, default=None
            The current axes.

        data_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.pyplot.plot`` for the data points.

        pred_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.pyplot.plot`` for the predictions values.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The current axes.
        """
        ax = plt.gca() if ax is None else ax

        data_kws = {} if data_kws is None else data_kws
        pred_kws = {} if pred_kws is None else pred_kws

        keys = ["color", "marker", "linestyle", "label"]

        for key, value in zip(keys, ["tab:blue", "s", "--", "data"]):
            data_kws.setdefault(key, value)

        for key, value in zip(keys, ["tab:orange", "", "-", "model"]):
            pred_kws.setdefault(key, value)

        ax.plot(X, y, **data_kws)

        xeval = np.linspace(X.min(), X.max(), 250).reshape(-1, 1)
        ax.plot(xeval, self.predict(xeval), **pred_kws)

        ax.set_xlabel("C-rates")
        ax.set_ylabel("SOC")

        ax.set_xscale("log")

        return ax

    def plot_in_surface(self, X, ax=None, **kwargs):
        """Plot showing in which region of the map the fit is found.

        Parameters
        ----------
        X : array-like
            C rates samples.

        ax : matplotlib.axes.Axes, default=None
            The current matplotlib axes.

        **kwargs
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.pyplot.plot``

        Returns
        -------
        ax : matplotlib.axes.Axes
            The current axes.

        Notes
        -----
        Only plot the background surface if ax is None, otherwise assume it is
        already plotted and you just want to add the points on it, e.g., to
        compare different systems.
        """
        if ax is None:
            ax = plt.gca()
            ax = self._plot_surface(ax)

        # fitted data plot
        keys = ["color", "marker", "linestyle", "label"]
        for key, value in zip(keys, ["k", "o", "--", "fitted data"]):
            kwargs.setdefault(key, value)

        ax.plot(self._logl(X), self._logchi(X), **kwargs)

        return ax
