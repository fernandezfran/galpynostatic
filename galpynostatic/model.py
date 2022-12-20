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

"""Model for a galvanostatic fitting."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import matplotlib.pyplot as plt

import numpy as np

import scipy.interpolate

import sklearn.metrics

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticRegressor:
    """Galvanostatic Regressor model class.

    Parameters
    ----------
    dataset : pd.DataFrame
        dataset with a map of xmax as function of l and chi parameters, this
        can be loaded using `galpynostatic.dataset` load functions.

    d : float
        characteristic diffusion length.

    z : int
        geometric factor: 1 for planar, 2 for cylinder and 3 for sphere.

    t_h : int or float, default=3600
        time equivalent to one hour in suitable time units, by default in
        seconds.

    Attributes
    ----------
    dcoeff_ : float
        estimated diffusion coefficient.

    k0_ : float
        estimated kinetic rate constant.

    mse_ : float
        mean squared error of the fitted model.
    """

    def __init__(self, dataset, d, z, t_h=3600):
        self.dataset = dataset
        self.d = d
        self.z = z

        self.t_h = t_h

        self.dcoeff_ = None
        self.k0_ = None
        self.mse_ = None

        self._dcoeffs = 10.0 ** np.arange(-15, -6, 0.1)
        self._k0s = 10.0 ** np.arange(-14, -5, 0.1)

        self._surface()

    def _l(self, c_rate):
        """Value of l parameter."""
        return (self.d**2 * c_rate) / (self.z * self.t_h * self.dcoeff_)

    def _chi(self, c_rate):
        """Value of chi parameter."""
        return self.k0_ * np.sqrt(self.t_h / (c_rate * self.dcoeff_))

    def _surface(self):
        """Surface spline."""
        ls = np.unique(self.dataset.l)
        chis = np.unique(self.dataset.chi)

        k, xmaxs = 0, []
        for l, chi in it.product(ls, chis[::-1]):
            xmax = 0
            try:
                if l == self.dataset.l[k] and chi == self.dataset.chi[k]:
                    xmax = self.dataset.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                xmaxs.append(xmax)

        xmaxs = np.asarray(xmaxs).reshape(ls.size, chis.size)[:, ::-1]

        self._surface_spl = scipy.interpolate.RectBivariateSpline(
            ls, chis, xmaxs
        )

    def _xmax_in_surface(self, l, chi):
        """Find the xmax value in the dataset surface."""
        return max(
            0, min(1, self._surface_spl(np.log10(l), np.log10(chi))[0][0])
        )

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

    def fit(self, C_rates, xmaxs):
        """Fit the galvanostatic model.

        Parameters
        ----------
        C_rates : array-like of shape (n_measurements, 1).
            C-rate samples.

        xmaxs : array-like
            Target normalized discharge capacities.

        Returns
        -------
        self : object
            Fitted model.
        """
        dks = list(it.product(self._dcoeffs, self._k0s))

        mse = [
            sklearn.metrics.mean_squared_error(xmaxs, self.predict(C_rates))
            for self.dcoeff_, self.k0_ in dks
        ]

        idx = np.argmin(mse)

        self.dcoeff_, self.k0_ = dks[idx]
        self.mse_ = mse[idx]

        return self

    def predict(self, C_rates):
        """Predict using the galvanostatic model.

        Parameters
        ----------
        C_rates : array-like of shape (n_measurements, 1).
            C_rate samples.

        Returns
        -------
        `np.array`
            an array with the predicted normalized discharge capacities.
        """
        return np.array(
            [
                self._xmax_in_surface(self._l(c_rate[0]), self._chi(c_rate[0]))
                for c_rate in C_rates
            ]
        )

    def plot_vs_data(
        self, C_rates, xmaxs, ax=None, data_kws=None, predictions_kws=None
    ):
        """Plot predictions against data.

        Parameters
        ----------
        C_rates : array-like
            C_rate samples.

        xmaxs : array-like
            Data of normalized discharge capacities.

        ax : matplotlib.pyplot.Axis, default=None
            the current axes.

        data_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            matplotlib.pyplot.plot for the data points.

        predictions_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            matplotlib.pyplot.plot for the predictions points.

        Returns
        -------
        matplotlib.pyplot.Axis
            the current axes.
        """
        ax = plt.gca() if ax is None else ax

        data_kws = {} if data_kws is None else data_kws
        predictions_kws = {} if predictions_kws is None else predictions_kws

        keys = ["marker", "linestyle", "label"]

        for key, value in zip(keys, ["s", "--", "data"]):
            data_kws.setdefault(key, value)

        for key, value in zip(keys, ["o", "--", "model predictions"]):
            predictions_kws.setdefault(key, value)

        ax.plot(C_rates, xmaxs, **data_kws)
        ax.plot(C_rates, self.predict(C_rates), **predictions_kws)

        return ax

    def plot_in_surface(self, C_rates, ax=None):
        """Plot showing in which region of the map the fit is found.

        Parameters
        ----------
        C_rates : array-like
            C_rate samples.

        ax : matplotlib.pyplot.Axis, default=None
            current matplotlib axis.

        Returns
        -------
        matplotlib.pyplot.Axis
        """
        ax = plt.gca() if ax is None else ax

        ls = np.unique(self.dataset.l)
        chis = np.unique(self.dataset.chi)

        leval = np.linspace(np.min(ls), np.max(ls), num=1000)
        chieval = np.linspace(np.min(chis), np.max(chis), num=1000)

        z = self._surface_spl(leval, chieval)
        z[z > 1] = 1.0
        z[z < 0] = 0.0

        im = ax.imshow(
            z.T,
            extent=[
                self.dataset.l.min(),
                self.dataset.l.max(),
                self.dataset.chi.min(),
                self.dataset.chi.max(),
            ],
            origin="lower",
        )
        clb = plt.colorbar(im)
        clb.ax.set_ylabel(r"x$_{max}$")
        clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        # fitted data plot
        ax.scatter(
            np.log10(self._l(C_rates)),
            np.log10(self._chi(C_rates)),
            color="k",
            linestyle="--",
            label="fitted data",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))

        return ax
