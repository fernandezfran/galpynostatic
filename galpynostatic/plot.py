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

"""Plot helper for GalvanostaticRegressor object."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticPlotter:
    r"""GalvanostaticRegressor plot utilities.

    Kind of plots to produce:

    - 'surface' : the surface on which the data was fitted.
    - 'in_surface' : :math:`\Xi` and :math:`\ell` data points in the surface.
    - 'versus_data' : predicted & true maximum SOC values versus C-rate data.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted galvanostatic model.
    """

    def __init__(self, greg):
        self.greg = greg

    def surface(self, ax=None):
        """Plot 2D surface on which data was fitted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The current axes.
        """
        ax = plt.gca() if ax is None else ax

        logelleval = np.linspace(
            np.min(self.greg._surface.logells),
            np.max(self.greg._surface.logells),
            num=1000,
        )
        logxieval = np.linspace(
            np.min(self.greg._surface.logxis),
            np.max(self.greg._surface.logxis),
            num=1000,
        )

        z = self.greg._surface.soc(logelleval, logxieval)

        im = ax.imshow(
            z.T,
            extent=[
                logelleval.min(),
                logelleval.max(),
                logxieval.min(),
                logxieval.max(),
            ],
            origin="lower",
        )
        clb = plt.colorbar(im)
        clb.ax.set_ylabel("SOC")
        clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        return ax

    def in_surface(self, X, ax=None, **kwargs):
        """Plot showing in which region of the map the fit is found.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        ax : matplotlib.axes.Axes, default=None
            The current matplotlib axes.

        **kwargs
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot``

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
        ax = self.surface() if ax is None else ax

        keys = ["color", "marker", "linestyle", "label"]
        for key, value in zip(keys, ["k", "o", "--", "fitted data"]):
            kwargs.setdefault(key, value)

        ax.plot(self.greg._logell(X), self.greg._logxi(X), **kwargs)

        return ax

    def versus_data(
        self, X, y, X_eval=None, ax=None, data_kws=None, pred_kws=None
    ):
        """Plot SOC predictions against true data as a function of C-rates.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Target maximum SOC values.

        X_eval : array-like of shape (n_measurements, 1), default=None.
            C-rates values to evalute the model to compare against data. When
            is defined as `None`, it evaluetes 250 points between the maximum
            and the minimum of X.

        ax : matplotlib.axes.Axes, default=None
            The current axes.

        data_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot`` for the data points.

        pred_kws : dict, default=None
            Additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot`` for the predictions values.

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

        X_eval = (
            np.linspace(X.min(), X.max(), 250).reshape(-1, 1)
            if X_eval is None
            else X_eval
        )
        ax.plot(X_eval, self.greg.predict(X_eval), **pred_kws)

        ax.set_xlabel("C-rates")
        ax.set_ylabel("SOC")

        ax.set_xscale("log")

        return ax
