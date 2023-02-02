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
    """GalvanostaticRegressor plot utilities.

    Kind of plots to produce:

    - 'surface' : the surface on which it was fitted.
    - 'in_surface' : fitted data points in the surface.
    - 'versus_data' : predicted values versus true data.

    Parametes
    ---------
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

        leval = np.linspace(
            np.min(self.greg._surface.ls),
            np.max(self.greg._surface.ls),
            num=1000,
        )
        chieval = np.linspace(
            np.min(self.greg._surface.chis),
            np.max(self.greg._surface.chis),
            num=1000,
        )

        z = self.greg._surface.spline(leval, chieval)
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

    def in_surface(self, X, ax=None, **kwargs):
        """Plot showing in which region of the map the fit is found.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates measurements.

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
        ax = self.surface() if ax is None else ax

        keys = ["color", "marker", "linestyle", "label"]
        for key, value in zip(keys, ["k", "o", "--", "fitted data"]):
            kwargs.setdefault(key, value)

        ax.plot(self.greg._logl(X), self.greg._logchi(X), **kwargs)

        return ax

    def versus_data(self, X, y, ax=None, data_kws=None, pred_kws=None):
        """Plot predictions against data.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C rates measurements.

        y : array-like
            Target State of Charge (SOC).

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
        ax.plot(xeval, self.greg.predict(xeval), **pred_kws)

        ax.set_xlabel("C-rates")
        ax.set_ylabel("SOC")

        ax.set_xscale("log")

        return ax
