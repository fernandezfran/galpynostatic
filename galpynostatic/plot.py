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

"""Plot helper for GalvanostaticRegressor object."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

from .utils import logell, logxi

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticPlotter:
    r"""GalvanostaticRegressor plot utilities.

    Kind of plots to produce:

    - 'render_map' : the map on which the data were fitted.
    - 'in_render_map' : :math:`\Xi` and :math:`\ell` data points in the map.
    - 'versus_data' : predicted and actual maximum SOC values versus C-rate.

    Parameters
    ----------
    greg : galpynostatic.model.GalvanostaticRegressor
        An already fitted GalvanostaticRegressor model.

    Notes
    -----
    The map will only be plotted in ``self.in_render_map(X)`` if ax is None,
    otherwise assumes it is already plotted and you just want to add
    the points on it, e.g. to compare different systems.
    """

    def __init__(self, greg):
        self.greg = greg

    def render_map(self, ax=None, clb=True, clb_label="maximum SOC"):
        """Plot the map on which data was fitted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The current axes.

        clb: bool, default=True
            Add the colorbar to the figure.

        clb_label : str, default="maximum SOC"
            The label for the color bar.
        """
        ax = plt.gca() if ax is None else ax

        logelleval = np.linspace(
            np.min(self.greg._map.logells_),
            np.max(self.greg._map.logells_),
            num=1000,
        )
        logxieval = np.linspace(
            np.min(self.greg._map.logxis_),
            np.max(self.greg._map.logxis_),
            num=1000,
        )

        z = self.greg._map.soc(logelleval, logxieval, grid=True)

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

        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")

        return ax

    def in_render_map(self, X, ax=None, **kwargs):
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
        """
        ax = self.render_map() if ax is None else ax

        keys = ["color", "marker", "linestyle", "label"]
        for key, value in zip(keys, ["k", "o", "--", "fitted data"]):
            kwargs.setdefault(key, value)

        ax.plot(
            logell(X.ravel(), self.greg.d, self.greg.z, self.greg.dcoeff_),
            logxi(X.ravel(), self.greg.dcoeff_, self.greg.k0_),
            **kwargs,
        )

        return ax

    def versus_data(
        self, X, y, X_eval=None, ax=None, data_kws=None, pred_kws=None
    ):
        """Plot SOC predictions against actual data as a function of C-rates.

        Parameters
        ----------
        X : array-like of shape (n_measurements, 1)
            C-rates used in experiments.

        y : array-like of shape (n_measurements,)
            Target maximum SOC values.

        X_eval : array-like of shape (n_measurements, 1), default=None.
            C-rates values to evalute the model and compare it with the data.
            When set to `None`, it evaluates 250 points between the maximum
            and minimum of X.

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
        ax.set_ylabel("maximum SOC")

        ax.set_xscale("log")

        return ax
