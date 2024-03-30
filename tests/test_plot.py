#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import galpynostatic.model
from galpynostatic.utils import logell, logxi

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest

import scipy.interpolate

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("experiment"),
    [
        ("nishikawa"),
        ("mancini"),
        ("he"),
        ("wang"),
        ("lei"),
        ("bak"),
        ("dokko"),
    ],
)
class TestPlots:
    """Test the plots with parametrization shared."""

    @check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
    def test_plot_versus_data(
        self, fig_test, fig_ref, experiment, request, spherical
    ):
        """Test the plot of predictions versus data points."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            d=experiment["d"], z=3
        )
        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
        greg._map = galpynostatic.base.MapSpline(spherical)

        # g reg plot
        test_ax = fig_test.subplots()
        greg.plot.versus_data(
            experiment["C_rates"], experiment["soc"], ax=test_ax
        )
        plt.clf()

        # ref plot
        ref_ax = fig_ref.subplots()
        ref_ax.plot(
            experiment["C_rates"],
            experiment["soc"],
            marker="s",
            linestyle="--",
        )

        xeval = np.linspace(
            experiment["C_rates"].min(), experiment["C_rates"].max(), 250
        ).reshape(-1, 1)
        ref_ax.plot(xeval, greg.predict(xeval), marker="", linestyle="-")

        ref_ax.set_xlabel("C-rates")
        ref_ax.set_ylabel("maximum SOC")
        ref_ax.set_xscale("log")

    @check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
    def test_plot_in_render_map(
        self, fig_test, fig_ref, experiment, request, spherical
    ):
        """Test the plot of data points in map."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            d=experiment["d"], z=3
        )
        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
        greg._map = galpynostatic.base.MapSpline(spherical)

        # g reg plot
        test_ax = fig_test.subplots()
        greg.plot.render_map(ax=test_ax)
        greg.plot.in_render_map(experiment["C_rates"], ax=test_ax)
        plt.clf()

        # ref plot
        ref_ax = fig_ref.subplots()

        # ref map
        ls = np.unique(spherical.l)
        xis = np.unique(spherical.xi)
        soc = spherical.xmax.to_numpy().reshape(ls.size, xis.size)[:, ::-1]

        spl = scipy.interpolate.RectBivariateSpline(ls, xis, soc)

        leval = np.linspace(np.min(ls), np.max(ls), num=1000)
        xieval = np.linspace(np.min(xis), np.max(xis), num=1000)
        z = np.clip(spl(leval, xieval), 0, 1)

        im = ref_ax.imshow(
            z.T,
            extent=[
                spherical.l.min(),
                spherical.l.max(),
                spherical.xi.min(),
                spherical.xi.max(),
            ],
            origin="lower",
        )
        clb = plt.colorbar(im)
        clb.ax.set_xlabel("")
        clb.ax.set_ylabel("maximum SOC")
        clb.ax.set_ylim((0, 1))
        ref_ax.scatter(spherical.l, spherical.xi, 400, facecolors="none")

        # ref data
        ref_ax.plot(
            logell(experiment["C_rates"], greg.d, greg.z, greg.dcoeff_),
            logxi(experiment["C_rates"], greg.dcoeff_, greg.k0_),
            color="k",
            marker="o",
            linestyle="--",
            label="fitted data",
        )

        # ref labels
        ref_ax.set_xlabel(r"log($\ell$)")
        ref_ax.set_ylabel(r"log($\Xi$)")
