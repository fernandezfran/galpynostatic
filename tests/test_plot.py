#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

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
            spherical, experiment["d"], 3
        )

        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]

        # g reg plot
        test_ax = fig_test.subplots()
        greg.plot.versus_data(
            experiment["C_rates"], experiment["soc"], ax=test_ax
        )

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
    def test_plot_in_surface(
        self, fig_test, fig_ref, experiment, request, spherical
    ):
        """Test the plot of data points in surface."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            spherical, experiment["d"], 3
        )

        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]

        # g reg plot
        test_ax = fig_test.subplots()
        greg.plot.surface(ax=test_ax)
        greg.plot.in_surface(experiment["C_rates"], ax=test_ax)

        # ref plot
        fig_ref.axes[0].set_visible(False)
        ref_ax = fig_ref.subplots()

        # ref map
        ls = np.unique(spherical.l)
        chis = np.unique(spherical.chi)

        k, soc = 0, []
        for logell_value, logxi_value in it.product(ls, chis[::-1]):
            xmax = 0
            try:
                if (
                    logell_value == spherical.l[k]
                    and logxi_value == spherical.chi[k]
                ):
                    xmax = spherical.xmax[k]
                    k += 1
            except KeyError:
                ...
            finally:
                soc.append(xmax)

        soc = np.asarray(soc).reshape(ls.size, chis.size)[:, ::-1]

        spl = scipy.interpolate.RectBivariateSpline(ls, chis, soc)

        leval = np.linspace(np.min(ls), np.max(ls), num=1000)
        chieval = np.linspace(np.min(chis), np.max(chis), num=1000)
        z = spl(leval, chieval)
        z[z > 1] = 1.0
        z[z < 0] = 0.0

        im = ref_ax.imshow(
            z.T,
            extent=[
                spherical.l.min(),
                spherical.l.max(),
                spherical.chi.min(),
                spherical.chi.max(),
            ],
            origin="lower",
        )
        clb = plt.colorbar(im)
        clb.ax.set_xlabel("")
        clb.ax.set_ylabel("maximum SOC")
        clb.ax.set_ylim((0, 1))
        ref_ax.scatter(spherical.l, spherical.chi, 400, facecolors="none")

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
