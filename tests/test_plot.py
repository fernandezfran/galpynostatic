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
        ("nishikawa_experiment"),
        ("mancini_experiment"),
        ("he_experiment"),
        ("wang_experiment"),
        ("lei_experiment"),
        ("bak_experiment"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_versus_data(fig_test, fig_ref, experiment, request, spherical):
    """Test the plot vs data points."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(
        spherical, experiment["d"], 3
    )

    # fitted res
    greg.dcoeff_ = experiment["dcoeff"]
    greg.k0_ = experiment["k0"]

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot.versus_data(experiment["C_rates"], experiment["soc"], ax=test_ax)

    # ref plot
    ref_ax = fig_ref.subplots()
    ref_ax.plot(
        experiment["C_rates"], experiment["soc"], marker="s", linestyle="--"
    )

    xeval = np.linspace(
        experiment["C_rates"].min(), experiment["C_rates"].max(), 250
    ).reshape(-1, 1)
    ref_ax.plot(xeval, greg.predict(xeval), marker="", linestyle="-")

    ref_ax.set_xlabel("C-rates")
    ref_ax.set_ylabel("SOC")
    ref_ax.set_xscale("log")


@pytest.mark.parametrize(
    ("experiment"),
    [
        ("nishikawa_experiment"),
        ("mancini_experiment"),
        ("he_experiment"),
        ("wang_experiment"),
        ("lei_experiment"),
        ("bak_experiment"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_in_surface(fig_test, fig_ref, experiment, request, spherical):
    """Test the plot vs data points."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(
        spherical, experiment["d"], 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = experiment["dcoeff"]
    greg.k0_ = experiment["k0"]

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
    for logl, logchi in it.product(ls, chis[::-1]):
        xmax = 0
        try:
            if logl == spherical.l[k] and logchi == spherical.chi[k]:
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
    clb.ax.set_ylabel(r"x$_{max}$")
    clb.ax.set_ylim((0, 1))
    ref_ax.scatter(spherical.l, spherical.chi, 400, facecolors="none")

    # ref data
    ref_ax.plot(
        greg._logl(experiment["C_rates"]),
        greg._logchi(experiment["C_rates"]),
        color="k",
        marker="o",
        linestyle="--",
        label="fitted data",
    )

    # ref labels
    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")
