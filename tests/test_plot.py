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

import galpynostatic.datasets
import galpynostatic.model

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest

import scipy.interpolate

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET = galpynostatic.datasets.load_spherical()

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates", "xmaxs"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array(
                [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
            ),
        ),
        (
            0.00075,
            1.0e-10,
            1.0e-6,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
            np.array(
                [
                    0.99244268,
                    0.98205007,
                    0.96473524,
                    0.93494309,
                    0.85388692,
                    0.54003011,
                    0.29684304,
                    0.19500165,
                    0.12502474,
                ]
            ),
        ),
        (
            0.000175,
            1.0e-11,
            1.0e-8,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
            np.array([0.995197, 0.958646, 0.845837, 0.654458, 0.346546]),
        ),
        (
            0.002,
            1e-8,
            1e-6,
            np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
            np.array(
                [
                    0.99417946,
                    0.9675683,
                    0.9301233,
                    0.8345085,
                    0.73432816,
                    0.5696607,
                ]
            ),
        ),
        (
            3.5e-5,
            1e-13,
            1e-8,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
            np.array(
                [
                    0.9489587,
                    0.8360887,
                    0.7596236,
                    0.32932344,
                    0.02090921,
                    0.00960977,
                ]
            ),
        ),
        (
            2.5e-6,
            1e-14,
            1e-8,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
            np.array([0.9617, 0.938762, 0.9069, 0.863516, 0.696022, 0.421418]),
        ),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_versus_data(fig_test, fig_ref, d, dcoeff, k0, C_rates, xmaxs):
    """Test the plot vs data points."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fitted res
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot.versus_data(C_rates, xmaxs, ax=test_ax)

    # ref plot
    ref_ax = fig_ref.subplots()
    ref_ax.plot(C_rates, xmaxs, marker="s", linestyle="--")

    xeval = np.linspace(C_rates.min(), C_rates.max(), 250).reshape(-1, 1)
    ref_ax.plot(xeval, greg.predict(xeval), marker="", linestyle="-")

    ref_ax.set_xlabel("C-rates")
    ref_ax.set_ylabel("SOC")
    ref_ax.set_xscale("log")


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
        ),
        (
            0.00075,
            1.0e-10,
            1.0e-6,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
        ),
        (
            0.000175,
            1.0e-11,
            1.0e-8,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
        ),
        (
            0.002,
            1e-8,
            1e-6,
            np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
        ),
        (
            3.5e-5,
            1e-13,
            1e-8,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
        ),
        (
            2.5e-6,
            1e-14,
            1e-8,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
        ),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_in_surface(fig_test, fig_ref, d, dcoeff, k0, C_rates):
    """Test the plot vs data points."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # nishikawa fitted res
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot.surface(ax=test_ax)
    greg.plot.in_surface(C_rates, ax=test_ax)

    # ref plot
    fig_ref.axes[0].set_visible(False)
    ref_ax = fig_ref.subplots()

    # ref map
    ls = np.unique(DATASET.l)
    chis = np.unique(DATASET.chi)

    k, xmaxs = 0, []
    for logl, logchi in it.product(ls, chis[::-1]):
        xmax = 0
        try:
            if logl == DATASET.l[k] and logchi == DATASET.chi[k]:
                xmax = DATASET.xmax[k]
                k += 1
        except KeyError:
            ...
        finally:
            xmaxs.append(xmax)

    xmaxs = np.asarray(xmaxs).reshape(ls.size, chis.size)[:, ::-1]

    spl = scipy.interpolate.RectBivariateSpline(ls, chis, xmaxs)

    leval = np.linspace(np.min(ls), np.max(ls), num=1000)
    chieval = np.linspace(np.min(chis), np.max(chis), num=1000)
    z = spl(leval, chieval)
    z[z > 1] = 1.0
    z[z < 0] = 0.0

    im = ref_ax.imshow(
        z.T,
        extent=[
            DATASET.l.min(),
            DATASET.l.max(),
            DATASET.chi.min(),
            DATASET.chi.max(),
        ],
        origin="lower",
    )
    clb = plt.colorbar(im)
    clb.ax.set_xlabel("")
    clb.ax.set_ylabel(r"x$_{max}$")
    clb.ax.set_ylim((0, 1))
    ref_ax.scatter(DATASET.l, DATASET.chi, 400, facecolors="none")

    # ref data
    ref_ax.plot(
        greg._logl(C_rates),
        greg._logchi(C_rates),
        color="k",
        marker="o",
        linestyle="--",
        label="fitted data",
    )

    # ref labels
    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")
