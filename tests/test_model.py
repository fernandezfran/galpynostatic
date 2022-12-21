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


def test_dcoeffs():
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    np.testing.assert_array_almost_equal(
        greg.dcoeffs, 10.0 ** np.arange(-15, -6, 0.1)
    )


def test_k0s():
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    np.testing.assert_array_almost_equal(
        greg.k0s, 10.0 ** np.arange(-14, -5, 0.1)
    )


@pytest.mark.parametrize(
    ("ref", "d", "C_rates", "xmaxs"),
    [
        (  # nishikawa data
            {"dcoeff": 1e-9, "k0": 1e-6, "mse": 0.00469549},
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array(
                [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
            ),
        ),
        (  # mancini data
            {"dcoeff": 1e-10, "k0": 1e-6, "mse": 0.00069059},
            0.00075,
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
    ],
)
def test_fit(ref, d, C_rates, xmaxs):
    """Test the fitting of the model: dcoeff, k0 and mse."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-11, -6, 1)
    greg.k0s = 10.0 ** np.arange(-10, -5, 1)

    greg = greg.fit(C_rates, xmaxs)

    np.testing.assert_almost_equal(greg.dcoeff_, ref["dcoeff"], 11)
    np.testing.assert_almost_equal(greg.k0_, ref["k0"], 10)
    np.testing.assert_almost_equal(greg.mse_, ref["mse"], 6)


@pytest.mark.parametrize(
    ("ref", "dcoeff", "k0", "C_rates"),
    [
        (  # nishikawa data
            np.array([0.937788, 0.878488, 0.81915, 0.701, 0.427025]),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
        ),
        (  # mancini data
            np.array(
                [
                    0.97696,
                    0.956831,
                    0.929995,
                    0.896439,
                    0.795823,
                    0.434239,
                    0.251205,
                    0.172261,
                    0.118832,
                ]
            ),
            1e-10,
            1e-6,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
        ),
    ],
)
def test_predict(ref, dcoeff, k0, C_rates):
    """Test the predict of the xmaxs values."""
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # fit results
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    xmaxs = greg.predict(C_rates)

    np.testing.assert_array_almost_equal(xmaxs, ref, 6)


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates", "xmaxs"),
    [
        (  # nishikawa data
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array(
                [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
            ),
        ),
        (  # mancini data
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
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_vs_data(fig_test, fig_ref, d, dcoeff, k0, C_rates, xmaxs):
    """Test the plot vs data points."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fitted res
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot_vs_data(C_rates, xmaxs, ax=test_ax)

    # ref plot
    ref_ax = fig_ref.subplots()
    ref_ax.plot(C_rates, xmaxs, marker="s", linestyle="--")
    ref_ax.plot(C_rates, greg.predict(C_rates), marker="o", linestyle="--")


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates"),
    [
        (  # nishikawa data
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
        ),
        (  # mancini data
            0.00075,
            1.0e-10,
            1.0e-6,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
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

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot_in_surface(C_rates, ax=test_ax)

    # ref plot
    fig_ref.axes[0].set_visible(False)
    ref_ax = fig_ref.subplots()

    # ref map
    ls = np.unique(DATASET.l)
    chis = np.unique(DATASET.chi)

    k, xmaxs = 0, []
    for l, chi in it.product(ls, chis[::-1]):
        xmax = 0
        try:
            if l == DATASET.l[k] and chi == DATASET.chi[k]:
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
    ref_ax.scatter(
        np.log10(greg._l(C_rates)),
        np.log10(greg._chi(C_rates)),
        color="k",
        linestyle="--",
        label="fitted data",
    )

    # ref labels and legend
    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")
    ref_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
