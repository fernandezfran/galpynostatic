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
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, 1.0, 3)

    np.testing.assert_array_almost_equal(
        greg.dcoeffs, np.logspace(-15, -6, num=100)
    )


def test_k0s():
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, 1.0, 3)

    np.testing.assert_array_almost_equal(
        greg.k0s, np.logspace(-14, -5, num=100)
    )


@pytest.mark.parametrize(
    ("ref", "d", "C_rates", "xmaxs"),
    [  # nishikawa, mancini, he, wang data
        (
            {"dcoeff": 1e-9, "k0": 1e-6, "mse": 0.00469549},
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array(
                [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
            ),
        ),
        (
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
        (
            {"dcoeff": 1e-11, "k0": 1e-8, "mse": 0.006482},
            0.000175,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
            np.array([0.995197, 0.958646, 0.845837, 0.654458, 0.346546]),
        ),
        (
            {"dcoeff": 1e-8, "k0": 1e-6, "mse": 0.000863},
            0.002,
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
    ],
)
def test_fit(ref, d, C_rates, xmaxs):
    """Test the fitting of the model: dcoeff, k0 and mse."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-12, -6, 1)
    greg.k0s = 10.0 ** np.arange(-10, -5, 1)

    greg = greg.fit(C_rates, xmaxs)

    np.testing.assert_almost_equal(greg.dcoeff_, ref["dcoeff"], 12)
    np.testing.assert_almost_equal(greg.k0_, ref["k0"], 10)
    np.testing.assert_almost_equal(greg.mse_, ref["mse"], 6)


@pytest.mark.parametrize(
    ("ref", "d", "dcoeff", "k0", "C_rates"),
    [  # nishikawa, mancini, he, wang data
        (
            np.array([0.937788, 0.878488, 0.81915, 0.701, 0.427025]),
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            1.0e-09,
            1.0e-6,
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
        ),
        (
            np.array(
                [
                    0.979367,
                    0.961645,
                    0.938008,
                    0.908508,
                    0.819804,
                    0.48611,
                    0.290725,
                    0.197544,
                    0.135119,
                ]
            ),
            0.00075,
            1e-10,
            1e-6,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
        ),
        (
            np.array([0.978918, 0.906247, 0.815342, 0.633649, 0.179112]),
            0.000175,
            1.0e-11,
            1.0e-8,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
        ),
        (
            np.array(
                [0.985938, 0.974779, 0.952477, 0.885544, 0.774095, 0.550382]
            ),
            0.002,
            1e-8,
            1e-6,
            np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
        ),
    ],
)
def test_predict(ref, d, dcoeff, k0, C_rates):
    """Test the predict of the xmaxs values."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fit results
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()
    xmaxs = greg.predict(C_rates)

    np.testing.assert_array_almost_equal(xmaxs, ref, 6)


@pytest.mark.parametrize(
    ("ref", "d", "dcoeff", "k0"),
    [  # nishikawa, mancini, he, wang data
        (6.501643, np.sqrt(0.25 * 8.04e-6 / np.pi), 1.0e-09, 1.0e-6),
        (2.213407, 0.00075, 1e-10, 1e-6),
        (0.280568, 0.000175, 1.0e-11, 1.0e-8),
        (16.25661, 0.002, 1e-8, 1e-6),
    ],
)
def test_t_minutes_lenght(ref, d, dcoeff, k0):
    """Test the t minutes lenght."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fit results
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()
    lenght = greg.t_minutes_lenght()

    np.testing.assert_array_almost_equal(lenght, ref, 6)


def test_t_minutes_raise():
    """Test the t minutes lenght ValueError raise."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, 0.005, 3)

    # fictional fit results
    greg.dcoeff_ = 3e-5
    greg.k0_ = 1e-7

    greg._surface()
    with pytest.raises(ValueError):
        greg.t_minutes_lenght()


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates", "xmaxs"),
    [  # nishikawa, mancini, he, wang data
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
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_vs_data(fig_test, fig_ref, d, dcoeff, k0, C_rates, xmaxs):
    """Test the plot vs data points."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fitted res
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot_vs_data(C_rates, xmaxs, ax=test_ax)

    # ref plot
    ref_ax = fig_ref.subplots()
    ref_ax.plot(C_rates, xmaxs, marker="s", linestyle="--")

    xeval = np.linspace(C_rates.min(), C_rates.max(), 250).reshape(-1, 1)
    ref_ax.plot(xeval, greg.predict(xeval), marker="", linestyle="-")


@pytest.mark.parametrize(
    ("d", "dcoeff", "k0", "C_rates"),
    [  # nishikawa, mancini, he, wang data
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
    greg._plot_surface(ax=test_ax)
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
    ref_ax.plot(
        np.log10(greg._l(C_rates)),
        np.log10(greg._chi(C_rates)),
        color="k",
        marker="o",
        linestyle="--",
        label="fitted data",
    )

    # ref labels
    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")
