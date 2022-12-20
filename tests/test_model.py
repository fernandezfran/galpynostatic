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


def test_fit():
    """Test the fitting of the model: dcoeff, k0 and mse."""
    # reference values
    ref_dcoeff = 1e-9
    ref_k0 = 1e-6
    ref_mse = 0.004695

    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-10, -6, 1)
    greg.k0s = 10.0 ** np.arange(-9, -5, 1)

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1)
    xmaxs = np.array(
        [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
    )

    # fit
    greg = greg.fit(crates, xmaxs)

    # tests
    np.testing.assert_almost_equal(greg.dcoeff_, ref_dcoeff, 10)
    np.testing.assert_almost_equal(greg.k0_, ref_k0, 7)
    np.testing.assert_almost_equal(greg.mse_, ref_mse, 6)


def test_predict():
    """Test the predict of the xmaxs values."""
    # reference xmaxs predictions
    ref = np.array([0.937788, 0.878488, 0.81915, 0.701, 0.427025])

    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = 1.0e-09
    greg.k0_ = 1.0e-6

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1)

    # predict
    xmaxs = greg.predict(crates)

    # tests
    np.testing.assert_array_almost_equal(xmaxs, ref, 6)


@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_vs_data(fig_test, fig_ref):
    """Test the plot vs data points."""
    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = 1.0e-09
    greg.k0_ = 1.0e-6

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1)
    xmaxs = np.array(
        [0.99656589, 0.97625474, 0.83079658, 0.72518132, 0.52573576]
    )

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot_vs_data(crates, xmaxs, ax=test_ax)

    # ref plot
    ref_ax = fig_ref.subplots()
    ref_ax.plot(crates, xmaxs, marker="s", linestyle="--")
    ref_ax.plot(crates, greg.predict(crates), marker="o", linestyle="--")


@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_in_surface(fig_test, fig_ref):
    """Test the plot vs data points."""
    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = 1.0e-09
    greg.k0_ = 1.0e-6

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1)

    # g reg plot
    test_ax = fig_test.subplots()
    greg.plot_in_surface(crates, ax=test_ax)

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
        np.log10(greg._l(crates)),
        np.log10(greg._chi(crates)),
        color="k",
        linestyle="--",
        label="fitted data",
    )

    # ref labels and legend
    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")
    ref_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
