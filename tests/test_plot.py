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

import os
import pathlib

import galpynostatic.model
from galpynostatic.utils import logell, logxi
import galpynostatic.simulation as si

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import scipy.interpolate

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

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


@pytest.mark.parametrize(
    ("isotherm"),
    [
        (None),
        (PATH / "test_data" / "simulations" / "LMO-1C.csv"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_isotherm_plot(fig_test, fig_ref, isotherm):
    profile = si.GalvanostaticProfile(
        4.58,
        ell=-1,
        xi=1,
        time_steps=20000,
        isotherm=isotherm,
        specific_capacity=100,
    )
    profile.run()

    test_ax = fig_test.subplots()
    profile.isotherm_plot(ax=test_ax)

    ref_ax = fig_ref.subplots()
    ref_ax.plot(
        profile.isotherm_df["SOC"], 
        profile.isotherm_df["Potential"]
        )

    ref_ax.set_xlabel("SoC")
    ref_ax.set_ylabel("Potential")


@pytest.mark.parametrize(
    ("isotherm"),
    [
        (None),
        (PATH / "test_data" / "simulations" / "LMO-1C.csv"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_consentration_plot(fig_test, fig_ref, isotherm):
    profile = si.GalvanostaticProfile(
        4.58,
        ell=-1,
        xi=1,
        time_steps=20000,
        isotherm=isotherm,
        specific_capacity=100,
    )
    profile.run()

    test_ax = fig_test.subplots()
    profile.consentration_plot(ax=test_ax)

    ref_ax = fig_ref.subplots()
    ref_ax.plot(
        profile.concentration_df["r_norm"], 
        profile.concentration_df["theta"],
        color="tab:red")

    ref_ax.set_xlabel("$r_{norm}$")
    ref_ax.set_ylabel(r"$\theta$")


@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_fit_plot(fig_test, fig_ref):
    data = pd.read_csv(
        PATH / "test_data" / "simulations" / "LMO-1C.csv", 
        names=['capacity', 'voltage']
        )

    df20C = pd.read_csv(
        PATH / "test_data" / "simulations" / "LMO-20C.dat", 
        delimiter=' ', 
        header=None
        )

    fit = si.ProfileFitting(data, df20C, 4.58, 20, 2.5e-6)
    _, _ = fit.fit_data()

    test_ax = fig_test.subplots()
    fit.plot_fit(ax=test_ax)

    iso = si.GalvanostaticProfile(
            fit.density,
            fit.logxi,
            fit.logell,
            isotherm=fit.isotherm,
        )
    iso.run()        

    ref_ax = fig_ref.subplots()
    ref_ax.plot(iso.isotherm_df["SOC"], iso.isotherm_df["Potential"])

    ref_ax.set_xlabel("SoC")
    ref_ax.set_ylabel("Potential / V")