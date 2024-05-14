#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

# import galpynostatic.simulation.spline
from .spline import SplineParams

from .Simulation import GalvanostaticMap, GalvanostaticProfile

import numpy as np

import pandas as pd

import pytest

import scipy

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("capacity", "potential", "refs"),
    [
        (
            [0, 1, 2],
            [0, 1, 4],
            [[0, 1, 4], [1.0, 4.0], [0.0, 6.0, 0.0], [4.0, -4.0]],
        ),
        (
            [0, 1, 2],
            [0, 1, 4],
            [[0, 1, 4], [1.0, 4.0], [0.0, 6.0, 0.0], [4.0, -4.0]],
        ),
    ],
)
def test_spline(capacity, potential, refs):
    """Test the spline of simulation module."""

    df = pd.DataFrame({"capacity": capacity, "potential": potential})

    spl = SplineParams(df)

    spl.iso_spline()

    np.testing.assert_array_almost_equal(spl.ai, refs[0], 6)
    np.testing.assert_array_almost_equal(spl.bi, refs[1], 6)
    np.testing.assert_array_almost_equal(spl.ci, refs[2], 6)
    np.testing.assert_array_almost_equal(spl.di, refs[3], 6)


@pytest.mark.parametrize(
    ("method", "isotherm", "refs"),
    [
        (
            "CN",
            False,
            [
                [0.238529, 0.0, 0.972238],
                [-0.000523, -0.133836, 0.201108],
                PATH / "test_data" / "profileCN.csv",
                PATH / "test_data" / "conCN.csv",
            ],
        ),
        (
            "CN",
            PATH / "LMO-1C.csv",
            [
                [0.262806, 0.0, 0.972238],
                [2.219972, 0.0, 4.39561],
                PATH / "test_data" / "profileCN_iso.csv",
                PATH / "test_data" / "conCN_iso.csv",
            ],
        ),
        (
            "BI",
            False,
            [
                [0.238529, 0.0, 0.972238],
                [-0.000523, -0.133836, 0.201108],
                PATH / "test_data" / "profileBI.csv",
                PATH / "test_data" / "conBI.csv",
            ],
        ),
        (
            "BI",
            PATH / "LMO-1C.csv",
            [
                [0.262806, 0.0, 0.972238],
                [2.219972, 0.0, 4.39561],
                PATH / "test_data" / "profileBI_iso.csv",
                PATH / "test_data" / "conBI_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticProfile:
    def test_profile_soc(self, method, isotherm, refs):
        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            method=method,
            L=-1,
            xi=1,
            Npt=20000,
            isotherm=isotherm,
        )
        profile.calc()

        np.testing.assert_almost_equal(np.mean(profile.SOC), refs[0][0], 6)
        np.testing.assert_almost_equal(np.min(profile.SOC), refs[0][1], 6)
        np.testing.assert_almost_equal(np.max(profile.SOC), refs[0][2], 6)
    
    def test_potential(self, method, isotherm, refs):
        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            method=method,
            L=-1,
            xi=1,
            Npt=20000,
            isotherm=isotherm,
        )
        profile.calc()

        np.testing.assert_almost_equal(np.mean(profile.E), refs[1][0], 6)
        np.testing.assert_almost_equal(np.min(profile.E), refs[1][1], 6)
        np.testing.assert_almost_equal(np.max(profile.E), refs[1][2], 6)

        
    def test_profile_dataframe(self, method, isotherm, refs):
        df = pd.read_csv(refs[2])

        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            L=-1,
            xi=1,
            Npt=20000,
            method=method,
            isotherm=isotherm,
        )
        profile.calc()

        pd.testing.assert_frame_equal(profile._df, df)

    def test_concentration_dataframe(self, method, isotherm, refs):
        df = pd.read_csv(refs[3])

        profile = GalvanostaticProfile(
            180.815,
            2.26e-3,
            4.58,
            L=-1,
            xi=1,
            Npt=20000,
            method=method,
            isotherm=isotherm,
        )
        profile.calc()

        pd.testing.assert_frame_equal(profile.condf, df)
    

@pytest.mark.parametrize(
    ("method", "isotherm"),
    [
        ("CN", False),
        ("CN", PATH / "LMO-1C.csv"),
        ("BI", False),
        ("BI", PATH / "LMO-1C.csv"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_plot_profile(fig_test, fig_ref, method, isotherm):
    profile = GalvanostaticProfile(
        180.815,
        2.26e-3,
        4.58,
        L=-1,
        xi=1,
        Npt=20000,
        method=method,
        isotherm=isotherm,
    )
    profile.calc()

    test_ax = fig_test.subplots()
    profile.plot(ax=test_ax)

    ref_ax = fig_ref.subplots()
    ref_ax.plot(profile._df["SOC"], profile._df["Potential"])

    ref_ax.set_xlabel("SOC")
    ref_ax.set_ylabel("Potential")
    ref_ax.set_title("Isotherm")


@pytest.mark.parametrize(
    ("method", "isotherm", "refs"),
    [
        (
            "CN",
            False,
            [
                [0.425895, 0.000100, 0.998085],
                PATH / "test_data" / "mapCN.csv",
            ],
        ),
        (
            "CN",
            PATH / "LMO-1C.csv",
            [
                [0.599868, 7.44792e-7, 1.000937],
                PATH / "test_data" / "mapCN_iso.csv",
            ],
        ),
        (
            "BI",
            False,
            [
                [0.425895, 0.000100, 0.998085],
                PATH / "test_data" / "mapBI.csv",
            ],
        ),
        (
            "BI",
            PATH / "LMO-1C.csv",
            [
                [0.599868, 7.44792e-7, 1.000937],
                PATH / "test_data" / "mapBI_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticMap:
    def test_map_soc(self, method, isotherm, refs):
        galvamap = GalvanostaticMap(
            180.815,
            2.26e-3,
            4.58,
            Npt=20000,
            NL=3,
            Nxi=3,
            method=method,
            isotherm=isotherm,
        )
        galvamap.calc()

        np.testing.assert_almost_equal(np.mean(galvamap.SOC), refs[0][0], 4)
        np.testing.assert_almost_equal(np.min(galvamap.SOC), refs[0][1], 4)
        np.testing.assert_almost_equal(np.max(galvamap.SOC), refs[0][2], 6)

    def test_map_dataframe(self, method, isotherm, refs):
        df = pd.read_csv(refs[1])

        galvamap = GalvanostaticMap(
            180.815,
            2.26e-3,
            4.58,
            NL=2,
            Nxi=2,
            Npt=20000,
            method=method,
            isotherm=isotherm,
        )
        galvamap.calc()

        pd.testing.assert_frame_equal(galvamap.to_dataframe().reset_index(drop=True), df)


@pytest.mark.parametrize(
    ("method", "isotherm"),
    [
        ("CN", False),
        ("CN", PATH / "LMO-1C.csv"),
        ("BI", False),
        ("BI", PATH / "LMO-1C.csv"),
    ],
)
@check_figures_equal(extensions=["png", "pdf"], tol=0.000001)
def test_map_plot(fig_test, fig_ref, method, isotherm):
    galvamap = GalvanostaticMap(
        180.815,
        2.26e-3,
        4.58,
        NL=4,
        Nxi=4,
        Npt=20000,
        method=method,
        isotherm=isotherm,
    )
    galvamap.calc()

    test_ax = fig_test.subplots()
    galvamap.plot(ax=test_ax)

    plt.clf()
    ref_ax = fig_ref.subplots()

    x = galvamap._df.L
    y = galvamap._df.xi

    logells_ = np.unique(x)
    logxis_ = np.unique(y)
    socs = galvamap._df.SOC.to_numpy().reshape(logells_.size, logxis_.size)

    spline_ = scipy.interpolate.RectBivariateSpline(logells_, logxis_, socs)

    xeval = np.linspace(x.min(), x.max(), 1000)
    yeval = np.linspace(y.min(), y.max(), 1000)

    z = spline_(xeval, yeval, grid=True)

    im = ref_ax.imshow(
        z.T,
        extent=[
            xeval.min(),
            xeval.max(),
            yeval.min(),
            yeval.max(),
        ],
        origin="lower",
    )

    clb = plt.colorbar(im, ax=ref_ax)
    clb.ax.set_ylabel("SOC")
    clb.ax.set_ylim((0, 1))

    ref_ax.set_xlabel(r"log($\ell$)")
    ref_ax.set_ylabel(r"log($\Xi$)")

    plt.close(fig_ref)
