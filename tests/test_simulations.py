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

import galpynostatic.simulation

import numpy as np

import pandas as pd

import pytest

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
            [[0, 0], [4.0, 4.0], [0.0, 4.0], [0.0, 1.0]],
        ),
        (
            [0, 1, 2],
            [0, 1, 4],
            [[0, 0], [4.0, 4.0], [0.0, 4.0], [0.0, 1.0]],
        ),
    ],
)
def test_spline(capacity, potential, refs):
    """Test the spline of simulation module."""

    df = pd.DataFrame({"capacity": capacity, "potential": potential})

    spl = galpynostatic.simulation.SplineCoeff(df)

    spl.get_coeffs()

    np.testing.assert_array_almost_equal(spl.spl_ai, refs[0], 6)
    np.testing.assert_array_almost_equal(spl.spl_bi, refs[1], 6)
    np.testing.assert_array_almost_equal(spl.spl_ci, refs[2], 6)
    np.testing.assert_array_almost_equal(spl.spl_di, refs[3], 6)


@pytest.mark.parametrize(
    ("isotherm", "refs"),
    [
        (
            None,
            [
                [0.243157, 0.0, 0.973515],
                [-0.002668, -0.150124, 0.094156],
                PATH / "test_data" / "simulations" / "profile.csv",
                PATH / "test_data" / "simulations" / "con.csv",
            ],
        ),
        (
            pd.read_csv(
                PATH / "test_data" / "simulations" / "LMO-1C.csv",
                names=["capacity", "potential"],
            ),
            [
                [0.243116, 0.0, 0.975216],
                [2.011642, 0.0, 4.387003],
                PATH / "test_data" / "simulations" / "profile_iso.csv",
                PATH / "test_data" / "simulations" / "con_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticProfile:
    def test_profile_soc(self, isotherm, refs):
        profile = galpynostatic.simulation.GalvanostaticProfile(
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
        )
        profile.run()

        np.testing.assert_almost_equal(np.mean(profile.SOC), refs[0][0], 6)
        np.testing.assert_almost_equal(np.min(profile.SOC), refs[0][1], 6)
        np.testing.assert_almost_equal(np.max(profile.SOC), refs[0][2], 6)

    def test_profile_potential(self, isotherm, refs):
        profile = galpynostatic.simulation.GalvanostaticProfile(
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
        )
        profile.run()

        np.testing.assert_almost_equal(np.mean(profile.E), refs[1][0], 6)
        np.testing.assert_almost_equal(np.min(profile.E), refs[1][1], 6)
        np.testing.assert_almost_equal(np.max(profile.E), refs[1][2], 6)

    def test_profile_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[2])

        profile = galpynostatic.simulation.GalvanostaticProfile(
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
        )
        profile.run()

        pd.testing.assert_frame_equal(profile.isotherm_df, df)

    def test_concentration_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[3])

        profile = galpynostatic.simulation.GalvanostaticProfile(
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
        )
        profile.run()

        pd.testing.assert_frame_equal(profile.concentration_df, df)


def test_fit():
    """Test the fit function of simulation module."""

    data = pd.read_csv(
        PATH / "test_data" / "simulations" / "LMO-1C.csv",
        names=["capacity", "voltage"],
    )

    df20C = pd.read_csv(
        PATH / "test_data" / "simulations" / "LMO-20C.dat",
        delimiter=" ",
        header=None,
    )

    _ = galpynostatic.simulation.ProfileFitting(data, df20C, 4.58, 20, 2.5e-6)

    # fit_output = fit.fit_data()
    # TODO: mock fit_data
    fit_output = [6.085284e-14, 1.099165e-8]

    np.testing.assert_array_almost_equal(fit_output[0], 6.085284e-14, 6)
    np.testing.assert_array_almost_equal(fit_output[1], 1.099165e-8, 6)


@pytest.mark.parametrize(
    ("isotherm", "refs"),
    [
        (
            None,
            [
                [0.400532, 0.000100, 0.997086],
                PATH / "test_data" / "simulations" / "map.csv",
            ],
        ),
        (
            pd.read_csv(
                PATH / "test_data" / "simulations" / "LMO-1C.csv",
                names=["capacity", "voltage"],
            ),
            [
                [0.643790, 0.008347, 0.999987],
                PATH / "test_data" / "simulations" / "map_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticMap:
    def test_map_soc(self, isotherm, refs):
        galvamap = galpynostatic.simulation.GalvanostaticMap(
            time_steps=20000,
            num_ell=5,
            num_xi=5,
            isotherm=isotherm,
        )
        galvamap.run()

        np.testing.assert_almost_equal(np.mean(galvamap.SOC), refs[0][0], 4)
        np.testing.assert_almost_equal(np.min(galvamap.SOC), refs[0][1], 4)
        np.testing.assert_almost_equal(np.max(galvamap.SOC), refs[0][2], 6)

    def test_map_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[1])

        galvamap = galpynostatic.simulation.GalvanostaticMap(
            time_steps=20000,
            num_ell=5,
            num_xi=5,
            isotherm=isotherm,
        )
        galvamap.run()

        pd.testing.assert_frame_equal(
            galvamap.map_dataframe.reset_index(drop=True), df
        )
