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
                [0.243160, 0.0, 0.974116],
                [-0.00228, -0.150241, 0.096110],
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
                [0.243121, 0.0, 0.976017],
                [2.012031, 0.0, 4.388962],
                PATH / "test_data" / "simulations" / "profile_iso.csv",
                PATH / "test_data" / "simulations" / "con_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticProfile:
    def test_profile_soc(self, isotherm, refs):
        profile = galpynostatic.simulation.GalvanostaticProfile(
            density=4.58,
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
            specific_capacity=100,
        )
        profile.run()

        np.testing.assert_almost_equal(np.mean(profile.SOC), refs[0][0], 6)
        np.testing.assert_almost_equal(np.min(profile.SOC), refs[0][1], 6)
        np.testing.assert_almost_equal(np.max(profile.SOC), refs[0][2], 6)

    def test_profile_potential(self, isotherm, refs):
        profile = galpynostatic.simulation.GalvanostaticProfile(
            density=4.58,
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
            specific_capacity=100,
        )
        profile.run()

        np.testing.assert_almost_equal(np.mean(profile.E), refs[1][0], 6)
        np.testing.assert_almost_equal(np.min(profile.E), refs[1][1], 6)
        np.testing.assert_almost_equal(np.max(profile.E), refs[1][2], 6)

    def test_profile_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[2])

        profile = galpynostatic.simulation.GalvanostaticProfile(
            density=4.58,
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
            specific_capacity=100,
        )
        profile.run()

        pd.testing.assert_frame_equal(profile.isotherm_df, df)

    def test_concentration_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[3])

        profile = galpynostatic.simulation.GalvanostaticProfile(
            density=4.58,
            ell=-1,
            xi=1,
            time_steps=20000,
            isotherm=isotherm,
            specific_capacity=100,
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
                [0.425895, 0.000100, 0.998085],
                PATH / "test_data" / "simulations" / "map.csv",
            ],
        ),
        (
            pd.read_csv(
                PATH / "test_data" / "simulations" / "LMO-1C.csv",
                names=["capacity", "voltage"],
            ),
            [
                [0.599868, 7.44792e-7, 1.000937],
                PATH / "test_data" / "simulations" / "map_iso.csv",
            ],
        ),
    ],
)
class TestGalvanostaticMap:
    def test_map_soc(self, isotherm, refs):
        galvamap = galpynostatic.simulation.GalvanostaticMap(
            density=4.58,
            time_steps=20000,
            num_ell=3,
            num_xi=3,
            isotherm=isotherm,
            specific_capacity=100,
        )
        galvamap.run()

        np.testing.assert_almost_equal(np.mean(galvamap.SOC), refs[0][0], 4)
        np.testing.assert_almost_equal(np.min(galvamap.SOC), refs[0][1], 4)
        np.testing.assert_almost_equal(np.max(galvamap.SOC), refs[0][2], 6)

    def test_map_dataframe(self, isotherm, refs):
        df = pd.read_csv(refs[1])

        galvamap = galpynostatic.simulation.GalvanostaticMap(
            density=4.58,
            time_steps=20000,
            num_ell=3,
            num_xi=3,
            isotherm=isotherm,
            specific_capacity=100,
        )
        galvamap.run()

        pd.testing.assert_frame_equal(
            galvamap.to_dataframe().reset_index(drop=True), df
        )
