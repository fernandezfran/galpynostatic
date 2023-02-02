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

import os
import pathlib

import galpynostatic.datasets
import galpynostatic.model

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

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
    ("ref", "d", "C_rates", "soc"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (
            {"dcoeff": 1e-9, "k0": 1e-6, "mse": 0.00469549},
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array([0.996566, 0.976255, 0.830797, 0.725181, 0.525736]),
        ),
        (
            {"dcoeff": 1e-10, "k0": 1e-6, "mse": 0.00069059},
            0.00075,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
            np.array(
                [
                    0.992443,
                    0.98205,
                    0.964735,
                    0.934943,
                    0.853887,
                    0.54003,
                    0.296843,
                    0.195002,
                    0.125025,
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
                [0.994179, 0.967568, 0.930123, 0.834509, 0.734328, 0.569661]
            ),
        ),
        (
            {"dcoeff": 1e-13, "k0": 1e-8, "mse": 0.006019},
            3.5e-5,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
            np.array(
                [0.948959, 0.836089, 0.759624, 0.329323, 0.020909, 0.00961]
            ),
        ),
        (
            {"dcoeff": 1e-14, "k0": 1e-8, "mse": 0.016352},
            2.5e-6,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
            np.array([0.9617, 0.938762, 0.9069, 0.863516, 0.696022, 0.421418]),
        ),
    ],
)
def test_fit(ref, d, C_rates, soc):
    """Test the fitting of the model: dcoeff, k0 and mse."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-14, -6, 1)
    greg.k0s = 10.0 ** np.arange(-13, -5, 1)

    greg = greg.fit(C_rates, soc)

    np.testing.assert_almost_equal(greg.dcoeff_, ref["dcoeff"], 12)
    np.testing.assert_almost_equal(greg.k0_, ref["k0"], 10)
    np.testing.assert_almost_equal(greg.mse_, ref["mse"], 6)


@pytest.mark.parametrize(
    ("ref", "d", "dcoeff", "k0", "C_rates"),
    [  # nishikawa, mancini, he, wang, lei, bak data
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
        (
            np.array(
                [0.918072, 0.799457, 0.604216, 0.325994, 0.112047, 0.046371]
            ),
            3.5e-5,
            1e-13,
            1e-8,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
        ),
        (
            np.array(
                [0.976568, 0.894486, 0.791879, 0.589535, 0.235351, 0.101137]
            ),
            2.5e-6,
            1e-14,
            1e-8,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
        ),
    ],
)
def test_predict(ref, d, dcoeff, k0, C_rates):
    """Test the predict of the soc values."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fit results
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    soc = greg.predict(C_rates)

    np.testing.assert_array_almost_equal(soc, ref, 6)


@pytest.mark.parametrize(
    ("ref", "d", "C_rates", "soc"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (
            0.8443919,
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array([0.996566, 0.976255, 0.830797, 0.725181, 0.525736]),
        ),
        (
            0.9941839,
            0.00075,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
            np.array(
                [
                    0.992443,
                    0.98205,
                    0.964735,
                    0.934943,
                    0.853887,
                    0.54003,
                    0.296843,
                    0.195002,
                    0.125025,
                ]
            ),
        ),
        (
            0.8859801,
            0.000175,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
            np.array([0.995197, 0.958646, 0.845837, 0.654458, 0.346546]),
        ),
        (
            0.9609059,
            0.002,
            np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
            np.array(
                [0.994179, 0.967568, 0.930123, 0.834509, 0.734328, 0.569661]
            ),
        ),
        (
            0.9589378,
            3.5e-5,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
            np.array(
                [0.948959, 0.836089, 0.759624, 0.329323, 0.020909, 0.00961]
            ),
        ),
        (
            0.5436137,
            2.5e-6,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
            np.array([0.9617, 0.938762, 0.9069, 0.863516, 0.696022, 0.421418]),
        ),
    ],
)
def test_score(ref, d, C_rates, soc):
    """Test the r2 score of the model."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-14, -6, 1)
    greg.k0s = 10.0 ** np.arange(-13, -5, 1)

    greg = greg.fit(C_rates, soc)

    r2 = greg.score(C_rates, soc)

    np.testing.assert_almost_equal(r2, ref)


@pytest.mark.parametrize(
    ("path", "d", "C_rates", "soc"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (
            "LMNO",
            np.sqrt(0.25 * 8.04e-6 / np.pi),
            np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
            np.array([0.996566, 0.976255, 0.830797, 0.725181, 0.525736]),
        ),
        (
            "NATURAL_GRAPHITE",
            0.00075,
            np.array(
                [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
            ).reshape(-1, 1),
            np.array(
                [
                    0.992443,
                    0.98205,
                    0.964735,
                    0.934943,
                    0.853887,
                    0.54003,
                    0.296843,
                    0.195002,
                    0.125025,
                ]
            ),
        ),
        (
            "LTO",
            0.000175,
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
            np.array([0.995197, 0.958646, 0.845837, 0.654458, 0.346546]),
        ),
        (
            "LCO",
            0.002,
            np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
            np.array(
                [0.994179, 0.967568, 0.930123, 0.834509, 0.734328, 0.569661]
            ),
        ),
        (
            "LFP",
            3.5e-5,
            np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
            np.array(
                [0.948959, 0.836089, 0.759624, 0.329323, 0.020909, 0.00961]
            ),
        ),
        (
            "LMO",
            2.5e-6,
            np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
            np.array([0.9617, 0.938762, 0.9069, 0.863516, 0.696022, 0.421418]),
        ),
    ],
)
def test_to_dataframe(path, d, C_rates, soc):
    """Test the dataframe."""
    df_ref = pd.read_csv(TEST_DATA_PATH / path / "df.csv", dtype=np.float32)

    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-14, -6, 1)
    greg.k0s = 10.0 ** np.arange(-13, -5, 1)

    greg = greg.fit(C_rates, soc)

    df = greg.to_dataframe(C_rates, y=soc)

    pd.testing.assert_frame_equal(df, df_ref)
