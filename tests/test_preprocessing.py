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

import galpynostatic.preprocessing

import numpy as np

import pandas as pd

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("ref", "dir_name", "file_names", "eq_pot"),
    [  # nishikawa, he, wang, lei, bak data
        (
            np.array(
                [0.37869504, 0.3709768, 0.3157027, 0.2755689, 0.19977959]
            ),
            "LMNO",
            ("1nA.csv", "2nA.csv", "3nA.csv", "5nA.csv", "10nA.csv"),
            4.739,
        ),
        (
            np.array([159.23154, 153.38335, 135.33395, 104.71328, 55.44732]),
            "LTO",
            ("0.1C.csv", "0.5C.csv", "1C.csv", "2C.csv", "5C.csv"),
            1.57,
        ),
        (
            np.array(
                [99.417946, 96.75683, 93.01233, 83.45085, 73.432816, 56.96607]
            ),
            "LCO",
            ("0.5C.csv", "1C.csv", "2C.csv", "5C.csv", "10C.csv", "20C.csv"),
            3.9,
        ),
        (
            np.array(
                [
                    160.27911,
                    141.21538,
                    128.30042,
                    55.622726,
                    3.5315654,
                    1.6230893,
                ]
            ),
            "LFP",
            ("0.2C.csv", "0.5C.csv", "1C.csv", "2C.csv", "5C.csv", "10C.csv"),
            3.45,
        ),
        (
            np.array(
                [
                    127.32904,
                    124.29212,
                    120.07357,
                    114.329475,
                    92.15335,
                    55.795765,
                ]
            ),
            "LMO",
            ("1C.csv", "5C.csv", "10C.csv", "20C.csv", "50C.csv", "100C.csv"),
            4.0,
        ),
    ],
)
def test_get_discharge_capacities(ref, dir_name, file_names, eq_pot):
    """Test the get of discharge capacities."""
    dfs = [
        pd.read_csv(TEST_DATA_PATH / dir_name / f, header=None)
        for f in file_names
    ]

    xmaxs = galpynostatic.preprocessing.get_discharge_capacities(dfs, eq_pot)

    np.testing.assert_array_almost_equal(xmaxs, ref, 5)


def test_get_discharge_capacities_raise():
    """Test the get of discharge capacities ValueError raise."""
    dfs = [
        pd.DataFrame(
            {
                0: np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                1: np.array([0.8, 0.75, 0.62, 0.43, 0.26]),
            }
        )
    ]
    with pytest.raises(ValueError):
        galpynostatic.preprocessing.get_discharge_capacities(dfs, 1.0)
