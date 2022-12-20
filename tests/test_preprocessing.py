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
    [
        (  # nishikawa data
            np.array(
                [0.37869504, 0.3709768, 0.3157027, 0.2755689, 0.19977959]
            ),
            "LMNO",
            ("1nA.csv", "2nA.csv", "3nA.csv", "5nA.csv", "10nA.csv"),
            4.739,
        )
    ],
)
def test_get_discharge_capacities(ref, dir_name, file_names, eq_pot):
    """Test the get of discharge capacities."""
    dfs = [
        pd.read_csv(TEST_DATA_PATH / dir_name / fname) for fname in file_names
    ]

    xmaxs = galpynostatic.preprocessing.get_discharge_capacities(dfs, eq_pot)

    np.testing.assert_array_almost_equal(xmaxs, ref, 6)
