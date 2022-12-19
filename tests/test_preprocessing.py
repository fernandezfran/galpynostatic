#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

import galpynostatic.preprocessing

import numpy as np

import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# =============================================================================
# TESTS
# =============================================================================


def test_get_discharge_capacities():
    """Test the get of discharge capacities."""
    # reference xmaxs
    ref = np.array([0.37869504, 0.3709768, 0.3157027, 0.2755689, 0.19977959])

    # read experimental data
    dfs = [
        pd.read_csv(TEST_DATA_PATH / "LMNO" / f"{i}nA.csv")
        for i in (1, 2, 3, 5, 10)
    ]

    xmaxs = galpynostatic.preprocessing.get_discharge_capacities(dfs, 4.739)

    np.testing.assert_array_almost_equal(xmaxs, ref, 6)
