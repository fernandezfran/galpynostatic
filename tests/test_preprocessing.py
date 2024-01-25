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

import galpynostatic.preprocessing

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("experiment"),
    [("nishikawa"), ("he"), ("wang"), ("lei"), ("bak")],
)
def test_get_discharge_capacities(experiment, request, data_path):
    """Test the get of discharge capacities."""
    experiment = request.getfixturevalue(experiment)

    dfs = [
        pd.read_csv(data_path / experiment["dir_name"] / f, header=None)
        for f in experiment["file_names"]
    ]

    gdc = galpynostatic.preprocessing.GetDischargeCapacities(
        experiment["eq_pot"]
    )
    dc = gdc.fit_transform(dfs)

    np.testing.assert_array_almost_equal(dc, experiment["ref"]["dc"], 5)


def test_get_discharge_capacities_internal_raise():
    """Test the get of discharge capacities IndexError raise.

    In this case the GetDischargeCapacities is not obtaining the zero value,
    it has been set to zero because the profile is below eq_pot from the
    beggining.
    """
    dfs = [
        pd.DataFrame(
            {
                0: np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                1: np.array([0.8, 0.75, 0.62, 0.43, 0.26]),
            }
        )
    ]

    gdc = galpynostatic.preprocessing.GetDischargeCapacities(1.0)
    dc = gdc.fit_transform(dfs)

    np.testing.assert_array_almost_equal(dc, [[0]])
