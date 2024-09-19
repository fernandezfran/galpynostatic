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

import galpynostatic.simulation
import galpynostatic.utils

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("c_rate", "ref"),
    [
        (4, -2.431364),
        (0.1, -4.033424),
        ([10.0, 1.0, 0.1], [-2.033424, -3.033424, -4.033424]),
    ],
)
def test_logell(c_rate, ref):
    """Test the log ell values."""
    res = galpynostatic.utils.logell(c_rate, 1e-4, 3, 1e-9)

    np.testing.assert_array_almost_equal(res, ref, 6)


@pytest.mark.parametrize(
    ("c_rate", "ref"),
    [
        (4, -1.022879),
        (0.1, -0.221849),
        ([10.0, 1.0, 0.1], [-1.221849, -0.721849, -0.221849]),
    ],
)
def test_logxi(c_rate, ref):
    """Test the log Xi values."""
    res = galpynostatic.utils.logxi(c_rate, 1e-9, 1e-7, 1)

    np.testing.assert_array_almost_equal(res, ref, 6)


def test_logcrate():
    xi_log = [-2, -1, 1, 2, 3]
    res = galpynostatic.simulation.logcrate(xi_log, 6.0852e-14, 1.0991e-8, 1)
    ref = np.asarray([4.854102, 2.854102, -1.145898, -3.145898, -5.145898])

    np.testing.assert_array_almost_equal(res, ref, 6)


def test_logd():
    logs = [-2, -1, 1, 2, 3]
    res = galpynostatic.simulation.logd(
        logs, logs, 6.085284e-14, 1.099165e-8, 1
    )
    ref = np.asarray([-8.256782, -6.756782, -3.756782, -2.256782, -0.756782])

    np.testing.assert_array_almost_equal(res, ref, 6)
