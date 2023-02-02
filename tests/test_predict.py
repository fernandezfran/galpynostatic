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

import galpynostatic.datasets
import galpynostatic.model
import galpynostatic.predict

import numpy as np

import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET = galpynostatic.datasets.load_spherical()

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("ref", "d", "dcoeff", "k0"),
    [  # nishikawa, mancini, he, wang, lei, bak data
        (6.501643, np.sqrt(0.25 * 8.04e-6 / np.pi), 1.0e-09, 1.0e-6),
        (2.213407, 0.00075, 1e-10, 1e-6),
        (0.280568, 0.000175, 1.0e-11, 1.0e-8),
        (16.25661, 0.002, 1e-8, 1e-6),
        (0.065173, 3.5e-5, 1e-13, 1e-8),
        (0.022026, 2.5e-6, 1e-14, 1e-8),
    ],
)
def test_t_minutes_lenght(ref, d, dcoeff, k0):
    """Test the t minutes lenght."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, d, 3)

    # fit results
    greg.dcoeff_ = dcoeff
    greg.k0_ = k0

    greg._surface()
    lenght = galpynostatic.predict.t_minutes_length(greg)

    np.testing.assert_array_almost_equal(lenght, ref, 6)


def test_t_minutes_raise():
    """Test the t minutes lenght ValueError raise."""
    greg = galpynostatic.model.GalvanostaticRegressor(DATASET, 0.005, 3)

    # fictional fit results
    greg.dcoeff_ = 3e-5
    greg.k0_ = 1e-7

    greg._surface()
    with pytest.raises(ValueError):
        galpynostatic.predict.t_minutes_length(greg)
