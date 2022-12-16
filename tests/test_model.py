#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================

import galpynostatic.datasets
import galpynostatic.model

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

DATASET = galpynostatic.datasets.load_spherical()

# =============================================================================
# TESTS
# =============================================================================


def test_fit():
    """Test the fitting of the model: dcoeff, k0 and mse."""
    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # regressor configuration to make it faster
    greg.dcoefs = 10.0 ** np.arange(-10, -6, 1)
    greg.k0s = 10.0 ** np.arange(-9, -5, 1)

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0])
    xmaxs = np.array([0.973333, 0.946667, 0.84, 0.68, 0.52])

    # fit
    greg = greg.fit(crates, xmaxs)

    # tests
    np.testing.assert_almost_equal(greg.dcoeff_, 1.584893e-9, 6 + 9)
    np.testing.assert_almost_equal(greg.k0_, 1e-6, 6 + 1)
    np.testing.assert_almost_equal(greg.mse_, 0.001783, 6)


def test_predict():
    """Test the predict of the xmaxs values."""
    # reference xmaxs predictions
    ref = np.array([0.95395, 0.91707, 0.85686, 0.75121, 0.56805])

    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = 1.5848931924610332e-09
    greg.k0_ = 1.0e-6

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0])

    # predict
    xmaxs = greg.predict(crates)

    # tests
    np.testing.assert_array_almost_equal(xmaxs, ref, 6)
