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
    # reference values
    ref_dcoeff = 1e-9
    ref_k0 = 1e-6
    ref_mse = 0.009160

    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # regressor configuration to make it faster
    greg.dcoeffs = 10.0 ** np.arange(-10, -6, 1)
    greg.k0s = 10.0 ** np.arange(-9, -5, 1)

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0])
    xmaxs = np.array([0.973333, 0.946667, 0.84, 0.68, 0.52])

    # fit
    greg = greg.fit(crates, xmaxs)

    print(greg.predict(crates))
    # tests
    np.testing.assert_almost_equal(greg.dcoeff_, ref_dcoeff, 10)
    np.testing.assert_almost_equal(greg.k0_, ref_k0, 7)
    np.testing.assert_almost_equal(greg.mse_, ref_mse, 6)


def test_predict():
    """Test the predict of the xmaxs values."""
    # reference xmaxs predictions
    ref = np.array([0.92744, 0.86974, 0.77282, 0.76325, 0.35772])

    # regressor obj
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    # nishikawa fitted res
    greg.dcoeff_ = 1.0e-09
    greg.k0_ = 1.0e-6

    # nishikawa data
    crates = np.array([2.5, 5, 7.5, 12.5, 25.0])

    # predict
    xmaxs = greg.predict(crates)

    # tests
    np.testing.assert_array_almost_equal(xmaxs, ref, 6)


def test_dcoeffs():
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    np.testing.assert_array_almost_equal(
        greg.dcoeffs, 10.0 ** np.arange(-15, -6, 0.1)
    )


def test_k0s():
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(
        DATASET, np.sqrt(0.25 * 8.04e-6 / np.pi), 3
    )

    np.testing.assert_array_almost_equal(
        greg.k0s, 10.0 ** np.arange(-14, -5, 0.1)
    )
