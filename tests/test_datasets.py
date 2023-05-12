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

import numpy as np

import pandas as pd


# =============================================================================
# TESTS
# =============================================================================


def test_load_planar():
    """Test the planar dataset."""
    pla = galpynostatic.datasets.load_planar()

    assert isinstance(pla, pd.DataFrame)

    assert pla.l.min() == -4
    np.testing.assert_almost_equal(pla.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(pla.l.mean(), -1.504081, 6)

    np.testing.assert_almost_equal(pla.xi.min(), -3.25, 6)
    np.testing.assert_almost_equal(pla.xi.max(), 2, 6)
    np.testing.assert_almost_equal(pla.xi.mean(), 0.017519, 6)

    np.testing.assert_almost_equal(pla.xmax.min(), 6e-5, 6)
    np.testing.assert_almost_equal(pla.xmax.max(), 0.997055, 6)
    np.testing.assert_almost_equal(pla.xmax.mean(), 0.702871, 6)


def test_load_cylindrical():
    """Test the cylindrical dataset."""
    cyl = galpynostatic.datasets.load_cylindrical()

    assert isinstance(cyl, pd.DataFrame)

    assert cyl.l.min() == -4
    np.testing.assert_almost_equal(cyl.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(cyl.l.mean(), -1.4597826, 6)

    np.testing.assert_almost_equal(cyl.xi.min(), -3.4, 6)
    np.testing.assert_almost_equal(cyl.xi.max(), 2, 6)
    np.testing.assert_almost_equal(cyl.xi.mean(), -0.04983696, 6)

    np.testing.assert_almost_equal(cyl.xmax.min(), 0.000239, 6)
    np.testing.assert_almost_equal(cyl.xmax.max(), 0.997055, 6)
    np.testing.assert_almost_equal(cyl.xmax.mean(), 0.695191, 6)


def test_load_spherical():
    """Test the spherical dataset."""
    sph = galpynostatic.datasets.load_spherical()

    assert isinstance(sph, pd.DataFrame)

    assert sph.l.min() == -4
    np.testing.assert_almost_equal(sph.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(sph.l.mean(), -1.488384, 6)

    np.testing.assert_almost_equal(sph.xi.min(), -3.5, 6)
    np.testing.assert_almost_equal(sph.xi.max(), 2, 6)
    np.testing.assert_almost_equal(sph.xi.mean(), -0.103535, 6)

    np.testing.assert_almost_equal(sph.xmax.min(), 0.000119, 6)
    np.testing.assert_almost_equal(sph.xmax.max(), 0.99706, 6)
    np.testing.assert_almost_equal(sph.xmax.mean(), 0.704005, 6)
