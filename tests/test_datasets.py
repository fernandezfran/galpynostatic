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

import galpynostatic.datasets

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_load_planar():
    """Test the planar dataset."""
    pla = galpynostatic.datasets.load_dataset(geometry="planar")

    assert isinstance(pla, pd.DataFrame)

    assert pla.l.min() == -4
    np.testing.assert_almost_equal(pla.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(pla.l.mean(), -1.125, 6)

    np.testing.assert_almost_equal(pla.xi.min(), -3.2, 6)
    np.testing.assert_almost_equal(pla.xi.max(), 2, 6)
    np.testing.assert_almost_equal(pla.xi.mean(), -0.6, 6)

    np.testing.assert_almost_equal(pla.xmax.min(), 0.0, 6)
    np.testing.assert_almost_equal(pla.xmax.max(), 0.997055, 6)
    np.testing.assert_almost_equal(pla.xmax.mean(), 0.515846, 6)


def test_load_cylindrical():
    """Test the cylindrical dataset."""
    cyl = galpynostatic.datasets.load_dataset(geometry="cylindrical")

    assert isinstance(cyl, pd.DataFrame)

    assert cyl.l.min() == -4
    np.testing.assert_almost_equal(cyl.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(cyl.l.mean(), -1.125, 6)

    np.testing.assert_almost_equal(cyl.xi.min(), -3.4, 6)
    np.testing.assert_almost_equal(cyl.xi.max(), 2, 6)
    np.testing.assert_almost_equal(cyl.xi.mean(), -0.7, 6)

    np.testing.assert_almost_equal(cyl.xmax.min(), 0.0, 6)
    np.testing.assert_almost_equal(cyl.xmax.max(), 0.997055, 6)
    np.testing.assert_almost_equal(cyl.xmax.mean(), 0.516647, 6)


def test_load_spherical():
    """Test the spherical dataset."""
    sph = galpynostatic.datasets.load_dataset()

    assert isinstance(sph, pd.DataFrame)

    assert sph.l.min() == -4
    np.testing.assert_almost_equal(sph.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(sph.l.mean(), -1.125, 6)

    np.testing.assert_almost_equal(sph.xi.min(), -3.5, 6)
    np.testing.assert_almost_equal(sph.xi.max(), 2, 6)
    np.testing.assert_almost_equal(sph.xi.mean(), -0.75, 6)

    np.testing.assert_almost_equal(sph.xmax.min(), 0.0, 6)
    np.testing.assert_almost_equal(sph.xmax.max(), 0.99706, 6)
    np.testing.assert_almost_equal(sph.xmax.mean(), 0.518575, 6)


def test_raise():
    """Test the raise of the ValueError."""
    with pytest.raises(ValueError):
        galpynostatic.datasets.load_dataset(geometry="plane")
