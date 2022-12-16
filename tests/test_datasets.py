#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    """Text the planar dataset."""
    with pytest.raises(NotImplementedError):
        galpynostatic.datasets.load_planar()


def test_load_cylindrical():
    """Text the cylindrical dataset."""
    with pytest.raises(NotImplementedError):
        galpynostatic.datasets.load_cylindrical()


def test_load_spherical():
    """Text the spherical dataset."""
    sph = galpynostatic.datasets.load_spherical()

    assert isinstance(sph, pd.DataFrame)

    assert sph.l.min() == -4
    np.testing.assert_almost_equal(sph.l.max(), 1.75, 6)
    np.testing.assert_almost_equal(sph.l.mean(), -1.488384, 6)

    np.testing.assert_almost_equal(sph.chi.min(), -3.5, 6)
    np.testing.assert_almost_equal(sph.chi.max(), 2, 6)
    np.testing.assert_almost_equal(sph.chi.mean(), -0.103535, 6)

    np.testing.assert_almost_equal(sph.xmax.min(), 0.000119, 6)
    np.testing.assert_almost_equal(sph.xmax.max(), 0.99706, 6)
    np.testing.assert_almost_equal(sph.xmax.mean(), 0.704005, 6)
