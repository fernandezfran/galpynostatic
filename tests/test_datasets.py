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


# =============================================================================
# planar geometry
# =============================================================================


def test_planar_dataset_type():
    """Test the type of the planar dataset."""
    assert isinstance(
        galpynostatic.datasets.load_dataset(geometry="planar"), pd.DataFrame
    )


def test_planar_dataset_logell_min():
    """Test minimum value of 'logell' in the planar dataset."""
    assert galpynostatic.datasets.load_dataset(geometry="planar").l.min() == -4


def test_planar_dataset_logell_max():
    """Test maximum value of 'logell' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").l.max(), 1.75, 6
    )


def test_planar_dataset_logell_mean():
    """Test mean value of 'logell' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").l.mean(),
        -1.125,
        6,
    )


def test_planar_dataset_logxi_min():
    """Test minimum value of 'logxi' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xi.min(),
        -3.2,
        6,
    )


def test_planar_dataset_logxi_max():
    """Test maximum value of 'logxi' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xi.max(), 2, 6
    )


def test_planar_dataset_logxi_mean():
    """Test mean value of 'logxi' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xi.mean(),
        -0.6,
        6,
    )


def test_planar_dataset_xmax_min():
    """Test minimum value of 'xmax' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xmax.min(),
        0.0,
        6,
    )


def test_planar_dataset_xmax_max():
    """Test maximum value of 'xmax' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xmax.max(),
        0.997055,
        6,
    )


def test_planar_dataset_xmax_mean():
    """Test mean value of 'xmax' in the planar dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="planar").xmax.mean(),
        0.515846,
        6,
    )


# =============================================================================
# cylindrical geometry
# =============================================================================


def test_cylindrical_dataset_type():
    """Test the type of the cylindrical dataset."""
    assert isinstance(
        galpynostatic.datasets.load_dataset(geometry="cylindrical"),
        pd.DataFrame,
    )


def test_cylindrical_dataset_logell_min():
    """Test minimum value of 'logell' in the cylindrical dataset."""
    assert (
        galpynostatic.datasets.load_dataset(geometry="cylindrical").l.min()
        == -4
    )


def test_cylindrical_dataset_logell_max():
    """Test maximum value of 'logell' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").l.max(),
        1.75,
        6,
    )


def test_cylindrical_dataset_logell_mean():
    """Test mean value of 'logell' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").l.mean(),
        -1.125,
        6,
    )


def test_cylindrical_dataset_logxi_min():
    """Test minimum value of 'logxi' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").xi.min(),
        -3.4,
        6,
    )


def test_cylindrical_dataset_logxi_max():
    """Test maximum value of 'logxi' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").xi.max(),
        2,
        6,
    )


def test_cylindrical_dataset_logxi_mean():
    """Test mean value of 'logxi' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").xi.mean(),
        -0.7,
        6,
    )


def test_cylindrical_dataset_xmax_min():
    """Test minimum value of 'xmax' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").xmax.min(),
        0.0,
        6,
    )


def test_cylindrical_dataset_xmax_max():
    """Test maximum value of 'xmax' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(geometry="cylindrical").xmax.max(),
        0.997055,
        6,
    )


def test_cylindrical_dataset_xmax_mean():
    """Test mean value of 'xmax' in the cylindrical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset(
            geometry="cylindrical"
        ).xmax.mean(),
        0.516647,
        6,
    )


# =============================================================================
# spherical geometry
# =============================================================================


def test_spherical_dataset_type():
    """Test the type of the spherical dataset."""
    assert isinstance(galpynostatic.datasets.load_dataset(), pd.DataFrame)


def test_spherical_dataset_logell_min():
    """Test minimum value of 'logell' in the spherical dataset."""
    assert galpynostatic.datasets.load_dataset().l.min() == -4


def test_spherical_dataset_logell_max():
    """Test maximum value of 'logell' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().l.max(), 1.75, 6
    )


def test_spherical_dataset_logell_mean():
    """Test mean value of 'logell' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().l.mean(), -1.125, 6
    )


def test_spherical_dataset_logxi_min():
    """Test minimum value of 'logxi' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xi.min(), -3.5, 6
    )


def test_spherical_dataset_logxi_max():
    """Test maximum value of 'logxi' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xi.max(), 2, 6
    )


def test_spherical_dataset_logxi_mean():
    """Test mean value of 'logxi' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xi.mean(), -0.75, 6
    )


def test_spherical_dataset_xmax_min():
    """Test minimum value of 'xmax' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xmax.min(), 0.0, 6
    )


def test_spherical_dataset_xmax_max():
    """Test maximum value of 'xmax' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xmax.max(), 0.99706, 6
    )


def test_spherical_dataset_xmax_mean():
    """Test mean value of 'xmax' in the spherical dataset."""
    np.testing.assert_almost_equal(
        galpynostatic.datasets.load_dataset().xmax.mean(), 0.518575, 6
    )


def test_value_error():
    """Test the raise of the ValueError."""
    with pytest.raises(ValueError):
        galpynostatic.datasets.load_dataset(geometry="plane")
