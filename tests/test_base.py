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

import galpynostatic.base

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("logell", "logxi", "grid", "ref"),
    [
        (-4.0, 2.0, False, 0.99706),
        (
            [-1, 0, 1],
            [-1, 0, 1],
            True,
            np.array(
                [
                    [8.6546e-01, 9.5395e-01, 9.6280e-01],
                    [3.6292e-01, 6.3441e-01, 6.6213e-01],
                    [1.1900e-04, 6.2640e-02, 7.5440e-02],
                ]
            ),
        ),
    ],
)
def test_map_spline(logell, logxi, grid, ref, spherical):
    """Test the MapSpline base class."""
    ms = galpynostatic.base.MapSpline(spherical)

    res = ms.soc(logell, logxi, grid=grid)

    np.testing.assert_array_almost_equal(res, ref, 6)


@pytest.mark.parametrize(("param"), [("ell"), ("Xi")])
def test_map_spline_masks(param, spherical):
    """Test the MapSpline masks."""
    ms = galpynostatic.base.MapSpline(spherical)

    res = {"ell": ms._mask_logell, "Xi": ms._mask_logxi}.get(param)(
        [-10, 0, 10]
    )

    np.testing.assert_equal(res, [False, True, False])
