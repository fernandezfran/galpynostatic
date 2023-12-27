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

import galpynostatic.datasets.map
import galpynostatic.metric
import galpynostatic.model

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("d", "dcoeff", "ref"),
    [
        (10.0, 7.8e-11, [0.101967, 12820.512821]),
        (8.0, 3.324154e-10, [0.599743, 1925.301896]),
        (3.5, 1.732051e-11, [0.258640, 7072.540012]),
        (7.5, 1.0e-09, [0.778005, 562.5]),
        (2.775, 3.872983e-10, [0.917220, 198.8293]),
        (0.5, 1.0e-09, [0.986813, 2.5]),
        (3.0, 8.246211e-11, [0.802635, 1091.410346]),
        (0.1, 1.34e-11, [0.99418, 7.462687]),
        (0.2, 9.04e-11, [0.992554, 4.424779]),
        (3.5, 1.5e-12, [0.022498, 81666.666667]),
        (0.3, 2.97e-12, [0.953725, 303.030303]),
        (3.5, 1.57e-07, [0.927188, 0.780255]),
        (10.0, 1.0e-08, [0.785281, 100.0]),
        (0.1, 1.07e-12, [0.983569, 93.457944]),
        (0.2, 1.486607e-14, [0.076591, 26906.909493]),
        (0.35, 8.726168e-12, [0.972794, 140.382353]),
        (0.3, 9.872386e-15, [0.023048, 91163.372259]),
        (0.2, 3.640055e-14, [0.184536, 10988.84494877138]),
        (0.3, 4.547417e-12, [0.966699, 197.914552]),
        (0.45, 4.0e-12, [0.925663, 506.25]),
        (0.935, 4.690416e-13, [0.106935, 18638.538671]),
        (0.2, 1.03e-13, [0.525938, 3883.495146]),
        (0.09, 1.04e-13, [0.899222, 778.846154]),
        (0.935, 1.0e-11, [0.870576, 874.225000]),
        (15.0, 1.430315e-10, [0.064599, 15730.800558]),
        (10.0, 4.690416e-10, [0.534568, 2132.007054]),
        (25.166667, 7.6e-11, [0.006433, 83336.990512]),
        (25.166667, 4.4e-06, [0.493498, 1.439457]),
        (50.5, 1.0e-07, [0.0, 255.025]),
    ],
)
class TestMetric:
    """Test the different metrics for benchmarking FC materials."""

    def test_fom(self, d, dcoeff, ref):
        """Test the Figure of Merit for fast-charging of Xia et al."""
        value = galpynostatic.metric.fom(1e-4 * d, dcoeff)

        np.testing.assert_almost_equal(value, ref[1], 6)

    def test_umbem(self, d, dcoeff, ref):
        """Test the UMBEM metric without fitting the model."""
        greg = galpynostatic.model.GalvanostaticRegressor(d=1e-4 * d)
        greg._validate_geometry()
        greg._map = galpynostatic.datasets.map.MapSpline(greg.dataset)
        greg.dcoeff_, greg.k0_ = dcoeff, 1e-7
        greg.dcoeff_err_ = None

        value = galpynostatic.metric.umbem(greg)

        np.testing.assert_almost_equal(value, ref[0], 6)

    def test_umbem_fit(self, d, dcoeff, ref):
        """Test the UMBEM metric using the fit argument."""
        greg = {"d": 1e-4 * d, "dcoeff_": dcoeff, "k0_": 1e-7}

        value = galpynostatic.metric.umbem(greg)

        np.testing.assert_almost_equal(value, ref[0], 6)
