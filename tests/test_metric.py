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

import galpynostatic.base
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
        (10.0, 7.8e-11, [0.1246232, 12820.512821]),
        (8.0, 3.324154e-10, [0.6676308, 1925.301896]),
        (3.5, 1.732051e-11, [0.2758745, 7072.540012]),
        (7.5, 1.0e-09, [0.8413092, 562.5]),
        (2.775, 3.872983e-10, [0.9406144, 198.8293]),
        (0.5, 1.0e-09, [0.9910296, 2.5]),
        (3.0, 8.246211e-11, [0.8278846, 1091.410346]),
        (0.1, 1.34e-11, [0.9950223, 7.462687]),
        (0.2, 9.04e-11, [0.9942393, 4.424779]),
        (3.5, 1.5e-12, [0.0239385, 81666.666667]),
        (0.3, 2.97e-12, [0.9562543, 303.030303]),
        (3.5, 1.57e-07, [0.9566931, 0.780255]),
        (10.0, 1.0e-08, [0.8696052, 100.0]),
        (0.1, 1.07e-12, [0.9844069, 93.457944]),
        (0.2, 1.486607e-14, [0.0768491, 26906.909493]),
        (0.35, 8.726168e-12, [0.9757423, 140.382353]),
        (0.3, 9.872386e-15, [0.0231658, 91163.372259]),
        (0.2, 3.640055e-14, [0.1851846, 10988.84494877138]),
        (0.3, 4.547417e-12, [0.9692263, 197.914552]),
        (0.45, 4.0e-12, [0.9294533, 506.25]),
        (0.935, 4.690416e-13, [0.1086396, 18638.538671]),
        (0.2, 1.03e-13, [0.5275187, 3883.495146]),
        (0.09, 1.04e-13, [0.8999858, 778.846154]),
        (0.935, 1.0e-11, [0.8784640, 874.225000]),
        (15.0, 1.430315e-10, [0.0898263, 15730.800558]),
        (10.0, 4.690416e-10, [0.6193627, 2132.007054]),
        (25.166667, 7.6e-11, [0.0129400, 83336.990512]),
        (25.166667, 4.4e-06, [0.7070715, 1.439457]),
        (50.5, 1.0e-07, [0.385020, 255.025]),
    ],
)
class TestMetric:
    """Test the different metrics for benchmarking FC materials."""

    def test_fom(self, d, dcoeff, ref):
        """Test the Figure of Merit for fast-charging of Xia et al."""
        value = galpynostatic.metric.fom(1e-4 * d, dcoeff)

        np.testing.assert_almost_equal(value, ref[1], 6)

    def test_bmxfc(self, d, dcoeff, ref):
        """Test the BMXFC metric without fitting the model."""
        greg = galpynostatic.model.GalvanostaticRegressor(d=1e-4 * d)
        greg._validate_geometry()
        greg._map = galpynostatic.base.MapSpline(greg.dataset)
        greg.dcoeff_, greg.k0_ = dcoeff, 1e-7
        greg.dcoeff_err_ = None

        value = galpynostatic.metric.bmxfc(greg)

        np.testing.assert_almost_equal(value, ref[0], 6)

    def test_bmxfc_fit(self, d, dcoeff, ref):
        """Test the BMXFC metric using the fit argument."""
        greg = {"d": 1e-4 * d, "dcoeff_": dcoeff, "k0_": 1e-7}

        value = galpynostatic.metric.bmxfc(greg)

        np.testing.assert_almost_equal(value, ref[0], 6)
