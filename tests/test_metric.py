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

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(("fit"), [(True), (False)])
def test_bmx_fc(fit, data_path):
    """Test the Benchmarking Materials for eXtreme fast-charging metric."""
    df = pd.read_csv(data_path / "metric_dataset.csv")

    for d, dcoeff, ref in zip(df["d"], df["dcoeff"], df["ref_soc"]):
        if fit:
            greg = {"d": 1e-4 * d, "dcoeff_": dcoeff, "k0_": 1e-7}
        else:
            greg = galpynostatic.model.GalvanostaticRegressor(d=1e-4 * d)
            greg._validate_geometry()
            greg._map = galpynostatic.datasets.map.MapSpline(greg.dataset)
            greg.dcoeff_, greg.k0_ = dcoeff, 1e-7
            greg.dcoeff_err_ = None

        value = galpynostatic.metric.bmx_fc(greg)

        np.testing.assert_almost_equal(value, ref, 6)


def test_fom(data_path):
    """Test the Figure of Merit for fast-charging of Xia et al."""
    df = pd.read_csv(data_path / "metric_dataset.csv")

    for d, dcoeff, ref in zip(df["d"], df["dcoeff"], df["ref_tau"]):
        value = galpynostatic.metric.fom(1e-4 * d, dcoeff)

        np.testing.assert_almost_equal(value, ref, 6)
