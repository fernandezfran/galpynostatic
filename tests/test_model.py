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
import galpynostatic.model

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("experiment"),
    [
        ("nishikawa"),
        ("mancini"),
        ("he"),
        ("wang"),
        ("lei"),
        ("bak"),
        ("dokko"),
    ],
)
class TestModel:
    """Test the galvanostatic regressor model with parametrization shared."""

    def test_fit(self, experiment, request, spherical):
        """Test the fitting of the model: dcoeff, k0 and mse."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            dataset=spherical,
            d=experiment["d"],
            z=3,
            dcoeff_lle=-14,
            dcoeff_ule=-7,
            k0_lle=-13,
            k0_ule=-6,
            dcoeff_num=8,
            k0_num=8,
        )

        greg = greg.fit(experiment["C_rates"], experiment["soc"])

        np.testing.assert_almost_equal(
            greg.dcoeff_, experiment["ref"]["dcoeff"], 13
        )
        np.testing.assert_almost_equal(
            greg.dcoeff_err_, experiment["ref"]["dcoeff_err"], 13
        )
        np.testing.assert_almost_equal(greg.k0_, experiment["ref"]["k0"], 11)
        np.testing.assert_almost_equal(
            greg.k0_err_, experiment["ref"]["k0_err"], 11
        )
        np.testing.assert_almost_equal(greg.mse_, experiment["ref"]["mse"], 6)

    def test_fit_with_str_dataset(self, experiment, request):
        """Test the fitting of the model: dcoeff, k0 and mse."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            dataset="spherical",
            d=experiment["d"],
            z=3,
            dcoeff_lle=-14,
            dcoeff_ule=-7,
            k0_lle=-13,
            k0_ule=-6,
            dcoeff_num=8,
            k0_num=8,
        )

        greg = greg.fit(experiment["C_rates"], experiment["soc"])

        np.testing.assert_almost_equal(
            greg.dcoeff_, experiment["ref"]["dcoeff"], 12
        )
        np.testing.assert_almost_equal(greg.k0_, experiment["ref"]["k0"], 10)
        np.testing.assert_almost_equal(greg.mse_, experiment["ref"]["mse"], 6)

    def test_predict(self, experiment, request, spherical):
        """Test the predict of the maximum SOC values."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            d=experiment["d"], z=3
        )
        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
        greg._map = galpynostatic.base.MapSpline(spherical)

        soc = greg.predict(experiment["C_rates"])

        np.testing.assert_array_almost_equal(soc, experiment["ref"]["soc"], 6)

    def test_score(self, experiment, request, spherical):
        """Test the r2 score of the model."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            d=experiment["d"], z=3
        )
        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
        greg._map = galpynostatic.base.MapSpline(spherical)

        r2 = greg.score(experiment["C_rates"], experiment["soc"])

        np.testing.assert_almost_equal(r2, experiment["ref"]["r2"])

    def test_to_dataframe(self, experiment, request, spherical, data_path):
        """Test the dataframe."""
        experiment = request.getfixturevalue(experiment)

        df_ref = pd.read_csv(data_path / experiment["dir_name"] / "df.csv")

        greg = galpynostatic.model.GalvanostaticRegressor(
            d=experiment["d"], z=3
        )
        greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
        greg._map = galpynostatic.base.MapSpline(spherical)

        df = greg.to_dataframe(experiment["C_rates"], y=experiment["soc"])

        pd.testing.assert_frame_equal(df, df_ref)


def test_raise():
    """Test the raise of the ValueError."""
    greg = galpynostatic.model.GalvanostaticRegressor("spherica", 1.0, 3)
    with pytest.raises(ValueError):
        greg.fit([[4.0]], [0.8])
