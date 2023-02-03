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

import galpynostatic.model

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_dcoeffs(spherical):
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(spherical, 1.0, 3)

    np.testing.assert_array_almost_equal(
        greg.dcoeffs, np.logspace(-15, -6, num=100)
    )


def test_k0s(spherical):
    """A property test."""
    greg = galpynostatic.model.GalvanostaticRegressor(spherical, 1.0, 3)

    np.testing.assert_array_almost_equal(
        greg.k0s, np.logspace(-14, -5, num=100)
    )


@pytest.mark.parametrize(
    ("experiment"),
    [
        ("nishikawa_experiment"),
        ("mancini_experiment"),
        ("he_experiment"),
        ("wang_experiment"),
        ("lei_experiment"),
        ("bak_experiment"),
    ],
)
class TestModel:
    """Test the galvanostatic regressor model with parametrization shared."""

    def test_fit(self, experiment, request, spherical):
        """Test the fitting of the model: dcoeff, k0 and mse."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            spherical, experiment["d"], 3
        )

        greg.dcoeffs = 10.0 ** np.arange(-14, -6, 1)
        greg.k0s = 10.0 ** np.arange(-13, -5, 1)

        greg = greg.fit(experiment["C_rates"], experiment["soc"])

        np.testing.assert_almost_equal(
            greg.dcoeff_, experiment["ref"]["dcoeff"], 12
        )
        np.testing.assert_almost_equal(greg.k0_, experiment["ref"]["k0"], 10)
        np.testing.assert_almost_equal(greg.mse_, experiment["ref"]["mse"], 6)

    def test_predict(self, experiment, request, spherical):
        """Test the predict of the soc values."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            spherical, experiment["d"], 3
        )

        greg.dcoeff_ = experiment["dcoeff"]
        greg.k0_ = experiment["k0"]

        soc = greg.predict(experiment["C_rates"])

        np.testing.assert_array_almost_equal(soc, experiment["ref"]["soc"], 6)

    def test_score(self, experiment, request, spherical):
        """Test the r2 score of the model."""
        experiment = request.getfixturevalue(experiment)

        greg = galpynostatic.model.GalvanostaticRegressor(
            spherical, experiment["d"], 3
        )

        greg.dcoeff_ = experiment["dcoeff"]
        greg.k0_ = experiment["k0"]

        r2 = greg.score(experiment["C_rates"], experiment["soc"])

        np.testing.assert_almost_equal(r2, experiment["ref"]["r2"])

    def test_to_dataframe(self, experiment, request, spherical, data_path):
        """Test the dataframe."""
        experiment = request.getfixturevalue(experiment)

        df_ref = pd.read_csv(
            data_path / experiment["dir_name"] / "df.csv", dtype=np.float32
        )

        greg = galpynostatic.model.GalvanostaticRegressor(
            spherical, experiment["d"], 3
        )

        greg.dcoeff_ = experiment["dcoeff"]
        greg.k0_ = experiment["k0"]

        df = greg.to_dataframe(experiment["C_rates"], y=experiment["soc"])

        pd.testing.assert_frame_equal(df, df_ref)
