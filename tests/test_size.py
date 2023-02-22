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
import galpynostatic.size

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


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
def test_predict_length(experiment, request, spherical):
    """Test the t minutes length."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(
        spherical, experiment["d"], 3
    )

    # fit results
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]

    length = galpynostatic.size.predict_length(greg)

    np.testing.assert_array_almost_equal(
        length, experiment["ref"]["length"], 6
    )
