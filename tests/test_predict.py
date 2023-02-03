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
import galpynostatic.predict

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
def test_t_minutes_lenght(experiment, request, spherical):
    """Test the t minutes lenght."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(
        spherical, experiment["d"], 3
    )

    # fit results
    greg.dcoeff_ = experiment["dcoeff"]
    greg.k0_ = experiment["k0"]

    lenght = galpynostatic.predict.t_minutes_length(greg)

    np.testing.assert_array_almost_equal(
        lenght, experiment["ref"]["length"], 6
    )


def test_t_minutes_raise(spherical):
    """Test the t minutes lenght ValueError raise."""
    greg = galpynostatic.model.GalvanostaticRegressor(spherical, 0.005, 3)

    # fictional fit results
    greg.dcoeff_ = 3e-5
    greg.k0_ = 1e-7

    with pytest.raises(ValueError):
        galpynostatic.predict.t_minutes_length(greg)
