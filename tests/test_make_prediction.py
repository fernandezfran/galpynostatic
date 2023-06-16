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

import galpynostatic.make_prediction
import galpynostatic.model

import numpy as np

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
def test_optimal_particle_size(experiment, request, spherical):
    """Test the prediction of the optimal particle size."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(
        spherical, experiment["d"], 3
    )

    # fit results
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
    greg.dcoeff_err_ = experiment["ref"]["dcoeff_err"]

    size, size_err = galpynostatic.make_prediction.optimal_particle_size(greg)

    np.testing.assert_array_almost_equal(
        size, experiment["ref"]["particle_size"], 6
    )
    np.testing.assert_array_almost_equal(
        size_err, experiment["ref"]["particle_size_err"], 6
    )


def test_raise():
    """Test the raise of the ValueError."""
    greg = galpynostatic.model.GalvanostaticRegressor("spherical", 0.0015, 3)

    greg.dcoeff_, greg.k0_ = 1.93e-10, 3.14e-7

    with pytest.raises(ValueError):
        galpynostatic.make_prediction.optimal_particle_size(
            greg, minutes=1, loaded=1
        )
