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

    length = galpynostatic.make_prediction.optimal_particle_size(greg)

    np.testing.assert_array_almost_equal(
        length, experiment["ref"]["particle_size"], 6
    )
