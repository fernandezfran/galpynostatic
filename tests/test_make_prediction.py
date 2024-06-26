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
def test_optimal_charging_rate(experiment, request, spherical):
    """Test the prediction of the optimal C-rate."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(d=experiment["d"], z=3)
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
    greg._map = galpynostatic.base.MapSpline(spherical)
    greg.dcoeff_err_, greg.k0_err_ = None, None

    c_rate = galpynostatic.make_prediction.optimal_charging_rate(greg)

    np.testing.assert_array_almost_equal(
        c_rate, experiment["ref"]["c_rate"], 6
    )


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
def test_optimal_charging_rate_err(experiment, request, spherical):
    """Test the prediction of the optimal C-rate."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(d=experiment["d"], z=3)
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
    greg._map = galpynostatic.base.MapSpline(spherical)
    greg.dcoeff_err_ = experiment["ref"]["dcoeff_err"]
    greg.k0_err_ = experiment["ref"]["k0_err"]

    c_rate, c_rate_err = galpynostatic.make_prediction.optimal_charging_rate(
        greg
    )

    np.testing.assert_array_almost_equal(
        c_rate, experiment["ref"]["c_rate"], 6
    )
    np.testing.assert_array_almost_equal(
        c_rate_err, experiment["ref"]["c_rate_err"], 6
    )


def test_raise_optimal_charging_rate(spherical):
    """Test the raise of the ValueError."""
    greg = galpynostatic.model.GalvanostaticRegressor(d=0.0015, z=3)
    greg.dcoeff_, greg.k0_ = 1.93e-10, 3.14e-7
    greg._map = galpynostatic.base.MapSpline(spherical)

    with pytest.raises(ValueError):
        galpynostatic.make_prediction.optimal_charging_rate(
            greg, c0=4.0, loaded=1
        )


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

    greg = galpynostatic.model.GalvanostaticRegressor(d=experiment["d"], z=3)
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
    greg._map = galpynostatic.base.MapSpline(spherical)
    greg.dcoeff_err_ = None

    size = galpynostatic.make_prediction.optimal_particle_size(
        greg, d0=1.2 * experiment["ref"]["particle_size"] / 10_000
    )

    np.testing.assert_array_almost_equal(
        size, experiment["ref"]["particle_size"], 6
    )


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
def test_optimal_particle_size_err(experiment, request, spherical):
    """Test the prediction of the optimal particle size."""
    experiment = request.getfixturevalue(experiment)

    greg = galpynostatic.model.GalvanostaticRegressor(d=experiment["d"], z=3)
    greg.dcoeff_, greg.k0_ = experiment["dcoeff"], experiment["k0"]
    greg._map = galpynostatic.base.MapSpline(spherical)
    greg.dcoeff_err_ = experiment["ref"]["dcoeff_err"]

    size, size_err = galpynostatic.make_prediction.optimal_particle_size(
        greg, d0=1.2 * experiment["ref"]["particle_size"] / 10_000
    )

    np.testing.assert_array_almost_equal(
        size, experiment["ref"]["particle_size"], 6
    )
    np.testing.assert_array_almost_equal(
        size_err, experiment["ref"]["particle_size_err"], 6
    )


def test_raise_optimal_particle_size(spherical):
    """Test the raise of the ValueError."""
    greg = galpynostatic.model.GalvanostaticRegressor(d=0.0015, z=3)
    greg.dcoeff_, greg.k0_ = 1.93e-10, 3.14e-7
    greg._map = galpynostatic.base.MapSpline(spherical)

    with pytest.raises(ValueError):
        galpynostatic.make_prediction.optimal_particle_size(
            greg, c_rate=60, loaded=1
        )
