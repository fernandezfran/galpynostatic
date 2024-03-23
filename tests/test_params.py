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

import galpynostatic.datasets.params

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("material", "density", "specific_capacity"),
    [
        ("Graphite", 2.2, 370),
        ("Graphite-Silicon", 1.25, 1000),
        ("Silicon", 2.33, 3500),
        ("LTO", 4.5, 170),
        ("LCO", 5.0, 140),
        ("LFP", 3.55, 170),
        ("LMO", 4.0, 120),
        ("NCA", 4.4, 200),
        ("NCO46", 4.5, 200),
        ("NMC", 4.5, 175),
        ("NMC111", 4.8, 200),
        ("NMC523", 4.8, 190),
        ("NMC622", 4.7, 200),
        ("NMC811", 4.8, 220),
    ],
)
class TestParams:
    """Test the dataset params of different electrode materials."""

    def test_density(self, material, density, specific_capacity):
        """Test the density param."""
        np.testing.assert_almost_equal(
            galpynostatic.datasets.params.Electrode(material).density,
            density,
            6,
        )

    def test_specific_capacity(self, material, density, specific_capacity):
        """Test the density param."""
        np.testing.assert_almost_equal(
            galpynostatic.datasets.params.Electrode(
                material
            ).specific_capacity,
            specific_capacity,
            6,
        )


def test_value_error():
    """Test the raise of the ValueError."""
    with pytest.raises(ValueError):
        galpynostatic.datasets.params.Electrode("graphite")
