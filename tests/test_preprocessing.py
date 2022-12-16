#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================

import galpynostatic.preprocessing

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_get_discharge_capacities():
    """Test the get of discharge capacities."""
    with pytest.raises(NotImplementedError):
        galpynostatic.preprocessing.get_discharge_capacities()


def test_substact_equilibrium_potential():
    """Test the substraction of the equilibrium potential."""
    with pytest.raises(NotImplementedError):
        galpynostatic.preprocessing.substract_equilibrium_potential()


def test_substact_resistance():
    """Test the subtraction of the resistive contribution."""
    with pytest.raises(NotImplementedError):
        galpynostatic.preprocessing.substract_resistance()
