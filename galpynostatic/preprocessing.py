#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================================
# DOCS
# ============================================================================

"""Common utilities for experimental data processing."""

# ============================================================================
# IMPORTS
# ============================================================================


# ============================================================================
# FUNCTIONS
# ============================================================================


def get_discharge_capacities():
    """Obtain the discharge capacities at a given cut-off potential."""
    raise NotImplementedError


def substract_equilibrium_potential():
    """Substract the equilibrium potential of the galvanostatic profiles."""
    raise NotImplementedError


def substract_resistance():
    """Substract the resistance contribution of the galvanostatic profiles."""
    raise NotImplementedError
