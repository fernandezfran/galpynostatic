#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

r"""The ``galpynostatic.dataset`` module loads the data required for the fits.

These datasets have been obtained using a continuum computational physics model
for different geometries [5]_. They result from a cut-off of multiple
galvanostatic profiles at a given cell potential, with respect to equilibrium,
with different combinations of the internal unitless parameters :math:`\Xi` and
:math:`\ell` covering a wide range of possible values of the experimental
descriptors involved.

References
----------
.. [5] E. M. Gavilán-Arriazu, D. E. Barraco, D., Y. Ein-Eli and E. P. M. Leiva.
   "Galvanostatic Fast Charging of Alkali‐Ion Battery Materials at the
   Single‐Particle Level: A Map‐Driven Diagnosis.i" `ChemPhysChem` 24, no. 6
   (2023): e202200665.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# FUNCTIONS
# ============================================================================


def load_planar():
    """Galvanostatic planar map for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "planar.csv")


def load_cylindrical():
    """Galvanostatic cylindrical map for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "cylindrical.csv")


def load_spherical():
    """Galvanostatic spherical map for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "spherical.csv")
