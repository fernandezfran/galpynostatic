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

r"""The ``galpynostatic.dataset`` module loads the data needed for the fits.

These datasets were obtained using a continuous computational physics model for
different geometries [2]_. They come from a cutoff to multiple galvanostatic
profiles at a given cell potential, with respect to equilibrium, with different
combinations of the internal parameters :math:`\Xi` and :math:`\ell` covering a
wide range of possible values of the experimental variables involved.

References
----------
.. [2] Gavilán-Arriazu, E.M., Barraco, D., Ein-Eli, Y. and Leiva, E.P.M., 2022.
   Galvanostatic Fast Charging of Alkali‐Ion Battery Materials at the
   Single‐Particle Level: A Map‐Driven Diagnosis. `ChemPhysChem`.
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
