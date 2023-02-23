#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

r"""The ``galpynostatic.dataset`` module loads the data needed for the fits.

These datasets were obtained using a computational physics continuum model [1]_
for different geometries. They come from a cutoff of a surface at a given cell
potential, with respect to equilibrium, and from different combinations of
internal parameters :math:`\Xi` and :math:`\ell` (see
:ref:`galpynostatic.utils`) that covers a wide range of possible values of the
experimental variables involved.

References
----------
.. [1] Gavilán-Arriazu, E.M., Barraco, D., Ein-Eli, Y. and Leiva, E.P.M., 2022.
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
    """Galvanostatic planar diagram for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "planar.tsv", delimiter="\t")


def load_cylindrical():
    """Galvanostatic cylindrical diagram for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "cylindrical.tsv", delimiter="\t")


def load_spherical():
    """Galvanostatic spherical diagram for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "spherical.tsv", delimiter="\t")
