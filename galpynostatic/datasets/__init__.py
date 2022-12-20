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

"""The `galpynostatic.dataset` module loads the data needed for the fits."""

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
    raise NotImplementedError


def load_cylindrical():
    """Galvanostatic cylindrical map for a cut-off potential of 150 mV."""
    raise NotImplementedError


def load_spherical():
    """Galvanostatic spherical map for a cut-off potential of 150 mV."""
    return pd.read_csv(PATH / "spherical.tsv", delimiter="\t")
