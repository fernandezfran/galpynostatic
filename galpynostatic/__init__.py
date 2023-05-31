#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""A physics-based heuristic model to predict the optimal electrode particle \
size for a fast-charging of lithium-ion batteries."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib_metadata

from . import datasets
from .make_prediction import optimal_particle_size
from .model import GalvanostaticRegressor
from .preprocessing import GetDischargeCapacities


# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = [
    "datasets",
    "GalvanostaticRegressor",
    "GetDischargeCapacities",
    "optimal_particle_size",
]


NAME = "galpynostatic"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
