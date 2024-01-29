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

"""A Python package with physics-based models to predict optimal conditions \
for fast-charging lithium-ion batteries."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib_metadata

from . import datasets
from .base import MapSpline
from .make_prediction import optimal_charging_rate, optimal_particle_size
from .metric import fom, umbem
from .model import GalvanostaticRegressor
from .preprocessing import GetDischargeCapacities


# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = [
    "datasets",
    "fom",
    "GalvanostaticRegressor",
    "GetDischargeCapacities",
    "MapSpline",
    "optimal_charging_rate",
    "optimal_particle_size",
    "umbem",
]


NAME = "galpynostatic"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
