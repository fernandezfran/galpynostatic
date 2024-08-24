#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""A Python package with physics-based and data-driven models to predict \
optimal conditions for fast-charging lithium-ion batteries."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib_metadata

from . import datasets
from .base import MapSpline
from .datasets import params
from .make_prediction import optimal_charging_rate, optimal_particle_size
from .metric import bmxfc, fom
from .model import GalvanostaticRegressor
from .preprocessing import GetDischargeCapacities
from .simulation import GalvanostaticMap, GalvanostaticProfile


# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = [
    "bmxfc",
    "datasets",
    "fom",
    "GalvanostaticMap",
    "GalvanostaticProfile",
    "GalvanostaticRegressor",
    "GetDischargeCapacities",
    "MapSpline",
    "optimal_charging_rate",
    "optimal_particle_size",
    "params",
]


NAME = "galpynostatic"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
