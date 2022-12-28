#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""A physics-based heuristic model for identifying the five minutes charging \
electrode material."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib_metadata

from . import datasets
from .model import GalvanostaticRegressor
from .preprocessing import get_discharge_capacities


# =============================================================================
# IMPORTS
# =============================================================================

__all__ = ["datasets", "GalvanostaticRegressor", "get_discharge_capacities"]


NAME = "galpynostatic"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
