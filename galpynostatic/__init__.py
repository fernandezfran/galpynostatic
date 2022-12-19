#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# DOCS
# =============================================================================

"""Galpynostatic fitting program."""

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
