#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import galpynostatic.lib

# =============================================================================
# TESTS
# =============================================================================


def test_docs():
    """Test the docs."""
    assert isinstance(galpynostatic.lib.__doc__, str)
