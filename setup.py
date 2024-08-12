#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

"""Compile C++ modules when `pip install galpynostatic`."""

from setuptools import Extension, setup

_flags = ["-fPIC", "-O3", "-ftree-vectorize", "-march=native"]

setup_args = dict(
    ext_modules=[
        Extension(
            name="galpynostatic.lib.profile",
            sources=["galpynostatic/lib/profile.cpp"],
            extra_compile_args=_flags,
        ),
        Extension(
            name="galpynostatic.lib.map",
            sources=["galpynostatic/lib/map.cpp"],
            extra_compile_args=_flags + ["-fopenmp"],
            extra_link_args=["-lgomp"],
        ),
    ]
)
setup(**setup_args)
