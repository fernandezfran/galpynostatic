#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

import galpynostatic.datasets

import numpy as np

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture()
def data_path():
    return pathlib.Path(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
    )


@pytest.fixture()
def planar():
    return galpynostatic.datasets.load_planar()


@pytest.fixture()
def cylindrical():
    return galpynostatic.datasets.load_cylindrical()


@pytest.fixture()
def spherical():
    return galpynostatic.datasets.load_spherical()


@pytest.fixture()
def nishikawa():
    return {
        "dir_name": "LMNO",
        "file_names": (
            "2.5C.csv",
            "5.0C.csv",
            "7.5C.csv",
            "12.5C.csv",
            "25.0C.csv",
        ),
        "eq_pot": 4.739,
        "d": np.sqrt(0.25 * 8.04e-6 / np.pi),
        "C_rates": np.array([2.5, 5, 7.5, 12.5, 25.0]).reshape(-1, 1),
        "soc": np.array([0.996566, 0.976255, 0.830797, 0.725181, 0.525736]),
        "dcoeff": 1.0e-9,
        "k0": 1.0e-6,
        "ref": {
            "dc": np.array(
                [0.37869504, 0.3709768, 0.3157027, 0.2755689, 0.19977959]
            ),
            "dcoeff": 1.0e-9,
            "k0": 1.0e-6,
            "dcoeff_err": 1.08424e-10,
            "k0_err": 4.790873e-07,
            "mse": 0.00469549,
            "soc": np.array([0.937788, 0.878488, 0.81915, 0.701, 0.427025]),
            "r2": 0.8443919,
            "particle_size": 11.855359,
        },
    }


@pytest.fixture()
def mancini():
    return {
        "dir_name": "NATURAL_GRAPHITE",
        "file_names": None,
        "eq_pot": None,
        "d": 0.00075,
        "C_rates": np.array(
            [0.1, 0.2, 0.33333333, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]
        ).reshape(-1, 1),
        "soc": np.array(
            [
                0.992443,
                0.98205,
                0.964735,
                0.934943,
                0.853887,
                0.54003,
                0.296843,
                0.195002,
                0.125025,
            ]
        ),
        "dcoeff": 1.0e-10,
        "k0": 1.0e-6,
        "ref": {
            "dc": None,
            "dcoeff": 1.0e-10,
            "k0": 1.0e-6,
            "dcoeff_err": 3.755249e-13,
            "k0_err": 1.031205e-07,
            "mse": 0.00069059,
            "soc": np.array(
                [
                    0.979367,
                    0.961645,
                    0.938008,
                    0.908508,
                    0.819804,
                    0.48611,
                    0.290725,
                    0.197544,
                    0.135119,
                ]
            ),
            "r2": 0.9941839,
            "particle_size": 3.916924,
        },
    }


@pytest.fixture()
def he():
    return {
        "dir_name": "LTO",
        "file_names": ("0.1C.csv", "0.5C.csv", "1C.csv", "2C.csv", "5C.csv"),
        "eq_pot": 1.57,
        "d": 0.000175,
        "C_rates": np.array([0.1, 0.5, 1.0, 2.0, 5.0]).reshape(-1, 1),
        "soc": np.array([0.995197, 0.958646, 0.845837, 0.654458, 0.346546]),
        "dcoeff": 1.0e-11,
        "k0": 1.0e-8,
        "ref": {
            "dc": np.array(
                [159.23154, 153.38335, 135.33395, 104.71328, 55.44732]
            ),
            "dcoeff": 1.0e-11,
            "k0": 1.0e-8,
            "dcoeff_err": 1.80632e-12,
            "k0_err": 1.5790638e-09,
            "mse": 0.006482,
            "soc": np.array(
                [0.978918, 0.906247, 0.815342, 0.633649, 0.179112]
            ),
            "r2": 0.8859801,
            "particle_size": 0.692001,
        },
    }


@pytest.fixture()
def wang():
    return {
        "dir_name": "LCO",
        "file_names": (
            "0.5C.csv",
            "1C.csv",
            "2C.csv",
            "5C.csv",
            "10C.csv",
            "20C.csv",
        ),
        "eq_pot": 3.9,
        "d": 0.002,
        "C_rates": np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]).reshape(-1, 1),
        "soc": np.array(
            [0.994179, 0.967568, 0.930123, 0.834509, 0.734328, 0.569661]
        ),
        "dcoeff": 1.0e-8,
        "k0": 1.0e-6,
        "ref": {
            "dc": np.array(
                [99.417946, 96.75683, 93.01233, 83.45085, 73.432816, 56.96607]
            ),
            "dcoeff": 1.0e-8,
            "k0": 1.0e-6,
            "dcoeff_err": 2.667671e-09,
            "k0_err": 3.22267e-07,
            "mse": 0.000863,
            "soc": np.array(
                [0.985938, 0.974779, 0.952477, 0.885544, 0.774095, 0.550382]
            ),
            "r2": 0.9609059,
            "particle_size": 32.699122,
        },
    }


@pytest.fixture()
def lei():
    return {
        "dir_name": "LFP",
        "file_names": (
            "0.2C.csv",
            "0.5C.csv",
            "1C.csv",
            "2C.csv",
            "5C.csv",
            "10C.csv",
        ),
        "eq_pot": 3.45,
        "d": 3.5e-5,
        "C_rates": np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1),
        "soc": np.array(
            [0.948959, 0.836089, 0.759624, 0.329323, 0.020909, 0.00961]
        ),
        "dcoeff": 1.0e-13,
        "k0": 1.0e-8,
        "ref": {
            "dc": np.array(
                [
                    160.27911,
                    141.21538,
                    128.30042,
                    55.622726,
                    3.5315654,
                    1.6230893,
                ]
            ),
            "dcoeff": 1.0e-13,
            "k0": 1.0e-8,
            "dcoeff_err": 2.834146e-15,
            "k0_err": 3.682211e-09,
            "mse": 0.006019,
            "soc": np.array(
                [0.918072, 0.799457, 0.604216, 0.325994, 0.112047, 0.046371]
            ),
            "r2": 0.9589378,
            "particle_size": 0.118554,
        },
    }


@pytest.fixture()
def bak():
    return {
        "dir_name": "LMO",
        "file_names": (
            "1C.csv",
            "5C.csv",
            "10C.csv",
            "20C.csv",
            "50C.csv",
            "100C.csv",
        ),
        "eq_pot": 4.0,
        "d": 2.5e-6,
        "C_rates": np.array([1, 5, 10, 20, 50, 100]).reshape(-1, 1),
        "soc": np.array(
            [0.9617, 0.938762, 0.9069, 0.863516, 0.696022, 0.421418]
        ),
        "dcoeff": 1.0e-13,
        "k0": 1.0e-8,
        "ref": {
            "dc": np.array(
                [
                    127.32904,
                    124.29212,
                    120.07357,
                    114.329475,
                    92.15335,
                    55.795765,
                ]
            ),
            "dcoeff": 1.0e-13,
            "k0": 1.0e-8,
            "dcoeff_err": 4.23587e-12,
            "k0_err": 6.579867e-07,
            "mse": 0.016352,
            "soc": np.array(
                [0.993918, 0.981223, 0.965346, 0.933609, 0.838375, 0.679639]
            ),
            "r2": 0.5436138,
            "particle_size": 0.118554,
        },
    }


@pytest.fixture()
def dokko():
    return {
        "dir_name": "GRAPHITE",
        "file_names": None,
        "eq_pot": None,
        "d": 0.0009,
        "C_rates": np.array([1.5, 4.5, 12.5, 25, 50, 100, 250]).reshape(-1, 1),
        "soc": np.array([0.952, 0.947, 0.928, 0.586, 0.214, 0.157, 0.013]),
        "dcoeff": 1.0e-9,
        "k0": 1.0e-6,
        "ref": {
            "dc": None,
            "dcoeff": 1.0e-9,
            "k0": 1.0e-6,
            "dcoeff_err": 1.472138e-10,
            "k0_err": 5.238026e-07,
            "mse": 0.02627097,
            "soc": np.array(
                [
                    0.952891,
                    0.864461,
                    0.630083,
                    0.333588,
                    0.120853,
                    0.031413,
                    0.0,
                ]
            ),
            "r2": 0.8194799,
            "particle_size": 11.855359,
        },
    }
