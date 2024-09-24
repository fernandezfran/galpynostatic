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
def spherical():
    return galpynostatic.datasets.load_dataset()


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
            "dcoeff_err": 1.392838e-10,
            "k0_err": 1.076618e-06,
            "mse": 0.0029335,
            "soc": np.array(
                [0.941998, 0.886919, 0.831773, 0.722105, 0.464918]
            ),
            "r2": 0.9027825,
            "c_rate": 8.942052,
            "c_rate_err": 0.003725,
            "particle_size": 12.180608,
            "particle_size_err": 0.848281,
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
            "k0_err": 1.666335e-07,
            "mse": 0.0006329,
            "soc": np.array(
                [
                    0.979528,
                    0.961963,
                    0.938542,
                    0.909303,
                    0.821389,
                    0.490425,
                    0.295582,
                    0.2022,
                    0.139601,
                ]
            ),
            "r2": 0.9946694,
            "c_rate": 1.122309,
            "c_rate_err": 0.000204,
            "particle_size": 3.950719,
            "particle_size_err": 0.007418,
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
            "dcoeff_err": 4.745976e-13,
            "k0_err": 7.622031e-10,
            "mse": 0.0010655,
            "soc": np.array(
                [0.982609, 0.924677, 0.852255, 0.707423, 0.312355]
            ),
            "r2": 0.9812547,
            "c_rate": 1.36112,
            "c_rate_err": 1.075418e-05,
            "particle_size": 0.880512,
            "particle_size_err": 0.020894,
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
            "dcoeff_err": 1.835289e-08,
            "k0_err": 3.875681e-06,
            "mse": 0.002888,
            "soc": np.array(
                [0.988041, 0.979001, 0.960908, 0.906617, 0.816162, 0.635376]
            ),
            "r2": 0.8691324,
            "c_rate": 10.894016,
            "c_rate_err": 0.015179,
            "particle_size": 35.571076,
            "particle_size_err": 32.641602,
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
        "dcoeff": 9.999939e-13,
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
            "k0_err": 6.4625e-09,
            "mse": 0.006013,
            "soc": np.array(
                [0.987526, 0.973163, 0.949237, 0.901407, 0.757801, 0.523202]
            ),
            "r2": -0.35330742,
            "c_rate": 4.117637,
            "c_rate_err": 0.000151,
            "particle_size": 0.35571,
            "particle_size_err": 0.000504,
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
        "dcoeff": 9.999939e-13,
        "k0": 9.999939e-10,
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
            "dcoeff": 9.999939e-13,
            "k0": 9.999939e-10,
            "dcoeff_err": 4.023782e-13,
            "k0_err": 3.359260e-11,
            "mse": 0.0054603,
            "soc": np.array(
                [0.989701, 0.96014, 0.923195, 0.849311, 0.627971, 0.258862]
            ),
            "r2": 0.8475935,
            "c_rate": 26.674768,
            "c_rate_err": 3.846772e-05,
            "particle_size": 0.147765,
            "particle_size_err": 0.029729,
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
            "dcoeff_err": 1.082707e-10,
            "k0_err": 6.288231e-07,
            "mse": 0.0201603,
            "soc": np.array(
                [
                    0.955737,
                    0.872999,
                    0.653636,
                    0.370556,
                    0.152468,
                    0.056525,
                    0.005295,
                ]
            ),
            "r2": 0.8614687,
            "c_rate": 7.146494,
            "c_rate_err": 0.001948,
            "particle_size": 12.180608,
            "particle_size_err": 0.659401,
        },
    }
