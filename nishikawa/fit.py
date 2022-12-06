#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

DF = pd.read_csv("../sphere_150mV.dat", delimiter="\t")


def l(crate, d, dcoeff):
    """Value of l given experimental conditions of the system."""
    return (d * d * crate) / (3 * 3600 * dcoeff)


def chi(crate, k0, dcoeff):
    """Value of chi given experimental conditions of the system."""
    return k0 * np.sqrt(3600 / (crate * dcoeff))


def find_nearest(arr, v):
    """Get the indices of the closest values of arr to v."""
    diffarr = np.abs(arr - v)
    return diffarr == diffarr.min()


def xmax_in_sphere_map(l, chi):
    """Find the xmax value given l and chi."""
    maskl = find_nearest(DF.l, np.log10(l))
    maskchi = find_nearest(DF.chi, np.log10(chi))

    idx = np.argwhere(np.asarray(maskl & maskchi))[0][0]

    return DF.xmax[idx]


def grid_search(y_true, d, crates, dcoeffs, k0s):
    """Grid search for diffusion coefficients and rate constants."""
    min_mse = np.inf

    for dcoeff, k0 in it.product(k0s, dcoeffs):

        y_pred = [
            xmax_in_sphere_map(l(crate, d, dcoeff), chi(crate, k0, dcoeff))
            for crate in crates
        ]

        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)

        if mse < min_mse:
            min_mse = mse
            pred = y_pred
            best_coeffs = (dcoeff, k0)

    return best_coeffs, pred, min_mse
