#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

DF = pd.read_csv("../sphere_150mV.dat", delimiter="\t")

logl = lambda d, crate, dcoeff: np.log10((d * d * crate) / (3 * 3600 * dcoeff))

logchi = lambda k0, crate, dcoeff: np.log10(k0 * np.sqrt(3600 / (crate * dcoeff)))

def find_nearest(arr, v):
    """Get the indices of the closest values of arr to v."""
    diffarr = np.abs(arr - v)
    return diffarr == diffarr.min()

def xmax_in_sphere_map(l, chi):
    """Find the xmax value given l and chi in base 10 logarithm."""
    maskl = find_nearest(DF.l, l)
    maskchi = find_nearest(DF.chi, chi)

    idx = np.argwhere(np.asarray(maskl & maskchi))[0][0]

    return DF.xmax[idx]
