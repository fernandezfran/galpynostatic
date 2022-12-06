#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/39727040/matplotlib-2d-plot-from-x-y-z-values
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate

DF = pd.read_csv("sphere_150mV.dat", delimiter="\t")

Z = scipy.interpolate.interp2d(DF.l, DF.chi, DF.xmax)(DF.l, DF.chi)

fig, ax = plt.subplots()

im = ax.imshow(
    Z, extent=[DF.l.min(), DF.l.max(), DF.chi.min(), DF.chi.max()], origin="lower"
)
clb = plt.colorbar(im)
clb.ax.set_ylabel(r"x$_{max}$")

ax.scatter(DF.l, DF.chi, 400, facecolors="none")

ax.set_xlabel(r"log($\ell$)")
ax.set_ylabel(r"log($\Xi$)")

fig.savefig("map.png")
