#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""GalvanostaticProfile class of simulation module."""

# ============================================================================
# IMPORTS
# ============================================================================

import ctypes as ct
import os
import pathlib

# import sysconfig

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.interpolate

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# TODO: modify to sysconfig
_PROFILE_LIBS = ct.CDLL(PATH / "lib" / "profile.so")
# end TODO

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticProfile:
    r"""A tool to extrapolate isotherms varing C-rate and particle size.

    This software simulates new isotherms individually, from an
    experimental or theoretical one, dependding on the value of
    :math:`\log(\ell)` and :math:`\log(\Xi)`. Given an isotherm for a
    particular particle size and C-rate this tool will predict the system
    behaviour varing :math:`\log(\ell)` and :math:`\log(\Xi)`. The
    resulting isotherms are calculated by means of the interpolation of
    the experimental or theoretical isotherm, the Fick's diffusion law and
    the Butler-Volmer charge transfer equation with a transfer coefficient
    of 0.5.

    ///------------------------------------------------------------------------
    ///         	 Galvanostatic simulation code for diagram construction
    ///------------------------------------------------------------------------
    ///------------------------------------------------------------------------
    /// This simulation code was written to simulate the charging process of a
    /// single-particle electrode of a lithium-ion battery. To solve the Fick
    /// diffusion equation the Backward-Implicit method was applied. The
    /// electrode/electrolyte interface kinetics is simulated by the
    /// Butler-Volmer equation using experimental curves for the equilibrium
    /// potential. The programm generates a diagram consisting in N (Xi,L)
    /// data points. The present codes was parallelized with OpenMP at the
    /// level of a point in L for different Xi.
    ///---------------------------------------------------------------------------

    Parameters
    ----------
    density : float
        Density of the electrode active material in :math:`g/cm^3`.

    xi : float, default=2.0
        Value of the :math:`\log(\Xi)`.

    ell : float, default=2.0
        Value of the :math:`\log(\ell)`.

    isotherm : bool or pandas.DataFrame, default=False
        A dataset containing the experimental isotherm values in the
        format SOC vs potential. The isotherm is used to calculate the
        equilibrium potential for a given SOC. If False the equilibrium
        potential is calculated using the theroretical model ... [].

    specific_capacity : bool or float, default=None
        Specific capacity of the material in `mAh/g`. If isotherm
        is None the specific capacity must be defined.

    mass : float, default=1
        Total mass of the electrode active material in :math:`g`.

    vcut : float, default=-0.15
        Cut potential if isotherm=False.

    geometrical_param : int default=2
        Active material particle geometrical_parammetry. 0=planar,
        1=cylindrical, 2=spherical.

    g : float, default=0.0
        Interaction parameter of the theroretical model used to obtain
        the equilibrium potential if isotherm=False.

    profile_soc : float default=0.5
        SOC value at wich the concentration profile will be calculated.

    temperature : float, default=298.0
        Working temperature of the cell.

    resistance : float, default=0.0
        Cell's resistance.

    grid_size : int, default=1000
        Size of the spatial grid in wich the Fick's equation will be
        solved.

    time_steps : int, default=3000000
        Size of the time grid in wich the Fick's equation will be solved.

    each : int, default=100
        time_steps/each time points at which SOC and potential are printed.

    Attributes
    ----------
    isotherm_df : pandas.DataFrame
        A dataset containig the resulting isotherm in the potential vs SOC
        format.

    concentration_df : pandas.DataFrame
        A dataset containing the resulting concentration profile in the
        form :math:`\theta` (concentration) vs :math:`r` (distance to the
        center of the particle).

    """

    def __init__(
        self,
        density,
        xi,
        ell,
        isotherm=False,
        specific_capacity=None,
        mass=1.0,
        vcut=-0.15,
        geometrical_param=2,
        g=0.0,
        profile_soc=0.5,
        temperature=298.0,
        resistance=0.0,
        grid_size=1_000,
        time_steps=100_000,
        each=100,
    ):
        self.density = density
        self.xi = xi
        self.ell = ell
        self.isotherm = isotherm
        self.specific_capacity = specific_capacity
        self.mass = mass
        self.vcut = vcut
        self.geometrical_param = geometrical_param
        self.g = g
        self.profile_soc = profile_soc
        self.temperature = temperature
        self.resistance = resistance
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.each = each

        if (
            not isinstance(self.isotherm, pd.DataFrame)
            and not self.specific_capacity
        ):
            raise ValueError(
                "If no isotherm is given specific_capacity must be defined"
            )

        if isinstance(isotherm, pd.DataFrame):
            self.frumkin = False
            self.isotherm = SplineCoeff(isotherm)
            self.isotherm.get_coeffs()
        else:
            self.isotherm = SplineCoeff(
                pd.DataFrame(
                    {
                        "capacity": [self.specific_capacity],
                        "potential": [self.vcut],
                    }
                )
            )
            self.isotherm.spl_ai = np.array(0)
            self.isotherm.spl_bi = np.array(0)
            self.isotherm.spl_ci = np.array(0)
            self.isotherm.spl_di = np.array(0)
            self.isotherm.capacity = np.array(0)
            self.frumkin = True

    def run(self):
        """Run the isotherm simulation."""
        lib_galva = _PROFILE_LIBS

        lib_galva.galva.argtypes = [
            ct.c_bool,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]

        N = int(self.time_steps / self.each)

        res_soc = (ct.c_double * N)()
        res_voltage = (ct.c_double * N)()
        res_norm = (ct.c_double * self.grid_size)()
        res_cons = (ct.c_double * self.grid_size)()

        lib_galva.galva(
            self.frumkin,
            self.g,
            self.grid_size,
            self.time_steps,
            self.each,
            self.isotherm.isotherm_len,
            self.temperature,
            self.mass,
            self.density,
            self.resistance,
            self.isotherm.vcut,
            self.isotherm.specific_capacity,
            self.geometrical_param,
            self.xi,
            self.ell,
            self.profile_soc,
            self.isotherm.spl_ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res_soc,
            res_voltage,
            res_norm,
            res_cons,
        )

        self.SOC = np.asarray(np.frombuffer(res_soc, dtype=np.double, count=N))

        self.E = np.asarray(np.frombuffer(res_voltage, dtype=float, count=N))

        self.r_norm = np.asarray(
            np.frombuffer(res_norm, dtype=np.double, count=self.grid_size)
        )

        self.tita1 = np.asarray(
            np.frombuffer(res_cons, dtype=np.double, count=self.grid_size)
        )

        self.isotherm_df = pd.DataFrame(
            {
                "SOC": self.SOC,
                "Potential": self.E,
            }
        )

        self.isotherm_df = self.isotherm_df.loc[
            (self.isotherm_df != 0).any(axis=1)
        ].reset_index()

        self.concentration_df = pd.DataFrame(
            {"r_norm": self.r_norm, "theta": self.tita1}
        )

    @property
    def isotherm_df(self):
        """Simulated isotherm data set in a pandas format."""
        return self.isotherm_df

    @property
    def concentration_dataframe(self):
        """Concentration dataset in a pandas format."""
        return self.concentration_df

    def isotherm_plot(self, ax=None, plt_kws=None):
        """Plot the simulated isotherm.

        Parameters
        ----------
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.isotherm_df["SOC"]
        y = self.isotherm_df["Potential"]

        ax.plot(x, y, **plt_kws)
        ax.set_xlabel("SOC")
        ax.set_ylabel("Potential")

        return ax

    def consentration_plot(self, ax=None, plt_kws=None):
        """Plot the consentration inside the particle.

        Parameters
        ----------
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.concentration_df["r_norm"]
        y = self.concentration_df["theta"]

        ax.plot(x, y, color="tab:red", **plt_kws)

        ax.set_xlabel("$r_{norm}$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(f"Concentration profile, SOC={self.profile_soc}")
        # ax.legend()

        return ax


class SplineCoeff:
    r"""Spline coefficients class.

    Spline coefficients of a given equilibrium isotherm. The coefficients are
    obtained using the interpolate.CubicSpline function of the scipy package.
    These coefficients are used to interpolate the equilibrium potential values
    as :math:`E^0(x_d)=a_i+b_i(x_d-x_j)+c_i(x_d-x_j)^2+d_i(x_d-x_i)^3`

    Parameters
    ----------
    dataset : pandas.DataFrame
        A dataset containing the experimental isotherm values in the
        format SOC vs potential.

    Attributes
    ----------
    spl_ai : numpy.ndarray
        Array of the independent term coefficients of the cubic spline.

    spl_bi : numpy.ndarray
        Array of the lineal term coefficients of the cubic spline.

    spl_ci : numpy.ndarray
        Array of the cuadratic term coefficients of the cubic spline.

    spl_di : numpy.ndarray
        Array of the cubic term coefficients of the cubic spline.
    """

    def __init__(self, dataset):
        self.dataset = dataset

        capacity = self.dataset.iloc[:, 0].values
        self.specific_capacity = np.max(capacity)
        self.capacity = capacity / self.specific_capacity

        self.potential = self.dataset.iloc[:, 1].values
        self.vcut = np.min(self.potential)

        self.isotherm_len = self.dataset.shape[0]

    def get_coeffs(self):
        """Calculate the spline coefficients.

        The function get_params takes the  normalized experimental
        capacity or the smooth isotherm.
        It returns the parameters ai, bi, ci, and di of the cubic
        spline of the isotherm. These parameters can be used to
        calculate the equilibrium potential.
        """
        isotherm_spl = scipy.interpolate.CubicSpline(
            self.capacity, self.potential
        )

        self.spl_ai = isotherm_spl.c[0, :]
        self.spl_bi = isotherm_spl.c[1, :]
        self.spl_ci = isotherm_spl.c[2, :]
        self.spl_di = isotherm_spl.c[3, :]
