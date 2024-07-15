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
import sysconfig

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.interpolate

from .utils import logcrate, logd

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class GalvanostaticMap:
    r"""Diagnostic map tool for galvanostatic intercalation material analysis.

    The present software performs a series of galvanostatic simulations
    to systematically investigate the maximum state of charge (SoC) that a
    material is capable of accommodating at the single particle level under
    different experimental conditions. These simulations are used to produce
    a diagnostic map or zone diagram [2, 3]_.
    A similar concept is used here to construct level diagrams for
    galvanostatic simulations using two variables:

    :math:`\Xi = k^{0}\left (\frac{t_{h}}{C_{r}D}  \right )^{1/2}` and
    :math:`\ell=\frac{r^{2}C_{r}}{zt_{h}D}`
    at constant :math:`D` and :math:`k_0`.

    The SOC are calculated by means of the interpolation of the experimental
    or theoretical isotherm, the Fick's diffusion law and the Butler-Volmer
    charge transfer equation with a transfer coefficient of 0.5. The Fick
    diffusion equation is solved using the Crank-Nicolson method.

    Parameters
    ----------
    density : float, default=1.0
        Density of the electrode active material in :math:`g/cm^3`.

    isotherm : bool or pandas.DataFrame, default=False
        A dataset containing the experimental isotherm values in the
        format potential vs capacity. The isotherm is used to calculate
        the equilibrium potential for a given SOC. If False the
        equilibrium potential is calculated using the theroretical model
        ... [].

    specific_capacity : bool or float, default=1.0
        Specific capacity of the material in `mAh/g`. If isotherm
        is None the specific capacity must be defined.

    mass : float
        Total mass of the electrode active material in :math:`g`.

    vcut : float, default=-0.15
        Cut potential of the simulation.

    g : float, default=0.0
        Interaction parameter of the theroretical model used to obtain
        the equilibrium potential if isotherm=False.

    geometrical_param : int default=3
        Active material particle geometrical_parammetry. 1=planar,
        1=cylindrical, 3=spherical.

    temperature : float, default=298.0
        Working temperature of the cell.

    logxi_lle : float, default=2.0
        Initial value of the :math:`\log(\Xi)`.

    logxi_ule : float, default=-4.0
        Final value of the :math:`\log(\Xi)`.

    num_xi : int, default=5
        Number of :math:`\log(\Xi)` values.

    logell_lle : float, default=2.0
        Initial value of the :math:`\log(\ell)`.

    logell_ule : float, default=-4.0
        Final value of the :math:`\log(\ell)`.

    num_ell : int, default=5
        Number of :math:`\log(\ell)` values.

    grid_size : int, default=1000
        Size of the spatial grid in wich the Fick's equation will be solved.

    time_steps : int, default=100000
        Size of the time grid in wich the Fick's equation will be solved.

    nthreads : int, default=-1
        Number of threads in which the diagram calculation will be performed.
        -1 means use all available threads.
    """

    def __init__(
        self,
        density=1.0,
        isotherm=None,
        specific_capacity=1.0,
        mass=1.0,
        vcut=-0.15,
        g=0.0,
        geometrical_param=3,
        temperature=298.0,
        logxi_lle=2.0,
        logxi_ule=-4.0,
        num_xi=32,
        logell_lle=2.0,
        logell_ule=-4.0,
        num_ell=32,
        grid_size=1_000,
        time_steps=100_000,
        nthreads=-1,
    ):
        self.density = density
        self.isotherm = isotherm
        self.specific_capacity = specific_capacity
        self.mass = mass
        self.temperature = temperature
        self.g = g
        self.geometrical_param = geometrical_param
        #        self.veq = veq
        self.vcut = vcut
        self.logxi_lle = logxi_lle
        self.logxi_ule = logxi_ule
        self.num_xi = num_xi
        self.logell_lle = logell_lle
        self.logell_ule = logell_ule
        self.num_ell = num_ell
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.nthreads = nthreads

        if isinstance(self.isotherm, pd.DataFrame):
            self.frumkin = False
            self.isotherm = SplineCoeff(self.isotherm)
            # self.isotherm = SplineParams(df)
            self.isotherm.get_coeffs()
            self.isotherm.vcut = self.vcut

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

        self.logL_ = np.linspace(
            self.logell_lle, self.logell_ule, self.num_ell
        )
        self.logxi_ = np.linspace(self.logxi_lle, self.logxi_ule, self.num_xi)

    def run(self):
        """Run the diagram simulation."""
        lib_map = ct.CDLL(
            str(PATH / "lib" / "map") + sysconfig.get_config_var("EXT_SUFFIX")
        )

        lib_map.run_map.argtypes = [
            ct.c_bool,
            ct.c_double,
            ct.c_int,
            ct.c_int,
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
            ct.POINTER(ct.c_double),
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

        N = int(self.num_ell * self.num_xi)

        res_logell = (ct.c_double * N)()
        res_logxi = (ct.c_double * N)()
        res_socmax = (ct.c_double * N)()

        lib_map.run_map(
            self.frumkin,
            self.g,
            self.nthreads,
            self.grid_size,
            self.time_steps,
            self.isotherm.isotherm_len,
            self.num_ell,
            self.num_xi,
            self.temperature,
            self.mass,
            self.density,
            self.isotherm.vcut,
            self.isotherm.specific_capacity,
            self.geometrical_param,
            self.logL_.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.logxi_.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_ai.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_bi.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_ci.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.spl_di.ctypes.data_as(ct.POINTER(ct.c_double)),
            self.isotherm.capacity.ctypes.data_as(ct.POINTER(ct.c_double)),
            res_logell,
            res_logxi,
            res_socmax,
        )

        self.logL = np.asarray(
            np.frombuffer(res_logell, dtype=np.double, count=N)
        )

        self.logxi = np.asarray(
            np.frombuffer(res_logxi, dtype=np.double, count=N)
        )

        self.SOC = np.asarray(
            np.frombuffer(res_socmax, dtype=np.double, count=N)
        )

        self.SOC = np.clip(self.SOC, 0, 1)

        self.df = pd.DataFrame(
            {
                "ell": self.logL,
                "xi": self.logxi,
                "SOC": self.SOC,
            }
        ).sort_values(
            by=["ell", "xi"], ascending=[True, True], ignore_index=True
        )

    @property
    def map_dataframe(self):
        """Convert the diagram dataset into a dataframe."""
        return self.df

    def map_plot(
        self, ax=None, plt_kws=None, clb=True, clb_label="SoC$_{max}$"
    ):
        """Plot the two dimensional diagram.

        Parameters
        ----------
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.

        clb : bool, default=True
            Parameter that determines if the color bar will be displayed.

        clb_label : str, default="SOC"
            Name of the color bar.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        x = self.df.ell
        y = self.df.xi

        logells_ = np.unique(x)
        logxis_ = np.unique(y)
        socs = self.df.SOC.to_numpy().reshape(logells_.size, logxis_.size)

        spline_ = scipy.interpolate.RectBivariateSpline(
            logells_, logxis_, socs
        )

        xeval = np.linspace(x.min(), x.max(), 1000)
        yeval = np.linspace(y.min(), y.max(), 1000)

        z = spline_(xeval, yeval, grid=True)

        im = ax.imshow(
            z.T,
            extent=[
                xeval.min(),
                xeval.max(),
                yeval.min(),
                yeval.max(),
            ],
            origin="lower",
            **plt_kws,
        )

        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($\ell$)")
        ax.set_ylabel(r"log($\Xi$)")
        # ax.set_title(f"Diagram, g={self.g}")

        return ax

    def real_plot(
        self,
        dcoeff,
        k0,
        ax=None,
        plt_kws=None,
        clb=True,
        clb_label="$SoC_{max}$",
    ):
        """Real diagram axis.

        Parameters
        ----------
        dcoeff : float
            Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

        k0 : float
            Kinetic rate constant, :math:`k^0`, in :math:`cm/s`.
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.

        clb : bool, default=True
            Parameter that determines if the color bar will be displayed.

        clb_label : str, default="SOC"
            Name of the color bar.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        l_log = self.df.ell.values
        xi_log = self.df.xi.values

        logcrate_ = logcrate(xi_log, dcoeff, k0)
        logd_ = logd(xi_log, l_log, dcoeff, k0, self.geometrical_param + 1)

        self.df["log_crate"] = logcrate_
        self.df["log_d"] = logd_

        x = np.linspace(
            self.df["log_crate"].min(), self.df["log_crate"].max(), 1000
        )
        y = np.linspace(self.df["log_d"].min(), self.df["log_d"].max(), 1000)

        X, Y = np.meshgrid(x, y)

        # Interpolar los datos en la malla uniforme
        Z = np.clip(
            scipy.interpolate.griddata(
                (self.df["log_crate"], self.df["log_d"]),
                self.df["SOC"],
                (X, Y),
                method="linear",
            ),
            0,
            1,
        )

        im = ax.imshow(
            Z,
            extent=[
                X.min(),
                X.max(),
                Y.min(),
                Y.max(),
            ],
            origin="lower",
            **plt_kws,
        )

        if clb:
            clb = plt.colorbar(im)
            clb.ax.set_ylabel(clb_label)
            clb.ax.set_ylim((0, 1))

        ax.set_xlabel(r"log($C_r$)")
        ax.set_ylabel(r"log($d$)")

        return ax


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
    of 0.5. The Fick diffusion equation is solved using the Crank-Nicolson
    method.

    Parameters
    ----------
    density : float, default=1.0
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

    specific_capacity : bool or float, default=1.0
        Specific capacity of the material in `mAh/g`. If isotherm
        is None the specific capacity must be defined.

    mass : float, default=1
        Total mass of the electrode active material in :math:`g`.

    vcut : float, default=-0.15
        Cut potential if isotherm=False.

    geometrical_param : int default=3
        Active material particle geometrical_parammetry. 1=planar,
        1=cylindrical, 3=spherical.

    g : float, default=0.0
        Interaction parameter of the theroretical model used to obtain
        the equilibrium potential if isotherm=False.

    profile_soc : float default=0.5
        SOC value at wich the concentration profile will be calculated.

    temperature : float, default=298.0
        Working temperature of the cell.

    grid_size : int, default=1000
        Size of the spatial grid in wich the Fick's equation will be
        solved.

    time_steps : int, default=100000
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
        xi,
        ell,
        density=1.0,
        isotherm=False,
        specific_capacity=1.0,
        mass=1.0,
        vcut=-0.15,
        geometrical_param=3,
        g=0.0,
        profile_soc=0.5,
        temperature=298.0,
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
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.each = each

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
        lib_profile = ct.CDLL(
            str(PATH / "lib" / "profile")
            + sysconfig.get_config_var("EXT_SUFFIX")
        )

        lib_profile.run_profile.argtypes = [
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

        lib_profile.run_profile(
            self.frumkin,
            self.g,
            self.grid_size,
            self.time_steps,
            self.each,
            self.isotherm.isotherm_len,
            self.temperature,
            self.mass,
            self.density,
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
        ].reset_index(drop=True)

        self.concentration_df = pd.DataFrame(
            {"r_norm": self.r_norm, "theta": self.tita1}
        )

    @property
    def profile_dataframe(self):
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
        ax.set_xlabel("SoC")
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

        The function get_coeffs takes the  normalized experimental
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


class ProfileFitting:
    r"""Tool to fit :math:`k^0` and :math:`D` for non equilibrium isotherms.

    :math:`k^0` and :math:`D` are obtained using a simulated isotherm
    as imput of a scipy.optimize.curve_fit considering an experimental
    equilibrium isotherm and an out equilibrium isotherm.

    Parameters
    ----------
    equilibrium_iso : pandas.DataFrame
        A dataset containing the experimental equilibrium isotherm values
        in the format SOC vs potential.

    objective_iso : pandas.DataFrame
        A dataset containing the experimental no equilibrium isotherm
        values in the format SOC vs potential.

    density : float
        Density of the material in :math: `g/cm^3`.

    crate : int or float
        Crate of the non equilibrium isotherm.

    particle_size : float
        Experimental active material diffusion length in :math: `cm`.

    geometrical_param : int, default=2
        Active material particle geometrical_parammetry. 1=planar,
        2=cylindrical, 3=spherical.

    Attributes
    ----------
    logxi : float
        Fitted log value in base 10 of :math:`\Xi`.

    logell : float
        Fitted log value in base 10 of :math:`\ell`.

    dcoeff : float
        Diffusion coefficient, :math:`D`, in :math:`cm^2/s`.

    k0 : float
        Kinetic rate constant, :math:`k^0`, in :math:`cm/s`.
    """

    def __init__(
        self,
        isotherm,
        objective_iso,
        density,
        crate,
        particle_size,
        geometrical_param=3,
    ):
        self.isotherm = isotherm
        self.objective_iso = objective_iso
        self.density = density
        self.crate = crate
        self.particle_size = particle_size
        self.geometrical_param = geometrical_param

    def fit_data(self):
        """Fit the non equilibrium isotherm."""

        def fit_function(xdata, xi, ell):
            iso = GalvanostaticProfile(
                self.density, xi, ell, isotherm=self.isotherm
            )
            iso.run()
            soc = iso.isotherm_df.SOC
            pot = iso.isotherm_df.Potential
            spl = scipy.interpolate.CubicSpline(soc, pot)
            return spl(xdata)

        maximum = self.isotherm.iloc[:, 0].max()
        fit = scipy.optimize.curve_fit(
            fit_function,
            self.objective_iso.iloc[:, 0] / maximum,
            self.objective_iso.iloc[:, 1],
            p0=[-1, -1],
            bounds=([[-4, -4], [2, 2]]),
        )

        self.logxi, self.logell = fit[0]

        self.dcoeff = (
            self.particle_size**2
            * self.crate
            / (self.geometrical_param * 3600 * 10**self.logell)
        )

        self.k0 = (
            10**self.logxi
            * self.crate
            * self.particle_size
            / 3600
            * np.sqrt(1 / ((self.geometrical_param - 1) * 10**self.logell))
        )

        return [self.dcoeff, self.k0]

    def plot_fit(self, ax=None, plt_kws=None):
        """Plot the fitted non equilibrium isotherm.

        Parameters
        ----------
        ax : axis, default=None
            Axis of wich the diagram plot.

        plt_kws : dict, default=None
            A dictionary containig the parameters to be passed to the axis.
        """
        ax = plt.gca() if ax is None else ax
        plt_kws = {} if plt_kws is None else plt_kws

        iso = GalvanostaticProfile(
            self.density,
            self.logxi,
            self.logell,
            isotherm=self.isotherm,
        )
        iso.run()

        df = iso.isotherm_df
        x = df["SOC"]
        y = df["Potential"]

        ax.plot(x, y, **plt_kws)

        ax.set_xlabel("SoC")
        ax.set_ylabel("Potential / V")

        return ax
