#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of galpynostatic
#   https://github.com/fernandezfran/galpynostatic/
# Copyright (c) 2022-2023, Francisco Fernandez
# Copyright (c) 2024, Francisco Fernandez, Maximilano Gavilán, Andres Ruderman
# License: MIT
#   https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Module with Li-ion battery electrode materials parameters for simulation.

Here we provide average densities and specific capacities of typical
lithium-ion battery electrode materials for further simulations. Others
parameters needed, like isotherms, diffusion coefficients, particle size and
kinetic rate constants can be extracted from databases. For example, see the
LiionDB SQL database for battery parameters [6]_.

References
----------
.. [6] A. A. Wang, S. E. J. O’Kane, F. B. Planella, J. Le Houx, K. O’Regan, M.
   Zyskin, ... & J. M. Foster. "Review of parameterisation and a novel database
   (LiionDB) for continuum Li-ion battery models." `Progress in Energy` 4.3
   (2022): 032004.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import yaml

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CLASSES
# ============================================================================


class Electrode:
    """Electrode parameters.

    Parameters
    ----------
    material : str
        The materials name. Available materials include: Graphite"`,
        `"Graphite-Silicon"`, `"Silicon"`, `"LTO"`, `"LCO"`, `"LFP"`, `"LMO"`,
        `"NCA"`, `"NCO46"`, `"NMC"`, `"NMC111"`, `"NMC523"`, `"NMC622"`,
        `"NMC811"`.

    Raises
    ------
    ValueError
        When the material provided is not available in the dataset.
    """

    def __init__(self, material):
        with open(PATH / "params.yml", "r") as fparams:
            materials = yaml.load(fparams, Loader=yaml.FullLoader)

        try:
            self.material = materials[material]
        except KeyError:
            raise ValueError(
                f"params are not provided for {material} material."
            )

    @property
    def density(self):
        """Get the density of the material in :math:`g/cm^3`."""
        return self.material["density"]

    @property
    def specific_capacity(self):
        """Get the specific capacity of the material in :math:`mAhg^{-1}`."""
        return self.material["specific_capacity"]
