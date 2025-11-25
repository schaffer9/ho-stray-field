from ho_stray_field.base import TPELM, FunctionalTucker, fit
from ho_stray_field.tensor_grid import TensorGrid
from ho_stray_field.bspline import BSpline
from ho_stray_field.magnetostatic import (
    PotentialState, 
    fit_mag,
    superpotential,
    newtonpotential,
    scalar_potential,
    stray_field,
    energy,
    solve_energy
)

__all__ = (
    "TPELM",
    "FunctionalTucker",
    "fit",
    "TensorGrid",
    "BSpline",
    "PotentialState",
    "fit_mag",
    "superpotential",
    "newtonpotential",
    "scalar_potential",
    "stray_field",
    "energy",
    "solve_energy",
)