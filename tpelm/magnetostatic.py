from typing import Callable, Self
from functools import partial
from dataclasses import dataclass

from jax.tree_util import register_dataclass, register_pytree_node_class
from quadax.utils import QuadratureInfo

from .bspline import BSpline
from .tensor_grid import TensorGrid
from .functional_tucker import FunctionalTucker, fit, fit_divergence, fit_laplace, fit_grad
from .tucker_tensor import TuckerTensor, Core, Factors, tucker_dot
from .gs import superpotential, fit_superpotential, superpotential_factors, merge_quad_info, GS


from . import *
from .integrate import gauss


@register_dataclass
@dataclass
class FittedFT:
    tt: TuckerTensor
    elm: FunctionalTucker

    def __call__(self, x: jax.Array | TensorGrid) -> jax.Array:
        return self.elm(x, self.tt.core)


@register_dataclass
@dataclass
class StrayFieldSolver:
    elm: FunctionalTucker
    factors: Factors
    inv_factors: Factors
    partials: tuple[Factors, ...]

    mag_elm: FunctionalTucker
    mag_factors: Factors
    mag_inv_factors : Factors
    sp_factors: tuple[Factors, ...]
    gk_quad_info: QuadratureInfo
    
    gs: GS
    quad_grid: TensorGrid

    @classmethod
    def create(
        cls, 
        elm: FunctionalTucker, 
        mag_elm: FunctionalTucker | None = None,
        quad_grid: TensorGrid | None = None,
        gs_terms: int = 46,
        gs_coeff: float = 1.9,
        quad_order: int = 5,
        gk_epsabs: float = 1e-12,
        gk_epsrel: float = 0.0,
        gk_order: int = 31,
        gk_max_ninter: int = 150
    ) -> Self:
        
        if mag_elm is None:
            mag_elm = elm
        if quad_grid is None:
            tg1 = elm.domain
            tg2 = mag_elm.domain
            tg = TensorGrid(*(ti1 if (ti1.shape[0] > ti2.shape[0]) else ti2 for ti1, ti2 in zip(tg1, tg2)))
            quad_grid = tg.to_gauss(quad_order)

        assert mag_elm is not None
        gs = GS.from_sinc_1_over_sqrtx(gs_terms, gs_coeff)
        inv_factors = elm.pinv(quad_grid)
        factors, partials = elm.factors_and_partials(quad_grid)
        mag_factors = mag_elm.factors(quad_grid)
        mag_inv_factors = mag_elm.pinv(quad_grid)
        sp_factors, gk_quad_info = superpotential_factors(
            mag_elm, quad_grid, mag_elm.domain, 
            gs, epsabs=gk_epsabs, epsrel=gk_epsrel, order=gk_order, max_ninter=gk_max_ninter
        )
        solver = cls(
            elm=elm, 
            factors=factors, 
            inv_factors=inv_factors, 
            partials=partials,
            mag_elm=mag_elm, 
            mag_factors=mag_factors, 
            mag_inv_factors=mag_inv_factors, 
            sp_factors=sp_factors, 
            gk_quad_info=gk_quad_info, 
            gs=gs,
            quad_grid=quad_grid
        )
        return solver

    def fit_mag(self, m: jax.Array | TuckerTensor | Callable) -> "Magnetization":
        if callable(m):
            core = fit(self.mag_inv_factors, m, self.quad_grid)
        else:
            core = fit(self.mag_inv_factors, m)
        
        return Magnetization(TuckerTensor(core, self.mag_factors), self.mag_elm, self)


@register_dataclass
@dataclass
class Magnetization(FittedFT):
    solver: StrayFieldSolver

    @property
    def magnetization(self) -> FittedFT:
        return FittedFT(self.tt, self.solver.mag_elm)
    
    def superpotential(self) -> FittedFT:
        sp_core = fit_superpotential(self.solver.inv_factors, self.solver.sp_factors, self.tt.core, self.solver.gs)
        return FittedFT(TuckerTensor(sp_core, self.solver.factors), self.solver.elm)
    
    def scalar_potential(self) -> FittedFT:
        sp = self.superpotential()
        core = fit_divergence(self.solver.inv_factors, self.solver.partials, sp.tt.core)
        core = fit_laplace(self.solver.inv_factors, self.solver.partials, core)
        return FittedFT(TuckerTensor(core, sp.tt.factors), self.solver.elm)
    
    def stray_field(self) -> FittedFT:
        scalar_potential = self.scalar_potential()
        core = fit_grad(self.solver.inv_factors, self.solver.partials, scalar_potential.tt.core)
        return FittedFT(TuckerTensor(core, scalar_potential.tt.factors), self.solver.elm)
    
    def energy(self) -> jax.Array:
        h = self.stray_field().tt
        quad_weights = self.solver.quad_grid.weights
        factors = self.solver.mag_factors
        weighted_factors = tuple(w[:, *([None for _ in f.shape[1:]])] * f for w, f in zip(quad_weights, factors))
        m = TuckerTensor(self.tt.core, weighted_factors)
        return -1 / (16 * jnp.pi) * tucker_dot(h, m)
