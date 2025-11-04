from typing import Any, Callable, Self, Sequence, NamedTuple, TypeAlias
import itertools
from dataclasses import dataclass

from jax.tree_util import register_dataclass, register_pytree_node_class
from quadax.utils import QuadratureInfo

from . import gs
from .bspline import BSpline
from .tensor_grid import TensorGrid
from .base import TPELM, FunctionalTucker, fit, fit_divergence, fit_laplace, fit_grad, factors_pinv
from .tucker_tensor import TuckerTensor, Core, Factors, tucker_dot
#from .gs import superpotential as , fit_superpotential, superpotential_factors, merge_quad_info, GS

from . import *
from .integrate import gauss


Superpotential: TypeAlias = FunctionalTucker
Magnetization: TypeAlias = FunctionalTucker
ScalarPotential: TypeAlias = FunctionalTucker
StrayField: TypeAlias = FunctionalTucker


class FTState(NamedTuple):
    """Stores data to approximate functions with a `TPELM` with the specified quadrature rule `quad_grid`.
    """
    elm: TPELM
    factors: Factors
    inv_factors: Factors
    quad_grid: TensorGrid

    @classmethod
    def init(cls, elm: TPELM, quad_grid: TensorGrid | int, tol: float | tuple[float, ...] = 0.0) -> Self:
        if isinstance(quad_grid, int):
            quad_grid = elm.domain.to_gauss(quad_grid)

        factors = elm.factors(quad_grid)
        inv_factors = factors_pinv(factors, weights=quad_grid.weights, tol=tol)
        return cls(elm, factors, inv_factors, quad_grid)


class SPState(NamedTuple):
    """Stores data for solving for the superpotential. `mag_state` is the magnetization solver for a specific domain
    and `sp_state` can be the solver state for a different domain.
    """
    sp_state: FTState
    mag_state: FTState

    sp_factors: tuple[Factors, ...]
    inv_factors: Factors
    target_quad_grid: TensorGrid
    
    gs: gs.GS
    gk_quad_info: QuadratureInfo

    @classmethod
    def init(
        cls, 
        sp_elm: TPELM, 
        mag_elm: TPELM | None = None,
        sp_quad_grid: TensorGrid | int = 5,
        mag_quad_grid: TensorGrid | int | None = None,
        target_quad_grid: TensorGrid | int | None = None,
        gs_terms: int = 46,
        gs_coeff: float = 1.9,
        gk_epsabs: float = 1e-12,
        gk_epsrel: float = 0.0,
        gk_order: int = 31,
        gk_max_ninter: int = 150,
        sp_pinv_tol: float | tuple[float, ...] = 0.0,
        mag_pinv_tol: float | tuple[float, ...] = 0.0,
    ) -> Self:
        
        if mag_elm is None:
            mag_elm = sp_elm

        assert mag_elm is not None

        _gs = gs.GS.from_sinc_1_over_sqrtx(gs_terms, gs_coeff)
        sp_state = FTState.init(sp_elm, sp_quad_grid, sp_pinv_tol)
        mag_state = FTState.init(mag_elm, sp_state.quad_grid if mag_quad_grid is None else mag_quad_grid, mag_pinv_tol)
        
        if target_quad_grid is None:
            target_quad_grid = sp_state.quad_grid
        elif isinstance(target_quad_grid, int):
            target_quad_grid = sp_elm.domain.to_gauss(target_quad_grid)

        sp_factors, gk_quad_info = gs.superpotential_factors(
            mag_elm, target_quad_grid, mag_elm.domain, 
            _gs, epsabs=gk_epsabs, epsrel=gk_epsrel, order=gk_order, max_ninter=gk_max_ninter
        )
        inv_factors = factors_pinv(sp_elm.factors(target_quad_grid), weights=target_quad_grid.weights, tol=sp_pinv_tol)

        return cls(
            sp_state=sp_state,
            mag_state=mag_state,
            sp_factors=sp_factors,
            inv_factors=inv_factors,
            target_quad_grid=target_quad_grid,
            gs=_gs,
            gk_quad_info= gk_quad_info
        )
    

class DomainState(NamedTuple):
    """Stores data to solve superpotential and magnetostatic fields on multiple domains.
    """
    sp: dict[int, FTState]
    sp_domain_states: dict[int, dict[int, SPState]]
    mags: dict[int, FTState]
    quad_grids: dict[int, TensorGrid]

    @classmethod
    def init(
        cls,
        sp_elm: TPELM | dict[int, TPELM], 
        mag_elm: TPELM | dict[int, TPELM] | None = None,
        sp_quad_grid: TensorGrid | dict[int, TensorGrid] | int = 5,
        mag_quad_grid: TensorGrid | dict[int, TensorGrid] | None = None,
        target_quad_grid: TensorGrid | dict[int, TensorGrid] | None = None,
        gs_terms: int = 46,
        gs_coeff: float = 1.9,
        gk_epsabs: float = 1e-12,
        gk_epsrel: float = 0.0,
        gk_order: int = 31,
        gk_max_ninter: int = 150,
        sp_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
        mag_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
    ) -> Self:
        if isinstance(sp_elm, TPELM):  # single somain case
            sp_elm = {0: sp_elm}

        mag_elm = _to_dict(mag_elm, sp_elm)
        sp_quad_grid = _to_dict(sp_quad_grid, sp_elm)
        mag_quad_grid = _to_dict(mag_quad_grid, mag_elm)
        target_quad_grid = _to_dict(target_quad_grid, sp_elm)
        sp_pinv_tol = _to_dict(sp_pinv_tol, sp_elm)
        mag_pinv_tol = _to_dict(mag_pinv_tol, mag_elm)

        sp = {i: FTState.init(sp_elm[i], sp_quad_grid[i], sp_pinv_tol[i]) for i in sp_elm.keys()}
        mags = {i: FTState.init(mag_elm[i], mag_quad_grid[i], mag_pinv_tol[i]) for i in mag_elm.keys()}
        sp_domain_states = {
            i: {k: SPState.init(
                sp_elm[i], 
                mag_elm[k], 
                sp_quad_grid[i], 
                mag_quad_grid[k],
                target_quad_grid[i],
                gs_terms,
                gs_coeff,
                gk_epsabs,
                gk_epsrel,
                gk_order,
                gk_max_ninter,
                sp_pinv_tol[i],
                mag_pinv_tol[k]
                ) 
                for k in mag_elm.keys()} for i in sp_elm.keys()}
        target_quad_grid = {i: sp_domain_states[i][0].target_quad_grid for i in sp_elm.keys()}
        return cls(
            sp=sp, sp_domain_states=sp_domain_states, mags=mags, quad_grids=target_quad_grid
        )
    

def _to_dict(v: Any | dict[int, Any], ref_dict: dict[int, Any]) -> dict[int, Any]:
    if not isinstance(v, dict):
        return {i: v for i in ref_dict.keys()}
    else:
        return v


def superpotential(
        state: DomainState, 
        m: dict[int, jax.Array | TuckerTensor | Callable | FunctionalTucker]
    ) -> tuple[dict[int, Superpotential], dict[int, Magnetization]]:
    def _fit_mag(state: FTState, mag) -> Magnetization:
        return state.elm.fit(state.quad_grid, mag, state.inv_factors)
    
    mags = {i: _fit_mag(state.mags[i], m[i]) for i in state.sp.keys()}  # fit the magnetization for each domain

    def fit_sp(state: dict[int, SPState], mags: dict[int, Magnetization]) -> Superpotential:
        # the magnetization in each domain contributes to the superpotential in the current domain
        sp_cores = jnp.asarray([_superpotential(s, mags[i]).core for i, s in state.items()])
        core = jnp.sum(sp_cores, axis=0)  # superpotential is the sum of all contributions
        return FunctionalTucker(core, state[0].sp_state.elm)

    superpotentials = {i: fit_sp(s, mags) for i, s in state.sp_domain_states.items()}
    return superpotentials, mags


def scalar_potential(state: DomainState, sp: dict[int, Superpotential]) -> dict[int, ScalarPotential]:
    return {i: _scalar_potential(s, sp[i]) for i, s in state.sp.items()}


def stray_field(state: DomainState, sp: dict[int, Superpotential]) -> dict[int, StrayField]:
    return {i: _stray_field(s, sp[i]) for i, s in state.sp.items()}


def energy(h: dict[int, StrayField], m: dict[int, Magnetization], quad_grids: dict[int, TensorGrid]) -> jax.Array:
    assert h.keys() == m.keys() == quad_grids.keys(), "`h` and `m` must describe same domains"
    return jnp.sum(jnp.asarray([_energy(h[i], m[i], quad_grids[i]) for i in h.keys()]))


def solve_energy(state: DomainState, m: dict[int, jax.Array | TuckerTensor | Callable | FunctionalTucker]):
    sp, mag = superpotential(state, m)
    h = stray_field(state, sp)
    return energy(h, mag, state.quad_grids)


def _superpotential(state: SPState, m: FunctionalTucker) -> Superpotential:
    sp_core = 1 / (8 * jnp.pi) * gs.fit_superpotential(state.inv_factors, state.sp_factors, m.core, state.gs)
    return Superpotential(sp_core, state.sp_state.elm)


def _scalar_potential(
    state: FTState, 
    sp: Superpotential
) -> Superpotential:
    div_sp = sp.divergence(state.quad_grid, state.inv_factors, )
    return div_sp.laplace(state.quad_grid, state.inv_factors)


def _stray_field(
    state: FTState, 
    sp: Superpotential
) -> StrayField:
    scalar_pot = _scalar_potential(state, sp)
    return scalar_pot.grad(state.quad_grid, state.inv_factors)


def _energy(h: StrayField, m: Magnetization, quad_grid: TensorGrid) -> jax.Array:
    H = h.tt(quad_grid, mul_weights=True)
    M = m.tt(quad_grid)
    return -1 / 2 * tucker_dot(H, M)


# @register_dataclass
# @dataclass
# class SuperPotentialSolver:
#     elm: TPELM
#     inv_factors: Factors

#     mag_elm: TPELM
#     mag_inv_factors : Factors
#     sp_factors: tuple[Factors, ...]
#     gk_quad_info: QuadratureInfo
    
#     gs: GS
#     quad_grid: TensorGrid
#     mag_quad_grid: TensorGrid

#     @classmethod
#     def create(
#         cls, 
#         elm: TPELM, 
#         mag_elm: TPELM | None = None,
#         quad_grid: TensorGrid | None = None,
#         mag_quad_grid: TensorGrid | None = None,
#         gs_terms: int = 46,
#         gs_coeff: float = 1.9,
#         quad_order: int = 5,
#         gk_epsabs: float = 1e-12,
#         gk_epsrel: float = 0.0,
#         gk_order: int = 31,
#         gk_max_ninter: int = 150,
#         pinv_tol: float = 0.0,
#         mag_pinv_tol: float = 0.0,
#     ) -> Self:
        
#         if mag_elm is None:
#             mag_elm = elm
#         if quad_grid is None:
#             quad_grid = elm.domain.to_gauss(quad_order)
#         if mag_quad_grid is None:
#             mag_quad_grid = mag_elm.domain.to_gauss(quad_order)

#         assert mag_elm is not None
#         gs = GS.from_sinc_1_over_sqrtx(gs_terms, gs_coeff)
#         factors = elm.factors(quad_grid)
#         inv_factors = factors_pinv(factors, quad_grid.weights, tol=pinv_tol)
#         mag_factors = mag_elm.factors(mag_quad_grid)
#         mag_inv_factors = factors_pinv(mag_factors, mag_quad_grid.weights, tol=mag_pinv_tol)
#         sp_factors, gk_quad_info = superpotential_factors(
#             mag_elm, quad_grid, mag_elm.domain, 
#             gs, epsabs=gk_epsabs, epsrel=gk_epsrel, order=gk_order, max_ninter=gk_max_ninter
#         )
#         solver = cls(
#             elm=elm, 
#             inv_factors=inv_factors, 
#             mag_elm=mag_elm, 
#             mag_inv_factors=mag_inv_factors, 
#             sp_factors=sp_factors, 
#             gk_quad_info=gk_quad_info, 
#             gs=gs,
#             quad_grid=quad_grid,
#             mag_quad_grid=mag_quad_grid,
#         )
#         return solver

#     def solve(self, m: jax.Array | TuckerTensor | Callable) -> "Superpotential":
#         mag = self.mag_elm.fit(self.mag_quad_grid, m, inv_factors=self.inv_factors)
#         sp_core = fit_superpotential(self.inv_factors, self.sp_factors, mag.core, self.gs)
#         return Superpotential(sp_core, self.elm)


# @register_dataclass
# @dataclass
# class Superpotential(FunctionalTucker):
#     def scalar_potential(
#         self, 
#         tg: TensorGrid, 
#         inv_factors: Factors | None = None, 
#         tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
#     ) -> FunctionalTucker:
#         divergence = self.divergence(tg, inv_factors, tol)
#         return divergence.laplace(tg, inv_factors, tol)
        
#     def stray_field(
#         self, 
#         tg: TensorGrid, 
#         inv_factors: Factors | None = None, 
#         tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
#     ) -> FunctionalTucker:
#         scalar_potential = self.scalar_potential(tg, inv_factors, tol)
#         return scalar_potential.grad(tg, inv_factors, tol)
     
#     def energy(self) -> jax.Array:
#         h = self.stray_field().tt
#         quad_weights = self.solver.quad_grid.weights
#         factors = self.solver.mag_factors
#         weighted_factors = tuple(w[:, *([None for _ in f.shape[1:]])] * f for w, f in zip(quad_weights, factors))
#         m = TuckerTensor(self.tt.core, weighted_factors)
#         return -1 / (16 * jnp.pi) * tucker_dot(h, m)

