from typing import Any, Callable, Self, NamedTuple, TypeAlias
from dataclasses import dataclass

from jax.tree_util import register_dataclass
from quadax.utils import QuadratureInfo

from .prelude import *
from . import gs
from .tensor_grid import TensorGrid
from .base import TPELM, FunctionalTucker, factors_pinv
from .tucker_tensor import TuckerTensor, Factors, tucker_dot

Potential: TypeAlias = FunctionalTucker
SuperPotential: TypeAlias = Potential
NewtonPotential: TypeAlias = Potential
ScalarPotential: TypeAlias = Potential
Magnetization: TypeAlias = FunctionalTucker
StrayField: TypeAlias = FunctionalTucker
F: TypeAlias = jax.Array | TuckerTensor | Callable[[jax.Array], jax.Array] | FunctionalTucker


class FTState(NamedTuple):
    """
    Stores data to approximate functions with a `TPELM` with the specified quadrature rule `quad_grid`.
    
    Attributes
    ----------
    elm: TPELM
    factors : Factors
        factor matrices corresponding to `quad_grid`
    inv_factors : Factors
        inverse factor matrices corresponding to `quad_grid`
    quad_grid : TensorGrid
        quadrature tensor grid
    """

    elm: TPELM
    factors: Factors
    inv_factors: Factors
    quad_grid: TensorGrid

    @classmethod
    def init(cls, elm: TPELM, quad_grid: TensorGrid | int, tol: float | tuple[float, ...] = 0.0) -> Self:
        """Initializes a `FTState`

        Parameters
        ----------
        elm : TPELM
        quad_grid : TensorGrid | int
        tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FTState
        """
        if isinstance(quad_grid, int):
            quad_grid = elm.domain.to_gauss(quad_grid)

        factors = elm.factors(quad_grid)
        inv_factors = factors_pinv(factors, weights=quad_grid.weights, tol=tol)
        return cls(elm, factors, inv_factors, quad_grid)

    def fit(self, f: F) -> FunctionalTucker:
        """Fit a function into a `FunctionalTucker`.

        Parameters
        ----------
        f : F

        Returns
        -------
        FunctionalTucker
        """
        return self.elm.fit(self.quad_grid, f, self.inv_factors)


class _PotentialState(NamedTuple):
    """Stores data for solving for a potential. This can either be 
    the superpotential or the Newton potential. `mag_state` is the 
    magnetization solver for a specific domain
    and `pot_state` is the solver state for the potential.
    
    Note that the domain of the potential and the magnetization can be different
    which is crucial for multiple domains.
    
    Attributes
    ----------
    pot_state : FTState
    mag_state : FTState
    pot_factors : tuple[Factors, ...]
        potential factors corresponding to targets in `target_quad_grid`
        w.r.t the domain described by `mag_state`.
    inv_factors : Factors
        inverse factors of the potential w.r.t `target_quad_grid` used
        to fit the potential
    target_quad_grid : TensorGrid
        targets at which the potential can be evaluated
    gs : GS
        Gaussian sum
    gk_quad_info : QuadratureInfo
        quadrature info for addaptive quadrature used for the potential
        integrals
    """

    pot_state: FTState
    mag_state: FTState

    pot_factors: tuple[Factors, ...]
    inv_factors: Factors
    target_quad_grid: TensorGrid

    gs: gs.GS
    gk_quad_info: QuadratureInfo

    @classmethod
    def init(
        cls,
        pot_elm: TPELM,
        target_quad_grid: TensorGrid,
        mag_elm: TPELM | None = None,
        pot_quad_grid: TensorGrid | None = None,
        mag_quad_grid: TensorGrid | None = None,
        gs_terms: int = 46,
        gs_coeff: float = 1.9,
        gk_epsabs: float = 1e-13,
        gk_epsrel: float = 1e-16,
        gk_order: int = 31,
        gk_max_ninter: int = 150,
        pot_pinv_tol: float | tuple[float, ...] = 0.0,
        mag_pinv_tol: float | tuple[float, ...] = 0.0,
        potential: str = "superpotential"
    ) -> Self:
        """Initializes the solver state `PotentialState` for the given potential.

        Parameters
        ----------
        pot_elm : TPELM
            potential TPELM
        target_quad_grid : TensorGrid
            targets where the potential is evaluated
        mag_elm : TPELM | None, optional
            magnetization TPELM; if `None` then `mag_elm=sp_elm` is used, by default None
        pot_quad_grid : TensorGrid, optional
            quadrature tensor grid for the potential; 
            defaults to `target_quad_grid`;
            by default None
        mag_quad_grid : TensorGrid | None, optional
            quadrature tensor grid for the magnetization; 
            defaults to `target_quad_grid`;
            by default None
        gs_terms : int, optional
            number of terms in the GS approximation, by default 46
        gs_coeff : float, optional
            sinc quadrature coefficient in the GS approximation, by default 1.9
        gk_epsabs : float, optional
            Gauss-Konrod adaptive quadrature tolerances, by default 1e-12
        gk_epsrel : float, optional
            Gauss-Konrod adaptive quadrature tolerances, by default 0.0
        gk_order : int, optional
            Gauss-Konrod adaptive quadrature order, by default 31
        gk_max_ninter : int, optional
            Gauss-Konrod adaptive quadrature maximum number of intervals, by default 150
        pot_pinv_tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0
        mag_pinv_tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0
        potential : str
            the potential which is solve for
        Returns
        -------
        SPState
        """
        if mag_elm is None:
            mag_elm = pot_elm

        assert mag_elm is not None
        if pot_quad_grid is None:
            pot_quad_grid = target_quad_grid
        
        if mag_quad_grid is None:
            mag_quad_grid = target_quad_grid

        _gs = gs.GS.from_sinc_1_over_sqrtx(gs_terms, gs_coeff)
        pot_state = FTState.init(pot_elm, pot_quad_grid, pot_pinv_tol)
        mag_state = FTState.init(mag_elm, pot_state.quad_grid if mag_quad_grid is None else mag_quad_grid, mag_pinv_tol)

        if potential.lower() in ("sp", "superpotential"):
            pot_factors, gk_quad_info = gs.superpotential_factors(
                mag_elm,
                target_quad_grid,
                mag_elm.domain,
                _gs,
                epsabs=gk_epsabs,
                epsrel=gk_epsrel,
                order=gk_order,
                max_ninter=gk_max_ninter,
            )
        elif potential.lower() in ("np", "newtonpotential"):
            pot_factors, gk_quad_info = gs.newtonpotential_factors(
                mag_elm,
                target_quad_grid,
                mag_elm.domain,
                _gs,
                epsabs=gk_epsabs,
                epsrel=gk_epsrel,
                order=gk_order,
                max_ninter=gk_max_ninter,
            )
        else:
            raise NotImplementedError(f"Potential {potential} is not implemented")
        
        inv_factors = factors_pinv(pot_elm.factors(target_quad_grid), weights=target_quad_grid.weights, tol=pot_pinv_tol)

        return cls(
            pot_state=pot_state,
            mag_state=mag_state,
            pot_factors=pot_factors,
            inv_factors=inv_factors,
            target_quad_grid=target_quad_grid,
            gs=_gs,
            gk_quad_info=gk_quad_info,
        )


@partial(register_dataclass, data_fields=["pot_state", "pot_domain_states", "mags", "quad_grids"], meta_fields=["potential"])
@dataclass
class PotentialState:
    """Stores data for solving for the required potential on one or multiple different domains.
    
    Attributes
    ----------
    potential : str
        the potential which is solve for
    pot_state : dict[int, FTState]
        mapping of domain index to potential solver state
    pot_domain_states : dict[int, dict[int, SPState]]
        for each domain index this attribute stores the potential solver state
        w.r.t each domain magnetization
    mags : dict[int, FTState]
        mapping of domain index to magnetization solver state
    quad_grids : dict[int, TensorGrid]
        mapping of domain index to target quadrature grid
    """
    potential: str
    pot_state: dict[int, FTState]
    pot_domain_states: dict[int, dict[int, _PotentialState]]
    mags: dict[int, FTState]
    quad_grids: dict[int, TensorGrid]

    @classmethod
    def init(
        cls,
        pot_elm: TPELM | dict[int, TPELM],
        target_quad_grid: TensorGrid | dict[int, TensorGrid],
        mag_elm: TPELM | dict[int, TPELM] | None = None,
        pot_quad_grid: TensorGrid | dict[int, TensorGrid] | None = None,
        mag_quad_grid: TensorGrid | dict[int, TensorGrid] | None = None,
        gs_terms: int = 100,
        gs_coeff: float = 1.9,
        gk_epsabs: float = 1e-13,
        gk_epsrel: float = 1e-16,
        gk_order: int = 31,
        gk_max_ninter: int = 150,
        pot_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
        mag_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
        potential: str = "superpotential"
    ) -> Self:
        """Initializes a multi-domain solver state.

        Parameters
        ----------
        pot_elm : TPELM | dict[int, TPELM]
            potential TPELM
        target_quad_grid : TensorGrid | dict[int, TensorGrid]
            targets where the potential is evaluated;
        mag_elm : TPELM | dict[int, TPELM] | None, optional
            magnetization TPELM; if `None` then `mag_elm=pot_elm` is used, by default None, by default None
        pot_quad_grid : TensorGrid | dict[int, TensorGrid], optional
            quadrature tensor grid for the potential;
            defaults to `target_quad_grid`;
            by default None
        mag_quad_grid : TensorGrid | dict[int, TensorGrid] | None, optional
            quadrature tensor grid for the magnetization; 
            defaults to `target_quad_grid`;
            by default None
        gs_terms : int, optional
            number of terms in the GS approximation, by default 46
        gs_coeff : float, optional
            sinc quadrature coefficient in the GS approximation, by default 1.9
        gk_epsabs : float, optional
            Gauss-Konrod adaptive quadrature tolerances, by default 1e-12
        gk_epsrel : float, optional
            Gauss-Konrod adaptive quadrature tolerances, by default 0.0
        gk_order : int, optional
            Gauss-Konrod adaptive quadrature order, by default 31
        gk_max_ninter : int, optional
            Gauss-Konrod adaptive quadrature maximum number of intervals, by default 150
        pot_pinv_tol : float | tuple[float, ...] | dict[int, float  |  tuple[float, ...]], optional
            cutoff tolerance for small singular values, by default 0.0
        mag_pinv_tol : float | tuple[float, ...] | dict[int, float  |  tuple[float, ...]], optional
            cutoff tolerance for small singular values, by default 0.0
        potential : str
            the potential which is solve for
        Returns
        -------
        DomainState
        """
        if potential.lower() in ("sp", "superpotential"):
            potential = "superpotential"
        elif potential.lower() in ("np", "newtonpotential"):
            potential = "newtonpotential"
        else:
            raise NotImplementedError
        
        if isinstance(pot_elm, TPELM):  # single domain case
            pot_elm = {0: pot_elm}
        
        if mag_elm is None:
            mag_elm = pot_elm
            
        target_quad_grid = _to_dict(target_quad_grid, pot_elm)
        if pot_quad_grid is None:
            pot_quad_grid = target_quad_grid
        if mag_quad_grid is None:
            mag_quad_grid = target_quad_grid
        
        mag_elm = _to_dict(mag_elm, pot_elm)
        pot_quad_grid = _to_dict(pot_quad_grid, pot_elm)
        mag_quad_grid = _to_dict(mag_quad_grid, mag_elm)
        pot_pinv_tol = _to_dict(pot_pinv_tol, pot_elm)
        mag_pinv_tol = _to_dict(mag_pinv_tol, mag_elm)

        # create potential FT for each domain
        pot = {i: FTState.init(pot_elm[i], pot_quad_grid[i], pot_pinv_tol[i]) for i in pot_elm.keys()}
        # create Magnetization FT for each domain
        mags = {i: FTState.init(mag_elm[i], mag_quad_grid[i], mag_pinv_tol[i]) for i in mag_elm.keys()}
        # create multi domain potential solvers
        pot_domain_states = {
            i: {
                k: _PotentialState.init(
                    pot_elm=pot_elm[i],
                    mag_elm=mag_elm[k],
                    pot_quad_grid=pot_quad_grid[i],
                    mag_quad_grid=mag_quad_grid[k],
                    target_quad_grid=target_quad_grid[i],
                    gs_terms=gs_terms,
                    gs_coeff=gs_coeff,
                    gk_epsabs=gk_epsabs,
                    gk_epsrel=gk_epsrel,
                    gk_order=gk_order,
                    gk_max_ninter=gk_max_ninter,
                    pot_pinv_tol=pot_pinv_tol[i],
                    mag_pinv_tol=mag_pinv_tol[k],
                    potential=potential
                )
                for k in mag_elm.keys()
            }
            for i in pot_elm.keys()
        }
        target_quad_grid = {i: pot_domain_states[i][0].target_quad_grid for i in pot_elm.keys()}
        return cls(
            potential=potential, 
            pot_state=pot, 
            pot_domain_states=pot_domain_states, 
            mags=mags, 
            quad_grids=target_quad_grid
        )


def _to_dict(v: Any | dict[int, Any], ref_dict: dict[int, Any]) -> dict[int, Any]:
    if not isinstance(v, dict):
        return {i: v for i in ref_dict.keys()}
    else:
        return v


RawMag: TypeAlias = F


def fit_mag(state: PotentialState, mag: RawMag | dict[int, RawMag]) -> dict[int, FunctionalTucker]:
    """Fits the magnetization for each domain to `FunctionalTucker` format.

    Parameters
    ----------
    state : DomainState
    mag : Mag | dict[int, Mag]

    Returns
    -------
    dict[int, FunctionalTucker]
    """
    if not isinstance(mag, dict):
        mag = {0: mag}
    
    return {i: state.mags[i].fit(mag[i]) for i in state.mags.keys()}


def superpotential(
    state: PotentialState, mags: Magnetization | dict[int, Magnetization]
) -> dict[int, SuperPotential]:
    """Computes the superpotential for the given magnetization.

    Parameters
    ----------
    state : DomainState
    mags : dict[int, Magnetization]
        in `FunctionalTucker` format

    Returns
    -------
    dict[int, Superpotential]
    """
    if not isinstance(mags, dict):
        mags = {0: mags}

    if state.potential != "superpotential":
        raise ValueError(
            "The superpotential can only be solved for with a solver for the superpotential. "
            "Set `potential='superpotential'"
        )
    
    def fit_sp(state: dict[int, _PotentialState], mags: dict[int, Magnetization]) -> SuperPotential:
        # the magnetization in each domain contributes to the superpotential in the current domain
        sp_cores = jnp.asarray([_superpotential(s, mags[i]).core for i, s in state.items()])
        core = jnp.sum(sp_cores, axis=0)  # superpotential is the sum of all contributions
        return FunctionalTucker(core, state[0].pot_state.elm)

    superpotentials = {i: fit_sp(s, mags) for i, s in state.pot_domain_states.items()}
    return superpotentials


def newtonpotential(
    state: PotentialState, mags: Magnetization | dict[int, Magnetization]
) -> dict[int, SuperPotential]:
    """Computes the newtonpotential for the given magnetization.

    Parameters
    ----------
    state : DomainState
    mags : dict[int, Magnetization]
        in `FunctionalTucker` format

    Returns
    -------
    dict[int, NewtonPotential]
    """
    if not isinstance(mags, dict):
        mags = {0: mags}
    if state.potential != "newtonpotential":
        raise ValueError(
            "The newtonpotential can only be solved for with a solver for the newtonpotential. "
            "Set `potential='newtonpotential'"
        )
    
    def fit_np(state: dict[int, _PotentialState], mags: dict[int, Magnetization]) -> SuperPotential:
        # the magnetization in each domain contributes to the superpotential in the current domain
        sp_cores = jnp.asarray([_newtonpotential(s, mags[i]).core for i, s in state.items()])
        core = jnp.sum(sp_cores, axis=0)  # superpotential is the sum of all contributions
        return FunctionalTucker(core, state[0].pot_state.elm)

    superpotentials = {i: fit_np(s, mags) for i, s in state.pot_domain_states.items()}
    return superpotentials


def scalar_potential(state: PotentialState, pot: dict[int, Potential]) -> dict[int, ScalarPotential]:
    """Computes the scalar potential from the given superpotential.

    Parameters
    ----------
    state : DomainState
    sp : dict[int, Superpotential]
        in `FunctionalTucker` format

    Returns
    -------
    dict[int, ScalarPotential]
    """
    if state.potential == "superpotential":
        return {i: _scalar_potential_from_sp(s, pot[i]) for i, s in state.pot_state.items()}
    elif state.potential == "newtonpotential":
        return {i: _scalar_potential_from_np(s, pot[i]) for i, s in state.pot_state.items()}
    else:
        raise NotImplementedError


def stray_field(state: PotentialState, scalar_pot: dict[int, ScalarPotential]) -> dict[int, StrayField]:
    """Computes the stray field from the given superpotential.

    Parameters
    ----------
    state : DomainState
    sp : dict[int, Superpotential]
        in `FunctionalTucker` format

    Returns
    -------
    dict[int, ScalarPotential]
    """
    return {i: _stray_field(s, scalar_pot[i]) for i, s in state.pot_state.items()}


def energy(h: dict[int, StrayField], m: dict[int, Magnetization], quad_grids: TensorGrid | dict[int, TensorGrid]) -> jax.Array:
    """Computes the magnetostatic energy.

    Parameters
    ----------
    h : dict[int, StrayField]
        stray field in `FunctionalTucker` format
    m : dict[int, Magnetization]
        magnetization in `FunctionalTucker` format
    quad_grids : dict[int, TensorGrid]
        quadrature tensor grid

    Returns
    -------
    jax.Array
        Energy
    """
    if not isinstance(quad_grids, dict):
        quad_grids = {0: quad_grids}
    assert h.keys() == m.keys() == quad_grids.keys(), "`h` and `m` must describe same domains"
    return jnp.sum(jnp.asarray([_energy(h[i], m[i], quad_grids[i]) for i in h.keys()]))


def solve_energy(
    state: PotentialState, 
    m: RawMag | dict[int, RawMag], 
    quad_grid: TensorGrid | dict[int, TensorGrid] | None = None
) -> jax.Array:
    """Solves for the magnetostatic energy for the provided magnetization.

    Parameters
    ----------
    state : DomainState
    m : RawMag | dict[int, RawMag]
        magnetization
    quad_grid : TensorGrid | dict[int, TensorGrid] | None
        quadrature grid, defaults to None
    Returns
    -------
    jax.Array
        Energy
    """
    if quad_grid is None:
        quad_grid = state.quad_grids
    
    if not isinstance(quad_grid, dict):
        quad_grid = {0: quad_grid}
        
    mag = fit_mag(state, m)
    if state.potential == "superpotential":
        pot = superpotential(state, mag)
    elif state.potential == "newtonpotential":
        pot = newtonpotential(state, mag)
    else:
        raise NotImplementedError
    scalar_pot = scalar_potential(state, pot)
    h = stray_field(state, scalar_pot)
    return energy(h, mag, quad_grid)


def _superpotential(state: _PotentialState, m: FunctionalTucker) -> SuperPotential:
    sp_core = 1 / (8 * jnp.pi) * gs.fit_superpotential(state.inv_factors, state.pot_factors, m.core, state.gs)
    return SuperPotential(sp_core, state.pot_state.elm)


def _newtonpotential(state: _PotentialState, m: FunctionalTucker) -> NewtonPotential:
    np_core = -1 / (4 * jnp.pi) * gs.fit_newtonpotential(state.inv_factors, state.pot_factors, m.core, state.gs)
    return SuperPotential(np_core, state.pot_state.elm)


def _scalar_potential_from_sp(state: FTState, pot: SuperPotential) -> ScalarPotential:
    div_sp = pot.divergence(
        state.quad_grid,
        state.inv_factors,
    )
    core = -div_sp.laplace(state.quad_grid, state.inv_factors).core
    return ScalarPotential(core, pot.elm)


def _scalar_potential_from_np(state: FTState, pot: NewtonPotential) -> ScalarPotential:
    return pot.divergence(
        state.quad_grid,
        state.inv_factors,
    )


def _stray_field(state: FTState, scalar_pot: ScalarPotential) -> StrayField:
    core = -scalar_pot.grad(state.quad_grid, state.inv_factors).core
    return StrayField(core, scalar_pot.elm)


def _energy(h: StrayField, m: Magnetization, quad_grid: TensorGrid) -> jax.Array:
    H = h.tt(quad_grid, mul_weights=True)
    M = m.tt(quad_grid)
    return -1 / 2 * tucker_dot(H, M)
