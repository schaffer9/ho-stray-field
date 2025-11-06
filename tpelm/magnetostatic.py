from typing import Any, Callable, Self, NamedTuple, TypeAlias

from quadax.utils import QuadratureInfo

from . import *
from . import gs
from .tensor_grid import TensorGrid
from .base import TPELM, FunctionalTucker, factors_pinv
from .tucker_tensor import TuckerTensor, Factors, tucker_dot


Superpotential: TypeAlias = FunctionalTucker
Magnetization: TypeAlias = FunctionalTucker
ScalarPotential: TypeAlias = FunctionalTucker
StrayField: TypeAlias = FunctionalTucker
F = jax.Array | TuckerTensor | Callable | FunctionalTucker


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


class SPState(NamedTuple):
    """Stores data for solving for the superpotential. `mag_state` is the magnetization solver for a specific domain
    and `sp_state` is the solver state for the superpotential.
    
    Note that the domain of the superpotential and the magnetization can be different
    which is crucial for multiple domains.
    
    Attributes
    ----------
    sp_state : FTState
    mag_state : FTState
    sp_factors : tuple[Factors, ...]
        superpotential factors corresponding to targets in `target_quad_grid`
        w.r.t the domain described by `mag_state`.
    inv_factors : Factors
        inverse factors of the superpotential w.r.t `target_quad_grid` used
        to fit the superpotential
    target_quad_grid : TensorGrid
        targets at which the superpotential can be evaluated
    gs : GS
        Gaussian sum
    gk_quad_info : QuadratureInfo
        quadrature info for addaptive quadrature used for the superpotential
        integrals
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
        """Initializes the solver state `SPState` for the superpotential.

        Parameters
        ----------
        sp_elm : TPELM
            superpotential TPELM
        mag_elm : TPELM | None, optional
            magnetization TPELM; if `None` then `mag_elm=sp_elm` is used, by default None
        sp_quad_grid : TensorGrid | int, optional
            quadrature tensor grid for the superpotential; 
            integer value creates a Gauss-Legendre rule based on the domain of `sp_elm`,
            by default 5
        mag_quad_grid : TensorGrid | int | None, optional
            quadrature tensor grid for the magnetization; 
            integer value creates a Gauss-Legendre rule based on the domain of `mag_elm`;
            if `None` then `sp_quad_grid` is used,
            by default 5, by default None
        target_quad_grid : TensorGrid | int | None, optional
            targets at which the superpotential can be evaluated;
            if `None` then `sp_quad_grid` is used;
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
        sp_pinv_tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0
        mag_pinv_tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        SPState
        """
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
            mag_elm,
            target_quad_grid,
            mag_elm.domain,
            _gs,
            epsabs=gk_epsabs,
            epsrel=gk_epsrel,
            order=gk_order,
            max_ninter=gk_max_ninter,
        )
        inv_factors = factors_pinv(sp_elm.factors(target_quad_grid), weights=target_quad_grid.weights, tol=sp_pinv_tol)

        return cls(
            sp_state=sp_state,
            mag_state=mag_state,
            sp_factors=sp_factors,
            inv_factors=inv_factors,
            target_quad_grid=target_quad_grid,
            gs=_gs,
            gk_quad_info=gk_quad_info,
        )


class DomainState(NamedTuple):
    """Stores data for solving for the superpotential on one or multiple different domains.
    
    Attributes
    ----------
    sp_state : dict[int, FTState]
        mapping of domain index to superpotential solver state
    sp_domain_states : dict[int, dict[int, SPState]]
        for each domain index this attribute stores the superpotential solver state
        w.r.t each domain magnetization
    mags : dict[int, FTState]
        mapping of domain index to magnetization solver state
    quad_grids : dict[int, TensorGrid]
        mapping of domain index to target quadrature grid
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
        mag_quad_grid: TensorGrid | dict[int, TensorGrid] | int | None = None,
        target_quad_grid: TensorGrid | dict[int, TensorGrid] | None = None,
        gs_terms: int = 46,
        gs_coeff: float = 1.9,
        gk_epsabs: float = 1e-14,
        gk_epsrel: float = 0.0,
        gk_order: int = 31,
        gk_max_ninter: int = 150,
        sp_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
        mag_pinv_tol: float | tuple[float, ...] | dict[int, float | tuple[float, ...]] = 0.0,
    ) -> Self:
        """Initializes a multi-domain solver state.

        Parameters
        ----------
        sp_elm : TPELM | dict[int, TPELM]
            superpotential TPELM
        mag_elm : TPELM | dict[int, TPELM] | None, optional
            magnetization TPELM; if `None` then `mag_elm=sp_elm` is used, by default None, by default None
        sp_quad_grid : TensorGrid | dict[int, TensorGrid] | int, optional
            quadrature tensor grid for the superpotential; 
            integer value creates a Gauss-Legendre rule based on the domain of `sp_elm`,
            by default 5, by default 5
        mag_quad_grid : TensorGrid | dict[int, TensorGrid] | int | None, optional
            quadrature tensor grid for the magnetization; 
            integer value creates a Gauss-Legendre rule based on the domain of `mag_elm`;
            if `None` then `sp_quad_grid` is used,
            by default 5, by default None
        target_quad_grid : TensorGrid | dict[int, TensorGrid] | None, optional
            targets at which the superpotential can be evaluated;
            if `None` then `sp_quad_grid` is used;
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
        sp_pinv_tol : float | tuple[float, ...] | dict[int, float  |  tuple[float, ...]], optional
            cutoff tolerance for small singular values, by default 0.0
        mag_pinv_tol : float | tuple[float, ...] | dict[int, float  |  tuple[float, ...]], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        DomainState
        """
        if isinstance(sp_elm, TPELM):  # single domain case
            sp_elm = {0: sp_elm}
        
        if mag_elm is None:
            mag_elm = sp_elm
            
        if mag_quad_grid is None:
            mag_quad_grid = sp_quad_grid
        
        mag_elm = _to_dict(mag_elm, sp_elm)
        sp_quad_grid = _to_dict(sp_quad_grid, sp_elm)
        mag_quad_grid = _to_dict(mag_quad_grid, mag_elm)
        target_quad_grid = _to_dict(target_quad_grid, sp_elm)
        sp_pinv_tol = _to_dict(sp_pinv_tol, sp_elm)
        mag_pinv_tol = _to_dict(mag_pinv_tol, mag_elm)

        sp = {i: FTState.init(sp_elm[i], sp_quad_grid[i], sp_pinv_tol[i]) for i in sp_elm.keys()}
        mags = {i: FTState.init(mag_elm[i], mag_quad_grid[i], mag_pinv_tol[i]) for i in mag_elm.keys()}
        sp_domain_states = {
            i: {
                k: SPState.init(
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
                    mag_pinv_tol[k],
                )
                for k in mag_elm.keys()
            }
            for i in sp_elm.keys()
        }
        target_quad_grid = {i: sp_domain_states[i][0].target_quad_grid for i in sp_elm.keys()}
        return cls(sp=sp, sp_domain_states=sp_domain_states, mags=mags, quad_grids=target_quad_grid)


def _to_dict(v: Any | dict[int, Any], ref_dict: dict[int, Any]) -> dict[int, Any]:
    if not isinstance(v, dict):
        return {i: v for i in ref_dict.keys()}
    else:
        return v


RawMag: TypeAlias = F


def fit_mag(state: DomainState, mag: RawMag | dict[int, RawMag]) -> dict[int, FunctionalTucker]:
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
    state: DomainState, mags: dict[int, Magnetization]
) -> dict[int, Superpotential]:
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
    
    def fit_sp(state: dict[int, SPState], mags: dict[int, Magnetization]) -> Superpotential:
        # the magnetization in each domain contributes to the superpotential in the current domain
        sp_cores = jnp.asarray([_superpotential(s, mags[i]).core for i, s in state.items()])
        core = jnp.sum(sp_cores, axis=0)  # superpotential is the sum of all contributions
        return FunctionalTucker(core, state[0].sp_state.elm)

    superpotentials = {i: fit_sp(s, mags) for i, s in state.sp_domain_states.items()}
    return superpotentials


def scalar_potential(state: DomainState, sp: dict[int, Superpotential]) -> dict[int, ScalarPotential]:
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
    return {i: _scalar_potential(s, sp[i]) for i, s in state.sp.items()}


def stray_field(state: DomainState, sp: dict[int, Superpotential]) -> dict[int, StrayField]:
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
    return {i: _stray_field(s, sp[i]) for i, s in state.sp.items()}


def energy(h: dict[int, StrayField], m: dict[int, Magnetization], quad_grids: dict[int, TensorGrid]) -> jax.Array:
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
    assert h.keys() == m.keys() == quad_grids.keys(), "`h` and `m` must describe same domains"
    return jnp.sum(jnp.asarray([_energy(h[i], m[i], quad_grids[i]) for i in h.keys()]))


def solve_energy(state: DomainState, m: RawMag | dict[int, RawMag]) -> jax.Array:
    """Solves for the magnetostatic energy for the provided magnetization.

    Parameters
    ----------
    state : DomainState
    m : RawMag | dict[int, RawMag]
        magnetization

    Returns
    -------
    jax.Array
        Energy
    """
    mag = fit_mag(state, m)
    sp = superpotential(state, mag)
    h = stray_field(state, sp)
    return energy(h, mag, state.quad_grids)


def _superpotential(state: SPState, m: FunctionalTucker) -> Superpotential:
    sp_core = 1 / (8 * jnp.pi) * gs.fit_superpotential(state.inv_factors, state.sp_factors, m.core, state.gs)
    return Superpotential(sp_core, state.sp_state.elm)


def _scalar_potential(state: FTState, sp: Superpotential) -> Superpotential:
    div_sp = sp.divergence(
        state.quad_grid,
        state.inv_factors,
    )
    return div_sp.laplace(state.quad_grid, state.inv_factors)


def _stray_field(state: FTState, sp: Superpotential) -> StrayField:
    scalar_pot = _scalar_potential(state, sp)
    return scalar_pot.grad(state.quad_grid, state.inv_factors)


def _energy(h: StrayField, m: Magnetization, quad_grid: TensorGrid) -> jax.Array:
    H = h.tt(quad_grid, mul_weights=True)
    M = m.tt(quad_grid)
    return -1 / 2 * tucker_dot(H, M)
