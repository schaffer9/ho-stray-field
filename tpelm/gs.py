from typing import NamedTuple, Self

from quadax import quadgk, quadts, quadcc
from quadax.utils import QuadratureInfo

from . import *
from .base import TPELM, fit
from .bspline import BSpline
from .tensor_grid import TensorGrid
from .integrate import sinc_quad_1_over_sqrtx
from .tucker_tensor import TuckerTensor, Factors, Core


class GS(NamedTuple):
    """Gaussian sum approximation

    Attributes
    ----------
    omega : jax.Array
    alpha : jax.Array

    """
    omega: jax.Array
    alpha: jax.Array

    @classmethod
    def from_sinc_1_over_sqrtx(cls, rank: int, c0: float = 1.9) -> Self:
        r"""Creates a Gaussian sum approximation for :math:`1/\sqrt{x}` with
        sinc quadrature.

        Parameters
        ----------
        rank : int
            number of terms in the Gaussian sum
        c0 : float, optional
            sinc quadrature coefficient, by default 1.9

        Returns
        -------
        GS
        """
        return cls(*sinc_quad_1_over_sqrtx(rank, c0))


def _quad(g, x, interval, alpha, stds: int = 4, **kwargs):
    s = jnp.sqrt(1 / (2 * jnp.maximum(alpha, 1.0)))
    lb, ub = jnp.min(interval), jnp.max(interval)
    gaussian_interval = jnp.linspace(-s * stds, s * stds, 2 * stds + 1) + x
    _interval = jnp.sort(jnp.concatenate([interval, gaussian_interval]))
    _interval = jnp.clip(_interval, lb, ub)
    return quadcc(g, _interval, **kwargs)


def integrate_gs_term(basis1d, x, interval, alpha, stds: int = 4, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return jnp.exp(-alpha * r) * basis1d(y)
    
    return _quad(g, x, interval, alpha, stds=stds, **kwargs)


def integrate_r2_gs_term(basis1d, x, interval, alpha, stds: int = 4, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return r * jnp.exp(-alpha * r) * basis1d(y)
    
    return _quad(g, x, interval, alpha, stds=stds, **kwargs)


def superpotential_factors(
    elm: TPELM,
    target_tg: TensorGrid,
    quad_tg: TensorGrid,
    gs: GS,
    batch_size: int | None = None,
    max_ninter=50,
    **kwargs
) -> tuple[tuple[Factors, ...], QuadratureInfo]:
    """Computes the factor matrices for :math:`|x|` for all `alpha` in the 
    Gaussian sum approximation `gs` for all targets on the provided tensor grid.
    `quad_tg` specifies the integration domain with optional breakpoints for the 
    addaptive quadrature.

    Parameters
    ----------
    elm : TPELM
    target_tg : TensorGrid
        targets at which the integrals are evaluated
    quad_tg : TensorGrid
        integration domain with optional breakpoints
    gs : GS
    batch_size : int | None, optional
        if provided performs serial computation with this batch size, by default None
    max_ninter : int
        maximum number of intervals for the adaptive quadrature. Note that this is added 
        to the number of intervals within `quad_tg`.
    kwargs : Any
        further kwargs for the adaptive quadrature

    Returns
    -------
    tuple[tuple[Factors, ...], QuadratureInfo]
    """
    modes = list(range(quad_tg.dim))

    def _integrate(f, alpha: jax.Array, mode: int):
        def integrate_target(target):
            def b(y):
                return elm.basis(y, mode=mode)
            interval = quad_tg[mode]
            _max_ninter = interval.shape[0] + max_ninter
            I, info = f(b, target, interval, alpha, max_ninter=_max_ninter, **kwargs)
            
            return I, info
        
        targets = target_tg[mode]
        if batch_size is None:
            return jax.vmap(integrate_target)(targets)
        else:
            return jax.lax.map(integrate_target, targets, batch_size=batch_size)
    
    I_gs, info1 = zip(*(lax.map(lambda a: _integrate(integrate_gs_term, a, mode), gs.alpha) for mode in modes))
    I_r2_gs, info2 = zip(*(lax.map(lambda a: _integrate(integrate_r2_gs_term, a, mode), gs.alpha) for mode in modes))
    factors = tuple(
        I_gs[:mode] + (I_r2_gs[mode],) + I_gs[mode + 1:]
        for mode in modes
    )
    return factors, merge_quad_info(*info1, *info2)


def fit_superpotential(
    inv_factors: Factors, factors_superpot: tuple[Factors, ...], core: Core, gs: GS
) -> Core:
    """Computes the core tensor corresponding the the inverse factors `inv_factors`
    for the given GS approximation of the superpotential given by `factors_superpot` and `core`.

    Parameters
    ----------
    inv_factors : Factors
    factors_superpot : tuple[Factors, ...]
    core : Core
    gs : GS

    Returns
    -------
    Core
    """
    cores = jnp.asarray([
        [fit(inv_factors, TuckerTensor(core, tuple(factors))) for factors in zip(*alpha_factors)]
        for alpha_factors in factors_superpot
    ])
    cores = gs.omega[None, :, *[None for _ in cores.shape[2:]]] * cores
    core = jnp.sum(cores, axis=(0, 1))
    return core


def superpotential(
    core: Core, factors_superpot: tuple[Factors, ...], gs: GS
) -> jax.Array:
    """Evaluates the superpotential

    Parameters
    ----------
    core : Core
    factors_superpot : tuple[Factors, ...]
    gs : GS

    Returns
    -------
    jax.Array
    """
    sp = jnp.asarray([
        [TuckerTensor(core, tuple(factors)).to_tensor() for factors in zip(*alpha_factors)]
        for alpha_factors in factors_superpot
    ])
    sp = gs.omega[None, :, *[None for _ in sp.shape[2:]]] * sp
    sp = jnp.sum(sp, axis=(0, 1))
    return sp


def merge_quad_info(*infos: QuadratureInfo) -> QuadratureInfo:
    return QuadratureInfo(
        err=jnp.max(jnp.asarray([jnp.max(i.err) for i in infos])),  # type: ignore
        neval=jnp.sum(jnp.asarray([jnp.sum(i.neval) for i in infos])),  # type: ignore
        status=jnp.max(jnp.asarray([jnp.max(i.status) for i in infos])),  # type: ignore
        info=None
    )
