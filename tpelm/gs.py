from typing import NamedTuple

from quadax import quadgk, quadcc
from quadax.utils import QuadratureInfo

from . import *
from .bspline import BSpline
from .tensor_grid import TensorGrid
from .integrate import sinc_quad


class GS(NamedTuple):
    omega: jax.Array
    alpha: jax.Array


def _quad(g, x, interval, alpha, stds: int = 9, **kwargs):
    s = jnp.sqrt(1 / (2 * alpha))
    lb, ub = jnp.min(interval), jnp.max(interval)
    gaussian_interval = jnp.linspace(-s * stds, s * stds, 2 * stds + 1) - x
    _interval = jnp.sort(jnp.concatenate([interval, gaussian_interval, jnp.asarray([x])]))
    _interval = jnp.clip(_interval, lb, ub)
    return quadgk(g, _interval, **kwargs)


def integrate_gs_term(basis1d, x, interval, alpha, stds: int = 9, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return jnp.exp(-alpha * r) * basis1d(y)
    
    return _quad(g, x, interval, alpha, stds=stds, **kwargs)


def integrate_r2_gs_term(basis1d, x, interval, alpha, stds: int = 9, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return r * jnp.exp(-alpha * r) * basis1d(y)
    
    return _quad(g, x, interval, alpha, stds=stds, **kwargs)


def superpotential_factors(
        bspline: BSpline,
        target_tg: TensorGrid,
        quad_tg: TensorGrid,
        gs: GS,
        **kwargs
    ):

    # TODO: eventually mul factor with omega
    modes = list(range(quad_tg.dim))

    def _integrate(f, alpha: jax.Array, mode: int):
        def integrate_target(target):
            def b(y):
                return bspline.basis(y, mode=mode)
            interval = quad_tg[mode]
            I, info = f(b, target, interval, alpha, **kwargs)
            return I, info
        
        targets = target_tg[mode]
        return jax.vmap(integrate_target)(targets)
    
    I_gs, info1 = zip(*(lax.map(lambda a: _integrate(integrate_gs_term, a, mode), gs.alpha) for mode in modes))
    I_r2_gs, info2 = zip(*(lax.map(lambda a: _integrate(integrate_r2_gs_term, a, mode), gs.alpha) for mode in modes))

    factors = tuple(
        I_gs[:mode] + (I_r2_gs[mode],) + I_gs[mode+1:]
        for mode in modes
    )
    return factors, merge_quad_info(*info1, *info2)


def sinc_quad_1_over_sqrtx(rank: int, c0: float = 1.9):
    omega, alpha = sinc_quad(rank, c0)
    alpha = alpha ** 2
    omega = 1 / jnp.sqrt(jnp.pi) * omega
    return GS(omega, alpha)


def merge_quad_info(*infos: QuadratureInfo) -> QuadratureInfo:
    return QuadratureInfo(
        err=jnp.max(jnp.asarray([jnp.max(i.err) for i in infos])),  # type: ignore
        neval=jnp.sum(jnp.asarray([jnp.sum(i.neval) for i in infos])),  # type: ignore
        status=jnp.max(jnp.asarray([jnp.max(i.status) for i in infos])),  # type: ignore
        info=None
    )