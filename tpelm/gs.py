from typing import NamedTuple

from quadax import quadgk
from quadax.utils import QuadratureInfo

from . import *
from .tpelm import fit
from .bspline import BSpline
from .tensor_grid import TensorGrid
from .integrate import sinc_quad_1_over_sqrtx
from .tucker_tensor import TuckerTensor


class GS(NamedTuple):
    omega: jax.Array
    alpha: jax.Array

    @classmethod
    def from_sinc_1_over_sqrtx(cls, rank: int, c0: float = 1.9):
        return cls(*sinc_quad_1_over_sqrtx(rank, c0))


def _quad(g, x, interval, alpha, stds: int = 9, **kwargs):
    s = jnp.sqrt(1 / (2 * jnp.maximum(alpha, 1.0)))
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
    I_r2_gs = tree.map(lambda I: 1 / (8 * jnp.pi) * gs.omega[:, *(None for _ in I.shape[1:])] * I, I_r2_gs)
    factors = tuple(
        I_gs[:mode] + (I_r2_gs[mode],) + I_gs[mode+1:]
        for mode in modes
    )
    return factors, merge_quad_info(*info1, *info2)


def fit_superpotential(
    factors_pinv, factors_superpot, core
):
    cores = jnp.asarray([
        [fit(factors_pinv, TuckerTensor(core, tuple(factors))) for factors in zip(*alpha_factors)]
        for alpha_factors in factors_superpot
    ])
    core = jnp.sum(cores, axis=(0, 1))
    return core


def merge_quad_info(*infos: QuadratureInfo) -> QuadratureInfo:
    return QuadratureInfo(
        err=jnp.max(jnp.asarray([jnp.max(i.err) for i in infos])),  # type: ignore
        neval=jnp.sum(jnp.asarray([jnp.sum(i.neval) for i in infos])),  # type: ignore
        status=jnp.max(jnp.asarray([jnp.max(i.status) for i in infos])),  # type: ignore
        info=None
    )