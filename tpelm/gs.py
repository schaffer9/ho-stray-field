from typing import NamedTuple

from quadax import quadgk, quadts, quadcc
from quadax.utils import QuadratureInfo

from . import *
from .functional_tucker import FunctionalTucker, fit
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


def _quad(g, x, interval, alpha, stds: int = 2, **kwargs):
    s = jnp.sqrt(1 / (2 * jnp.maximum(alpha, 1.0)))
    lb, ub = jnp.min(interval), jnp.max(interval)
    gaussian_interval = jnp.linspace(-s * stds, s * stds, 2 * stds + 1) + x
    _interval = jnp.sort(jnp.concatenate([interval, gaussian_interval]))
    _interval = jnp.clip(_interval, lb, ub)
    return quadgk(g, _interval, **kwargs)


# def integrate_gs_term(basis1d, x, interval, alpha, stds: int = 2, **kwargs):
#     def g(y):
#         r = (x - y) ** 2
#         return jnp.exp(-alpha * r) * basis1d(y)
    
#     return _quad(g, x, interval, alpha, stds=stds, **kwargs)


# def integrate_r2_gs_term(basis1d, x, interval, alpha, stds: int = 2, **kwargs):
#     def g(y):
#         r = (x - y) ** 2
#         return r * jnp.exp(-alpha * r) * basis1d(y)
    
#     return _quad(g, x, interval, alpha, stds=stds, **kwargs)
def integrate_gs_term(basis1d, x, interval, alpha, stds: int = 2, **kwargs):
    with jax.ensure_compile_time_eval():
        rank = basis1d(x).shape[0]

    def g_basis_k(k):
        def g(y):
            r = (x - y) ** 2
            return jnp.exp(-alpha * r) * basis1d(y)[k]
        
        return _quad(g, x, interval, alpha, stds=stds, **kwargs)

    return jax.vmap(g_basis_k)(jnp.arange(rank))

def integrate_r2_gs_term(basis1d, x, interval, alpha, stds: int = 2, **kwargs):
    with jax.ensure_compile_time_eval():
        rank = basis1d(x).shape[0]

    def g_basis_k(k):
        def g(y):
            r = (x - y) ** 2
            return r * jnp.exp(-alpha * r) * basis1d(y)[k]
        
        return _quad(g, x, interval, alpha, stds=stds, **kwargs)

    return jax.vmap(g_basis_k)(jnp.arange(rank))


def superpotential_factors(
    bspline: FunctionalTucker,
    target_tg: TensorGrid,
    quad_tg: TensorGrid,
    gs: GS,
    batch_size: int | None = None,
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
        if batch_size is None:
            return jax.vmap(integrate_target)(targets)
        else:
            return jax.lax.map(integrate_target, targets)
    
    I_gs, info1 = zip(*(lax.map(lambda a: _integrate(integrate_gs_term, a, mode), gs.alpha) for mode in modes))
    I_r2_gs, info2 = zip(*(lax.map(lambda a: _integrate(integrate_r2_gs_term, a, mode), gs.alpha) for mode in modes))
    factors = tuple(
        I_gs[:mode] + (I_r2_gs[mode],) + I_gs[mode+1:]
        for mode in modes
    )
    return factors, merge_quad_info(*info1, *info2)


def fit_superpotential(
    inv_factors, factors_superpot, core, gs
):
    cores = jnp.asarray([
        [fit(inv_factors, TuckerTensor(core, tuple(factors))) for factors in zip(*alpha_factors)]
        for alpha_factors in factors_superpot
    ])
    cores = gs.omega[None, :, *[None for _ in cores.shape[2:]]] * cores
    core = jnp.sum(cores, axis=(0, 1))
    return core


def superpotential(
    core, factors_superpot, gs
):
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
