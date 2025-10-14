from typing import NamedTuple

from quadax import quadgk

from . import *
from .bspline import BSpline
from .tensor_grid import TensorGrid


class GS(NamedTuple):
    omega: jax.Array
    alpha: jax.Array


def integrate_gs_term(basis1d, x, a, b, alpha, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return jnp.exp(-alpha * r) * basis1d(y)
    
    return quadgk(g, jnp.array([a, x, b]), **kwargs)


def integrate_r2_gs_term(basis1d, x, a, b, alpha, **kwargs):
    def g(y):
        r = (x - y) ** 2
        return r * jnp.exp(-alpha * r) * basis1d(y)
    
    return quadgk(g, jnp.array([a, x, b]), **kwargs)


def superpotential_factors(bspline: BSpline, tg: TensorGrid, gs: GS, **kwargs):
    grid = bspline.grid
    modes = list(range(grid.dim))

    def _integrate(f, alpha: jax.Array, mode: int):
        def integrate_target(target):
            def b(y):
                return bspline.basis(y, mode=mode)
            lb, ub = grid.bounds
            I, info = f(b, target, lb[mode], ub[mode], alpha, **kwargs)
            return I, info
        
        targets = tg[mode]
        return jax.vmap(integrate_target)(targets)
    
    I_gs, info1 = zip(*(jax.vmap(lambda a: _integrate(integrate_gs_term, a, mode))(gs.alpha) for mode in modes))
    I_r2_gs, info2 = zip(*(jax.vmap(lambda a: _integrate(integrate_r2_gs_term, a, mode))(gs.alpha) for mode in modes))

    factors = tuple(
        I_gs[:mode] + (I_r2_gs[mode],) + I_gs[mode+1:]
        for mode in modes
    )
    return factors, (info1, info2)


    # return I_gs, I_r2_gs, (info1, info2)