from typing import Any, Callable, TypeAlias

from numpy.polynomial.legendre import leggauss

from . import *


Weights: TypeAlias = jax.Array
Nodes: TypeAlias = jax.Array
Grid1d: TypeAlias = jax.Array
QuadRule: TypeAlias = Callable[[Grid1d], tuple[Weights, Nodes]]


def gauss(degree: int) -> QuadRule:
    nodes, weights = map(jnp.asarray, leggauss(degree))
    
    def quad(domain: Grid1d) -> tuple[Weights, Nodes]:
        def weights_nodes(a, b):
            w = (b - a) / 2 * jnp.asarray(weights)
            n = (a + b) / 2 + nodes * (b - a) / 2
            return w, n
        w, n = jax.vmap(weights_nodes)(domain[:-1], domain[1:])
        w, n = w.ravel(), n.ravel()
        return w, n

    return quad
