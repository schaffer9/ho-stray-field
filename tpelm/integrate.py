from typing import Any, Callable, Sequence, TypeAlias

from numpy.polynomial.legendre import leggauss

from . import *


Weights: TypeAlias = jax.Array
Nodes: TypeAlias = jax.Array
Grid1d: TypeAlias = jax.Array
QuadRule: TypeAlias = Callable[[Grid1d], tuple[Weights, Nodes]]


def midpoint(domain: jax.Array) -> tuple[Weights, Nodes]:
    w = domain[1:] - domain[:-1]
    nodes = (domain[1:] + domain[:-1]) / 2
    return w, nodes


def trap(domain: jax.Array) -> tuple[Weights, Nodes]:
    a, b = jnp.zeros(len(domain)), jnp.zeros(len(domain))
    d = domain[1:] - domain[:-1]
    a = a.at[:-1].set(d)
    b = b.at[1:].set(d)
    w = (a + b) / 2
    return w, domain


def simpson(domain: jax.Array) -> tuple[Weights, Nodes]:
    n = len(domain) + len(domain) - 1

    def weights(a, b):
        return (b - a) / 6 * jnp.array([1, 4, 1])

    _w = jax.vmap(weights)(domain[:-1], domain[1:])
    w = jnp.zeros(n)
    w = w.at[0].set(_w[0, 0])
    w = w.at[-1].set(_w[-1, -1])
    w = w.at[1::2].set(_w[:, 1])
    w = w.at[2:-1:2].set(_w[:-1, 2] + _w[1:, 0])
    m = (domain[:-1] + domain[1:]) / 2
    i = jnp.arange(1, len(domain))
    nodes = jnp.insert(domain, i, m)
    return w, nodes


def gauss(degree: int) -> QuadRule:
    nodes, weights = map(jnp.asarray, leggauss(degree))
    
    def quad(domain: jax.Array) -> tuple[Weights, Nodes]:
        def weights_nodes(a, b):
            w = (b - a) / 2 * jnp.asarray(weights)
            n = (a + b) / 2 + nodes * (b - a) / 2
            return w, n
        w, n = jax.vmap(weights_nodes)(domain[:-1], domain[1:])
        w, n = w.ravel(), n.ravel()
        return w, n

    return quad
