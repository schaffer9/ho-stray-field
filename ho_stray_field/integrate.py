from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.polynomial.legendre import leggauss

from .prelude import *


Weights: TypeAlias = jax.Array
Nodes: TypeAlias = jax.Array
Grid1d: TypeAlias = jax.Array
QuadRule: TypeAlias = Callable[[Grid1d], tuple[Weights, Nodes]]


def gauss(degree: int) -> QuadRule:
    """Gauss–Legendre quadrature rule.

    Parameters
    ----------
    degree : int

    Returns
    -------
    QuadRule
    
    Examples
    --------
    >>> weights, nodes = gauss(3)(jnp.array([-10, 0, 10]))
    >>> weights
    Array([2.777778 , 4.4444447, 2.777778 , 2.777778 , 4.4444447, 2.777778 ],      dtype=float32)
    >>> nodes
    Array([-8.872984 , -5.       , -1.1270165,  1.1270165,  5.       ,
            8.872984 ], dtype=float32)
    """
    nodes, weights = tree.map(jnp.asarray, leggauss(degree))
    
    def quad(domain: Grid1d) -> tuple[Weights, Nodes]:
        def weights_nodes(a, b):
            w = (b - a) / 2 * jnp.asarray(weights)
            n = (a + b) / 2 + nodes * (b - a) / 2
            return w, n
        w, n = jax.vmap(weights_nodes)(domain[:-1], domain[1:])
        w, n = w.ravel(), n.ravel()
        return w, n

    return quad


def sinc_quad(n: int, c0: float, dtype=np.float128) -> tuple[Weights, Nodes]:
    """Sinc quadrature

    Parameters
    ----------
    n : int
        number or quadrature nodes
    c0 : float
        spacing

    Returns
    -------
    tuple[Weights, Nodes]
    """
    n = np.array(n, dtype=dtype)
    c0 = np.array(c0, dtype=dtype)
    h = c0 * np.log(n) / n
    j = np.arange(n)
    x = np.sinh(j * h)
    w = 2 * h * np.cosh(j * h)
    w[0] = h
    return w, x


def sinc_quad_1_over_sqrtx(rank: int, c0: float = 1.9, dtype=np.float128):
    r"""Sinc quadrature for :math:`1/\sqrt{x}`.

    Parameters
    ----------
    n : int
        number or quadrature nodes
    c0 : float
        spacing

    Returns
    -------
    tuple[Weights, Nodes]
    """
    omega, alpha = sinc_quad(rank, c0, dtype=dtype)
    alpha = alpha ** 2
    omega = 1 / np.sqrt(np.pi) * omega
    return omega, alpha
