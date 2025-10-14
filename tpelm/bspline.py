from typing import Callable, Sequence
import dataclasses

from jax.tree_util import register_dataclass

from . import *
from .tensor_grid import TensorGrid
from .tucker_tensor import Core, Factors, TuckerTensor



def _base_fun(x: jax.Array, k: int, i: int, t: jax.Array, degree: int, extrapolate: bool) -> jax.Array:
    if k == 0:
        n = len(t) - k - 1
        a1 = jnp.where((t[..., i] <= x) & (x < t[..., i + 1]), 1.0, 0.0)
        if extrapolate:
            # a2 and a3 for extrapolation
            a2 = jnp.where((x < t[..., 0]) & (i <= degree), 1.0, 0.0)
            a3 = jnp.where((x >= t[..., -1]) & (i >= n - degree - 1), 1.0, 0.0)
            return a1 + a2 + a3
        else:
            return a1

    clamped_nodes = t[..., i + k] == t[..., i]
    c1: jax.Array = jnp.where(
        clamped_nodes,
        0.0,
        (x - t[..., i])
        / jnp.where(clamped_nodes, jnp.inf, (t[..., i + k] - t[..., i]))
        * _base_fun(x, k - 1, i, t, degree, extrapolate),
    )
    clamped_nodes = t[..., i + k + 1] == t[..., i + 1]
    c2: jax.Array = jnp.where(
        t[..., i + k + 1] == t[..., i + 1],
        0.0,
        (t[..., i + k + 1] - x)
        / jnp.where(clamped_nodes, jnp.inf, (t[..., i + k + 1] - t[..., i + 1]))
        * _base_fun(x, k - 1, i + 1, t, degree, extrapolate),
    )

    return c1 + c2


@partial(jax.jit, static_argnames=("degree", "open_spline"))
def eval_spline(x: jax.Array, grid: jax.Array, coefs: jax.Array, degree: int = 3, open_spline: bool = False):
    if open_spline:
        n = grid.shape[-1] - degree - 1
    else:
        n = grid.shape[-1] - degree - 1 + 2 * degree
        
    assert coefs.shape[-1] == n, f"B-Spline has {n} basis functions! Only {coefs.shape[-1]} coefficients given."
    
    y = basis(x, grid, degree, open_spline=open_spline)
    return jnp.sum(y * coefs, axis=-1)


@partial(jax.jit, static_argnames=("degree", "open_spline"))
def basis(x: jax.Array, grid: jax.Array, degree: int = 3, open_spline: bool = False) -> jax.Array:
    """Computes the spine basis with respect to the provided grid.

    Parameters
    ----------
    x : jax.Array
    grid : jax.Array
        spline grid; if multidimensional, this will evaluate the spline for each dimension
        and the output can be seen as a compressed tensor.
    degree : int, optional
        spline degree

    Returns
    -------
    jax.Array
    """
    if not open_spline:
        g0 = grid[..., 0:1]
        g1 = grid[..., -1:]
        grid = jnp.concatenate([jnp.repeat(g0, degree, axis=-1), grid, jnp.repeat(g1, degree, axis=-1)], axis=-1)
        extrapolate = True
    else:
        extrapolate = False

    n = grid.shape[-1] - degree - 1
    return jax.vmap(lambda i: _base_fun(x, degree, i, grid, degree, extrapolate), 0, -1)(jnp.arange(n))


def regularized_pinv(X: jax.Array, weights: jax.Array | None = None, alpha: jax.Array | float = 0.0, tol: jax.Array | float = 0.0):
    if weights is None:
        weights = jnp.ones((X.shape[0],))
    
    X = weights[:, None] * X
    U, S, VT = jnp.linalg.svd(X, full_matrices=False)
    Sinv = jnp.where((S ** 2 + alpha) < tol, 0.0, S / (S ** 2 + alpha))
    Xinv = (VT.T * Sinv) @ (weights[:, None] * U).T  # compute pseudoinverse
    return Xinv


@partial(register_dataclass,
         data_fields=["grid"],
         meta_fields=["degree"])
@dataclasses.dataclass
class BSpline:
    grid: TensorGrid
    degree: int = 3

    def __call__(self, tg: TensorGrid, core_tensor: Core) -> jax.Array:
        factors = self.factors(tg)
        tucker_tensor = TuckerTensor(core_tensor, factors)
        return tucker_tensor.to_tensor()
        
    def factors(self, tg: TensorGrid) -> Factors:
        _basis = tuple(basis(xi, t, degree=self.degree) for xi, t in zip(tg, self.grid))
        return _basis
    
    def basis(self, x: jax.Array, mode: int) -> jax.Array:
        _basis = basis(x, self.grid[mode], degree=self.degree)
        return _basis

    def factors_pinv(
            self, 
            tg: TensorGrid, 
            alpha: jax.Array | float | tuple[jax.Array | float, ...] = 0.0, 
            tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
        ) -> Factors:
        if not isinstance(alpha, Sequence):
            alpha = tuple(alpha for _ in tg)

        if not isinstance(tol, Sequence):
            tol = tuple(tol for _ in tg)

        factors = self.factors(tg)
        pinv_factors = tuple(regularized_pinv(X, wi, alpha_i, tol_i) for X, wi, alpha_i, tol_i in zip(factors, tg.weights, alpha, tol))
        return pinv_factors
    
    def factors_derivative(self, tg: TensorGrid) -> tuple[Factors, Factors]:
        _factors, _factors_derivative = tuple(zip(
            *(_elementwise_derivative(lambda xi: basis(xi, t, degree=self.degree), xi)
            for xi, t in zip(tg, self.grid))
        ))
        return _factors, _factors_derivative


def fit(factors_pinv: Factors, F: jax.Array | TuckerTensor):
    if isinstance(F, TuckerTensor):
        return fit_to_tucker(factors_pinv, F)
    elif isinstance(F, jax.Array):
        return fit_to_array(factors_pinv, F)
    else:
        raise NotImplementedError(f"Cannot fit TPELM to {type(F)}. Provide the right hand side as array or Tucker tensor.")


def fit_to_array(factors_pinv: Factors, F: jax.Array) -> Core:
    tucker_tensor = TuckerTensor(F, factors_pinv)
    return tucker_tensor.to_tensor()


def fit_to_tucker(factors_pinv: Factors, F: TuckerTensor) -> Core:
    core, factors = F
    factors = tuple(Xinv @ f for Xinv, f in zip(factors_pinv, factors))
    tucker_tensor = TuckerTensor(core, factors)
    return tucker_tensor.to_tensor()


def _elementwise_derivative(f, x):
    def _f(x):
        y = f(x)
        return y, y
    
    df, y = jax.vmap(jax.jacfwd(_f, has_aux=True))(x)
    return y, df
