import dataclasses

from jax.tree_util import register_dataclass

from . import *
from .base import TPELM
from .tensor_grid import TensorGrid


@partial(jax.jit, static_argnames=("degree", "extrapolate"))
def eval_spline(x: jax.Array, grid: jax.Array, coefs: jax.Array, degree: int = 3, extrapolate: bool = False):
    n = grid.shape[-1] - degree - 1 + 2 * degree
        
    assert coefs.shape[-1] == n, f"B-Spline has {n} basis functions! Only {coefs.shape[-1]} coefficients given."
    
    y = basis(x, grid, degree, extrapolate=extrapolate)
    return jnp.sum(y * coefs, axis=-1)


def _bspline_basis_deboor(x, t, degree, extrapolate):
    assert t.ndim == 1, "`grid` must be a ordered 1d array"
    
    t = jnp.concatenate([jnp.repeat(t[0], degree), t, jnp.repeat(t[-1], degree)])
    # Number of basis functions
    n = len(t) - degree - 1
    assert n > 0, f"`grid` not large enough for degree {degree}"

    # Find knot span index k such that t[k] <= x < t[k+1]
    k = jnp.searchsorted(t, x, side="right") - 1
    k = jnp.asarray(jnp.where(x == t[-1], n - 1, k))
    b = jnp.array([1.0])
    k = jnp.clip(k, degree, n - 1)
    
    for p in range(1, degree + 1):
        i = k - p + 1
        t1 = lax.dynamic_slice(t, i, p)
        t2 = lax.dynamic_slice(t, i + p, p)
        denom = (t2 - t1)
        a1 = jnp.where(
            denom != 0,
            (x - t1) / jnp.where(denom == 0, 1.0, denom),
            0.0
        ) * b
        i = k - p + 1
        t1 = lax.dynamic_slice(t, i + p, p)
        t2 = lax.dynamic_slice(t, i, p)
        denom = (t1 - t2)
        a2 = jnp.where(
            denom != 0, 
            (t1 - x) / jnp.where(denom == 0, 1.0, denom),
            0.0
        ) * b
        b = jnp.concatenate([a2[0][None], a1[:-1] + a2[1:], a1[-1][None]])
 
    basis = jnp.zeros((n,))
    basis = lax.dynamic_update_slice(basis, b, k - degree)
    if extrapolate:
        return basis
    else:
        return jnp.asarray(jnp.where((x < t[0]) | (t[-1] < x), jnp.zeros_like(basis), basis))


@partial(jax.jit, static_argnames=("degree", "extrapolate"))
def basis(x: jax.Array, grid: jax.Array, degree: int = 3, extrapolate: bool = False) -> jax.Array:
    def _basis(x, grid):
        return _bspline_basis_deboor(x, grid, degree, extrapolate)
    
    b = jnp.apply_along_axis(lambda x: _basis(x, grid), -1, x[..., None])
    return b


@partial(register_dataclass,
         data_fields=["grid"],
         meta_fields=["degree"])
@dataclasses.dataclass
class BSpline(TPELM):
    grid: TensorGrid
    degree: int = 3

    @property
    def dimension(self) -> int:
        return self.grid.dim
    
    @property
    def domain(self) -> TensorGrid:
        return self.grid
    
    def basis(self, x: jax.Array, mode: int) -> jax.Array:
        _basis = basis(x, self.grid[mode], degree=self.degree)
        return _basis
