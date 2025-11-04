from typing import Sequence, Self

from jax.tree_util import register_pytree_node_class

from . import *
from .integrate import gauss


@register_pytree_node_class
class TensorGrid:
    def __init__(self, *knots: jax.Array, weights: None | jax.Array | tuple[jax.Array, ...] = None) -> None:
        self.tensor_grid = tuple(knots)
        if weights is not None:
            self.weights = tuple(w for w in weights)
        else:
            self.weights = tuple(jnp.ones_like(kv) for kv in self.tensor_grid)

    @classmethod
    def from_gauss(cls, *knots, degree: int | tuple[int, ...] = 3):
        dim = len(knots)
        if not isinstance(degree, Sequence):
            degree = tuple(degree for _ in range(dim))

        weights, nodes = zip(*(gauss(d)(kv) for d, kv in zip(degree, knots)))
        return cls(*nodes, weights=weights)

    def to_gauss(self, degree: int | tuple[int, ...]) -> Self:
        return self.__class__.from_gauss(*self.tensor_grid, degree=degree)

    def __getitem__(self, index):
        return self.tensor_grid[index]
    
    def __iter__(self):
        return (kv for kv in self.tensor_grid)

    @property
    def grid(self) -> jax.Array:
        mesh = jnp.stack(jnp.meshgrid(*self.tensor_grid, indexing="ij"), axis=-1)
        return mesh
    
    @property
    def bounds(self) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        _lb = tree.map(jnp.min, self.tensor_grid)
        _ub = tree.map(jnp.max, self.tensor_grid)
        return _lb, _ub

    @property
    def dim(self):
        return len(self.tensor_grid)

    def tree_flatten(self):
        return ((self.tensor_grid, self.weights), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*(children[0]), weights=children[1])
