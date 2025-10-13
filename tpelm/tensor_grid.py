from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class

from . import *


@register_pytree_node_class
@dataclass(init=False)
class TensorGrid:
    tensor_grid: tuple[jax.Array, ...]
    _lb: tuple[jax.Array, ...]
    _ub: tuple[jax.Array, ...]
    weights: tuple[jax.Array, ...]

    def __init__(self, *knots: jax.Array, weights: None | jax.Array | tuple[jax.Array, ...] = None) -> None:
        self.tensor_grid = tuple(kv for kv in knots)
        self._lb = tuple(jnp.min(kv) for kv in self.tensor_grid)
        self._ub = tuple(jnp.max(kv) for kv in self.tensor_grid)
        if weights is not None:
            self.weights = tuple(w for w in weights)
        else:
            self.weights = tuple(jnp.ones_like(kv) for kv in self.tensor_grid)

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
        return self._lb, self._ub

    @property
    def dim(self):
        return len(self.tensor_grid)

    def tree_flatten(self):
        return ((self.tensor_grid, self.weights), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children[0], weights=children[1])