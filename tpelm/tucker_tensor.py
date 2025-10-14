from typing import NamedTuple, TypeAlias
import string

from . import *


Factors: TypeAlias = tuple[jax.Array, ...]
Core: TypeAlias = jax.Array


class TuckerTensor(NamedTuple):
    core: Core
    factors: Factors

    def to_tensor(self) -> jax.Array:
        # Number of modes
        n_modes = len(self.factors)

        # Start index letters for einsum subscripts (use ASCII letters)
        letters = string.ascii_lowercase
        core_dim = len(self.core.shape)
        extra_dims = core_dim - n_modes  # for multivariate output
        core_subs = letters[:n_modes + extra_dims]              # indices for core
        result_subs = letters[n_modes + extra_dims:2*n_modes + extra_dims]   # indices for result
        result_subs += letters[n_modes:n_modes + extra_dims]
        # Build einsum string dynamically
        einsum_str = core_subs
        for i in range(n_modes):
            einsum_str += f",{result_subs[i]}{core_subs[i]}"
        einsum_str += "->" + result_subs
        # Evaluate einsum
        return jnp.einsum(einsum_str, self.core, *self.factors)
