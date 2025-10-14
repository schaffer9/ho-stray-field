from typing import NamedTuple
import string

from . import *


Factors = tuple[jax.Array, ...]
Core = jax.Array


class TuckerTensor(NamedTuple):
    core: Core
    factors: Factors

    def to_tensor(self) -> jax.Array:
        # Number of modes
        n_modes = len(self.factors)

        # Start index letters for einsum subscripts (use ASCII letters)
        letters = string.ascii_lowercase
        core_subs = letters[:n_modes]              # indices for core
        result_subs = letters[n_modes:2*n_modes]   # indices for result

        # Build einsum string dynamically
        einsum_str = core_subs
        for i in range(n_modes):
            einsum_str += f",{result_subs[i]}{core_subs[i]}"
        einsum_str += "->" + result_subs

        # Evaluate einsum
        return jnp.einsum(einsum_str, self.core, *self.factors)
