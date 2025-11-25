from typing import NamedTuple, TypeAlias, Self
import string

from .prelude import *

Factors: TypeAlias = tuple[jax.Array, ...]
Core: TypeAlias = jax.Array


class TuckerTensor(NamedTuple):
    """Tucker tensor implementation

    Attributes
    ----------
    core : Core
    factors : Factors
    """
    core: Core
    factors: Factors

    def to_tensor(self) -> jax.Array:
        """Expands the Tucker tensor to a full tensor

        Returns
        -------
        jax.Array
        """
        letters = string.ascii_lowercase
        n_modes = len(self.factors)

        # Start index letters for einsum subscripts (use ASCII letters)
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

    def dot(self, B: Self) -> jax.Array:
        """Dot product with another Tucker Tensor

        Parameters
        ----------
        B : TuckerTensor

        Returns
        -------
        jax.Array
            Scalar
        """
        return tucker_dot(self, B)
    

def tucker_dot(A: TuckerTensor, B: TuckerTensor) -> jax.Array:
    """Dot product of two Tucker tensors

    Parameters
    ----------
    A : TuckerTensor
    B : TuckerTensor

    Returns
    -------
    jax.Array
        Scalar
    """
    letters = string.ascii_lowercase
    n_modes = len(A.factors)

    # Start index letters for einsum subscripts (use ASCII letters)
    core_dim1 = len(A.core.shape)
    core_dim2 = len(B.core.shape)
    
    extra_dims1 = core_dim1 - n_modes
    extra_dims2 = core_dim2 - n_modes
    
    core_subs1 = letters[:n_modes] + letters[2 * n_modes:2 * n_modes + extra_dims1]
    core_subs2 = letters[n_modes:2 * n_modes] + letters[2 * n_modes:2 * n_modes + extra_dims2]
    
    k = 2 * n_modes + max(extra_dims1, extra_dims2)
    factor_subs = letters[k:k + n_modes]
    
    einsum_str = f"{core_subs1},{core_subs2}"
    for i in range(n_modes):
        einsum_str += f",{factor_subs[i]}{core_subs1[i]}"
        
    for i in range(n_modes):
        einsum_str += f",{factor_subs[i]}{core_subs2[i]}"
    
    return jnp.einsum(einsum_str, A.core, B.core, *A.factors, *B.factors)

