import abc
from typing import Sequence, Callable, overload

from . import *
from .tucker_tensor import TuckerTensor, Factors, Core
from .tensor_grid import TensorGrid


class FunctionalTucker(abc.ABC):
    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        ...

    @abc.abstractmethod
    def basis(self, x: jax.Array, mode: int) -> jax.Array:
        ...

    def __call__(self, tg: TensorGrid, core_tensor: Core) -> jax.Array:
        factors = self.factors(tg)
        tucker_tensor = TuckerTensor(core_tensor, factors)
        return tucker_tensor.to_tensor()
    
    def factors(self, tg: TensorGrid, mul_weights: bool = False) -> Factors:
        modes = list(range(self.dimension))
        factors = tuple(self.basis(x, m) for x, m in zip(tg, modes))
        
        if mul_weights:
            factors = tuple(w[:, None] * f for w, f in zip(tg.weights, factors))
        
        return factors
        
    def pinv(
            self, 
            tg: TensorGrid, 
            tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
        ) -> Factors:
        if not isinstance(tol, Sequence):
            tol = tuple(tol for _ in tg)

        factors = self.factors(tg)
        pinv_factors = tuple(regularized_pinv(X, wi, tol_i) for X, wi, tol_i in zip(factors, tg.weights, tol))
        return pinv_factors
    
    def factors_and_partials(self, tg: TensorGrid) -> tuple[Factors, tuple[Factors, ...]]:
        modes = list(range(self.dimension))
        _factors, _factors_derivative = tuple(zip(
            *(_elementwise_derivative(lambda x: self.basis(x, m), xi)
            for xi, m in zip(tg, modes))
        ))
        def _partial(i):
            f = _factors[:i] + (_factors_derivative[i],) + _factors[i+1:]
            return f

        partials = tuple(_partial(i) for i in range(self.dimension))
        return _factors, partials


@overload
def fit(inv_factors: Factors, F: jax.Array) -> Core: ...


@overload
def fit(inv_factors: Factors, F: TuckerTensor) -> Core: ...


@overload
def fit(inv_factors: Factors, F: Callable, tg: TensorGrid) -> Core: ...


# @overload
# def fit(func_tucker: FunctionalTucker, F: Callable, tg: TensorGrid) -> Core: ...


# @overload
# def fit(func_tucker: FunctionalTucker, F: FunctionalTucker, core: Core, tg: TensorGrid) -> Core: ...


def fit(
    inv_factors: Factors,
    F: jax.Array | TuckerTensor | Callable,
    tg: TensorGrid | None = None
) -> Core:
    if tg is not None:
        if not callable(F):
            raise TypeError("`fit` requires a function is a tensor grid is provided.")
        F = jnp.apply_along_axis(F, -1, tg.grid)
        return fit_to_array(inv_factors, F)
    elif isinstance(F, TuckerTensor):
        return fit_to_tucker(inv_factors, F)
    elif isinstance(F, jax.Array):
        return fit_to_array(inv_factors, F)
    else:
        raise NotImplementedError(f"Cannot fit TPELM to {type(F)}. Provide the right hand side as array or Tucker tensor.")


def fit_to_array(inv_factors: Factors, F: jax.Array) -> Core:
    tucker_tensor = TuckerTensor(F, inv_factors)
    return tucker_tensor.to_tensor()


def fit_to_tucker(inv_factors: Factors, F: TuckerTensor) -> Core:
    core, factors = F
    factors = tuple(Xinv @ f for Xinv, f in zip(inv_factors, factors))
    tucker_tensor = TuckerTensor(core, factors)
    return tucker_tensor.to_tensor()


def _elementwise_derivative(f, x):
    def _f(x):
        y = f(x)
        return y, y
    
    df, y = jax.vmap(jax.jacfwd(_f, has_aux=True))(x)
    return y, df


def regularized_pinv(X: jax.Array, weights: jax.Array | None = None, tol: jax.Array | float = 1e-12):
    if weights is None:
        weights = jnp.ones((X.shape[0],))
    
    X = jnp.sqrt(weights[:, None]) * X
    Xinv = jnp.linalg.pinv(X, rtol=tol) * jnp.sqrt(weights[None, :])
    return Xinv


def fit_divergence(inv_factors: Factors, partials: tuple[Factors, ...], core: Core) -> Core:
    _check_divergence_dim(partials, core)

    _partials = tuple(
        TuckerTensor(core[..., i], f)
        for i, f in enumerate(partials)
    )
    core = jnp.sum(jnp.asarray([
        fit(inv_factors, dtt)
        for dtt in _partials
    ]), axis=0)
    return core


def fit_laplace(inv_factors: Factors, partials: tuple[Factors, ...], core: Core) -> Core:
    _partials = tuple(
        TuckerTensor(core, f)
        for f in partials
    )
    d_cores = tuple(fit(inv_factors, p) for p in _partials)
    
    _partials2 = tuple(
        TuckerTensor(c, f)
        for c, f in zip(d_cores, partials)
    )
    dd_cores = tuple(fit(inv_factors, p) for p in _partials2)
    core = jnp.sum(jnp.asarray(dd_cores), axis=0)
    return core


def fit_grad(inv_factors: Factors, partials: tuple[Factors, ...], core: Core) -> Core:
    _partials = tuple(
        TuckerTensor(core, f)
        for f in partials
    )
    d_cores = tuple(fit(inv_factors, p) for p in _partials)
    return jnp.stack(d_cores, axis=-1)
    

def _check_divergence_dim(partials: tuple[Factors, ...], core: Core) -> None:
    dims = len(partials)
    assert core.ndim == (dims + 1)
    assert core.shape[-1] == dims

