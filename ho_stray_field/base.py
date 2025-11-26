import abc
from typing import Self, Sequence, Callable, overload
from dataclasses import dataclass, replace

from jax.tree_util import register_dataclass

from .prelude import *
from .tucker_tensor import TuckerTensor, Factors, Core
from .tensor_grid import TensorGrid


Tol = jax.Array | float | tuple[jax.Array | float, ...]


class TPELM(abc.ABC):
    """Base class for Multilinear Tensor Product ELM
    """
    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        ...

    @abc.abstractmethod
    def basis(self, x: jax.Array, mode: int) -> jax.Array:
        ...

    @property
    @abc.abstractmethod
    def domain(self) -> TensorGrid:
        ... 

    def __call__(self, x: jax.Array | TensorGrid, core_tensor: Core) -> jax.Array:
        if isinstance(x, TensorGrid):
            factors = self.factors(x)
            tucker_tensor = TuckerTensor(core_tensor, factors)
            return tucker_tensor.to_tensor()
        else:
            def eval_point(x):
                x = TensorGrid(*(jnp.asarray(xi)[None] for xi in x))
                factors = self.factors(x)
                tucker_tensor = TuckerTensor(core_tensor, factors)
                return tucker_tensor.to_tensor()[*(0 for _ in range(len(factors)))]
            return jnp.apply_along_axis(eval_point, -1, x)

    def factors(self, tg: TensorGrid, mul_weights: bool = False) -> Factors:
        """Computes the factor matrices on the given tensor grid

        Parameters
        ----------
        tg : TensorGrid
        mul_weights : bool, optional
            multiply integration weights onto the factor matrices, by default False

        Returns
        -------
        Factors
        """
        modes = list(range(self.dimension))
        factors = tuple(self.basis(x, m) for x, m in zip(tg, modes))

        if mul_weights:
            factors = tuple(w[:, None] * f for w, f in zip(tg.weights, factors))

        return factors

    def pinv(self, tg: TensorGrid, tol: Tol = 0.0) -> Factors:
        """Computes the pseudoinverses of the factor matrices.
        Integration weights of the tensor grid are included

        Parameters
        ----------
        tg : TensorGrid
        tol : float | tuple[float, ...], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        Factors
            Inverse factors
        """
        factors = self.factors(tg)
        return factors_pinv(factors, tg.weights, tol=tol)

    def factors_and_partials(self, tg: TensorGrid) -> tuple[Factors, tuple[Factors, ...]]:
        """Computes the factors and partial derivatives of the factors for the given
        tensor grid.

        Parameters
        ----------
        tg : TensorGrid

        Returns
        -------
        tuple[Factors, tuple[Factors, ...]]
        """
        modes = list(range(self.dimension))
        _factors, _factors_derivative = tuple(
            zip(*(_elementwise_derivative(lambda x: self.basis(x, m), xi) for xi, m in zip(tg, modes)))
        )

        def _partial(i):
            f = _factors[:i] + (_factors_derivative[i],) + _factors[i + 1:]
            return f

        partials = tuple(_partial(i) for i in range(self.dimension))
        return _factors, partials

    def fit(
        self, 
        tg: TensorGrid, 
        f: TuckerTensor | jax.Array | Callable, 
        inv_factors: Factors | None = None,
        tol: Tol = 0.0,
    ) -> "FunctionalTucker":
        """Fits the TPELM to a function. The function can be given as
        a Tucker Tensor, array or callable corresponding to the given tensor grid
        or inverse factors.

        Parameters
        ----------
        tg : TensorGrid
        f : TuckerTensor | jax.Array | Callable
        inv_factors : Factors | None, optional
            if not provided, inverse factors are computed on the fly, by default None
        tol : float | tuple[jax.Array  |  float, ...], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FunctionalTucker
            the fitted function in `FunctionalTucker` format
        """
        if inv_factors is None:
            factors = self.factors(tg)
            inv_factors = factors_pinv(factors, tg.weights, tol=tol)

        if callable(f):
            core = fit(inv_factors, f, tg)
        else:
            core = fit(inv_factors, f)

        return FunctionalTucker(core, self)


@register_dataclass
@dataclass
class FunctionalTucker:
    """The Functional Tucker Tensor format represents a fitted function
    which included the core tensor and the corresponding basis functions
    given by a `TPELM`.
    """
    core: Core
    elm: TPELM

    def __call__(self, x: jax.Array | TensorGrid) -> jax.Array:
        return self.elm(x, self.core)
    
    def factors(self, tg: TensorGrid, mul_weights: bool = False) -> Factors:
        """Computes the factor matrices on the given tensor grid

        Parameters
        ----------
        tg : TensorGrid
        mul_weights : bool, optional
            multiply integration weights onto the factor matrices, by default False

        Returns
        -------
        Factors
        """
        return self.elm.factors(tg, mul_weights=mul_weights)
    
    def tt(self, tg: TensorGrid, mul_weights: bool = False) -> TuckerTensor:
        """Converts the Functional Tucker to a Tucker tensor.

        Parameters
        ----------
        tg : TensorGrid
        mul_weights : bool, optional
            multiply integration weights onto the factor matrices, by default False

        Returns
        -------
        Factors
        """
        factors = self.factors(tg, mul_weights=mul_weights)
        return TuckerTensor(self.core, factors)

    def factors_and_partials(self, tg: TensorGrid) -> tuple[Factors, tuple[Factors, ...]]:
        """Computes the factors and partial derivatives of the factors for the given
        tensor grid.

        Parameters
        ----------
        tg : TensorGrid

        Returns
        -------
        tuple[Factors, tuple[Factors, ...]]
        """
        return self.elm.factors_and_partials(tg)
    
    def refit(
        self, 
        tg: TensorGrid, 
        f: TuckerTensor | jax.Array | Callable,
        inv_factors: Factors | None = None,
        tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
    ) -> Self:
        """Refits the `FunctionalTucker` to some new function. The function can be given as
        a Tucker Tensor, array or callable corresponding to the given tensor grid
        or inverse factors.

        Parameters
        ----------
        tg : TensorGrid
        f : TuckerTensor | jax.Array | Callable
        inv_factors : Factors | None, optional
            if not provided, inverse factors are computed on the fly, by default None
        tol : float | tuple[jax.Array  |  float, ...], optional
            cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FunctionalTucker
            the fitted function in `FunctionalTucker` format
        """
        return replace(self, core=self.elm.fit(tg, f, inv_factors=inv_factors, tol=tol).core)
        
    def divergence(
        self, 
        tg: TensorGrid, 
        inv_factors: Factors | None = None, 
        tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
    ) -> Self:
        """Computes and refits the same Functional Tucker to the divergence.        

        Parameters
        ----------
        tg : TensorGrid
        f : TuckerTensor | jax.Array | Callable
        inv_factors : Factors | None, optional
            if not provided, inverse factors are computed on the fly, by default None
        tol : float | tuple[jax.Array  |  float, ...], optional
            Cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FunctionalTucker
            the divergence in `FunctionalTucker` format
        """
        factors, partials = self.factors_and_partials(tg)
        if inv_factors is None:
            inv_factors = factors_pinv(factors, weights=tg.weights, tol=tol)

        core = fit_divergence(inv_factors, partials, self.core)
        return replace(self, core=core)
    
    def laplace(
        self, 
        tg: TensorGrid, 
        inv_factors: Factors | None = None, 
        tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
    ) -> Self:
        """Computes and refits the same Functional Tucker to the Laplace operation.        

        Parameters
        ----------
        tg : TensorGrid
        f : TuckerTensor | jax.Array | Callable
        inv_factors : Factors | None, optional
            if not provided, inverse factors are computed on the fly, by default None
        tol : float | tuple[jax.Array  |  float, ...], optional
            Cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FunctionalTucker
            the Laplace operation in `FunctionalTucker` format
        """
        factors, partials = self.factors_and_partials(tg)
        if inv_factors is None:
            inv_factors = factors_pinv(factors, weights=tg.weights, tol=tol)

        core = fit_laplace(inv_factors, partials, self.core)
        return replace(self, core=core)
    
    def grad(
        self,
        tg: TensorGrid,
        inv_factors: Factors | None = None,
        tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
    ) -> Self:
        """Computes and refits the same Functional Tucker to the gradient.        

        Parameters
        ----------
        tg : TensorGrid
        f : TuckerTensor | jax.Array | Callable
        inv_factors : Factors | None, optional
            if not provided, inverse factors are computed on the fly, by default None
        tol : float | tuple[jax.Array  |  float, ...], optional
            Cutoff tolerance for small singular values, by default 0.0

        Returns
        -------
        FunctionalTucker
            the gradient operation in `FunctionalTucker` format
        """
        factors, partials = self.factors_and_partials(tg)
        if inv_factors is None:
            inv_factors = factors_pinv(factors, weights=tg.weights, tol=tol)

        core = fit_grad(inv_factors, partials, self.core)
        return replace(self, core=core)


@overload
def fit(inv_factors: Factors, f: jax.Array) -> Core: ...


@overload
def fit(inv_factors: Factors, f: TuckerTensor) -> Core: ...


@overload
def fit(inv_factors: Factors, f: FunctionalTucker, tg: TensorGrid) -> Core: ...


@overload
def fit(inv_factors: Factors, f: Callable, tg: TensorGrid) -> Core: ...


def fit(
    inv_factors: Factors,
    f: jax.Array | TuckerTensor | FunctionalTucker | Callable,
    tg: TensorGrid | None = None
) -> Core:
    """Fits a function `f`, which can be given as `Callable`, `FunctionalTucker`,
    as `TuckerTensor` or full tensor, and returns the core of the functional tucker
    approximation.

    Parameters
    ----------
    inv_factors : Factors
        inverse factors of the TPELM which should be used to fit the function
    f : jax.Array | TuckerTensor | FunctionalTucker | Callable
        the function which should be fitted in functional tucker format
    tg : TensorGrid | None, optional
        tensor grid which is used for quadrature

    Returns
    -------
    Core
    """
    if tg is not None:
        if not callable(f):
            raise TypeError("`fit` requires a function is a tensor grid is provided.")
        if isinstance(f, FunctionalTucker):
            tt = f.tt(tg)
            return fit_to_tucker(inv_factors, tt)
        else:
            f = jnp.apply_along_axis(f, -1, tg.grid)
            return fit_to_array(inv_factors, f)
    elif isinstance(f, TuckerTensor):
        return fit_to_tucker(inv_factors, f)
    elif isinstance(f, jax.Array):
        return fit_to_array(inv_factors, f)
    else:
        raise NotImplementedError(
            f"Cannot fit TPELM to {type(f)}. Provide the right hand side as array or Tucker tensor."
        )


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


def factors_pinv(
    factors: Factors,
    weights: tuple[jax.Array, ...] | None = None, 
    tol: jax.Array | float | tuple[jax.Array | float, ...] = 0.0
) -> Factors:
    """Computes the pseudoinverses of the given factor matrices

    Parameters
    ----------
    factors : Factors
    weights : tuple[jax.Array, ...] | None, optional
        optional quadrature weights, by default None
    tol : jax.Array | float | tuple[jax.Array  |  float, ...], optional
        cut-off tolerance for small singular values, by default 0.0

    Returns
    -------
    Factors
        inverse factor matrices
    """
    if not isinstance(tol, Sequence):
        tol = tuple(tol for _ in range(len(factors)))
    if weights is None:
        _weights = tuple(None for _ in range(len(factors)))
    else:
        _weights = weights

    pinv_factors = tuple(regularized_pinv(X, wi, tol_i) for X, wi, tol_i in zip(factors, _weights, tol))
    return pinv_factors


def regularized_pinv(X: jax.Array, weights: jax.Array | None = None, tol: jax.Array | float = 1e-12):
    if weights is None:
        weights = jnp.ones((X.shape[0],))
    
    X = jnp.sqrt(weights[:, None]) * X
    Xinv = jnp.linalg.pinv(X, rtol=tol) * jnp.sqrt(weights[None, :])
    return Xinv


def fit_divergence(inv_factors: Factors, partials: tuple[Factors, ...], core: Core) -> Core:
    """Fits the divergence to functional tucker format and 
    returns the core

    Parameters
    ----------
    inv_factors : Factors
        inverse factor matrices used for fitting
    partials : tuple[Factors, ...]
        partial derivatives of the factor matrices
    core : Core
        core of the original functional tucker

    Returns
    -------
    Core
        core of the new functional tucker which represents
        the divergence
    """
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
    """Fits the Laplace to functional tucker format and 
    returns the core

    Parameters
    ----------
    inv_factors : Factors
        inverse factor matrices used for fitting
    partials : tuple[Factors, ...]
        partial derivatives of the factor matrices
    core : Core
        core of the original functional tucker

    Returns
    -------
    Core
        core of the new functional tucker which represents
        the Laplace
    """
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
    """Fits the gradient to functional tucker format and 
    returns the core

    Parameters
    ----------
    inv_factors : Factors
        inverse factor matrices used for fitting
    partials : tuple[Factors, ...]
        partial derivatives of the factor matrices
    core : Core
        core of the original functional tucker

    Returns
    -------
    Core
        core of the new functional tucker which represents
        the gradient
    """
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
