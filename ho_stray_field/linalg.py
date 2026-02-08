
from typing import Callable, Sequence
from dataclasses import replace

from jax import custom_vjp
import optax
from jaxtyping import Scalar

from .prelude import *
from .base import FunctionalTucker, TensorGrid, TPELM
from .tucker_tensor import TuckerTensor, Factors, Core


def reyleigh_quotient(
    h: Callable[[FunctionalTucker], FunctionalTucker], 
    v: FunctionalTucker, 
    tg: TensorGrid,
    custom_vjp: bool = True,
) -> Scalar:
    weights = tuple(jnp.sqrt(w) for w in tg.weights) if tg.weights is not None else None
    tg = TensorGrid(*tg.tensor_grid, weights=weights)
    facs_v = v.factors(tg, mul_weights=True)

    def _vhv(h, v_core, tg, elm):
        _v = FunctionalTucker(v_core, elm)
        hv = h(_v)
        facs_v = _v.factors(tg, mul_weights=True)
        facs_hv = hv.factors(tg, mul_weights=True)
        vhv = TuckerTensor(_v.core, facs_v).dot(TuckerTensor(hv.core, facs_hv))
        return vhv

    if custom_vjp:
        # we use a custom vjp rule to avoid differentiation of the linear operator.
        _vhv = jax.custom_vjp(_vhv, nondiff_argnums=(0,))
    
        def _vhv_fwd(h, v_core, tg, elm):
            _v = FunctionalTucker(v_core, elm)
            hv = h(_v)
            facs_v = _v.factors(tg, mul_weights=True)
            facs_hv = hv.factors(tg, mul_weights=True)
            vhv = TuckerTensor(_v.core, facs_v).dot(TuckerTensor(hv.core, facs_hv))
            return vhv, (hv.core, facs_v, facs_hv)

        def _vhv_bwd(h, res, g):
            hv_core, facs_v, facs_hv = res
            facs = tuple(fv.T @ fhv for fv, fhv in zip(facs_v, facs_hv))
            hv = TuckerTensor(hv_core, facs).to_tensor()
            return (2 * hv * g, None, None)

        _vhv.defvjp(_vhv_fwd, _vhv_bwd)
    
    vv = TuckerTensor(v.core, facs_v).dot(TuckerTensor(v.core, facs_v))
    return _vhv(h, v.core, tg, v.elm) / vv


def ortho(u: FunctionalTucker, base: Sequence[Core] | Core, tg: TensorGrid, unroll: bool | int = 1) -> FunctionalTucker:
    base = jnp.asarray(base, dtype=u.core.dtype)
    factors = _factors(u.elm, tg)
    core_u_ortho = _ortho(u.core, base, factors, unroll=unroll)
    core_u_ortho = _normalize(core_u_ortho, factors)
    return replace(u, core=core_u_ortho)
   
    
@partial(jax.jit, static_argnames=("unroll",))
def gram_schmidt(elm: TPELM, base: Sequence[Core] | Core, tg: TensorGrid, unroll: bool | int = False):
    assert len(base) > 0
    cores = jnp.asarray([u for u in base])
    ortho_base = jnp.zeros_like(cores)
    factors = _factors(elm, tg)

    def orthonormalize(state, u):
        i, ortho_base = state
        u = _ortho(u, ortho_base, factors, unroll=unroll)
        ortho_base = ortho_base.at[i].set(u)
        return (i + 1, ortho_base), _normalize(u, factors)
    
    _, e = jax.lax.scan(orthonormalize, (0, ortho_base), cores, unroll=unroll)
    return e


def _inner_prod(c1, c2, factors):
    return TuckerTensor(c1, factors).dot(TuckerTensor(c2, factors))


def _proj(v, onto, factors):
    is_zero = jnp.all(onto == 0)
    nom = _inner_prod(onto, v, factors)
    denom = _inner_prod(onto, onto, factors)
    return jax.lax.cond(is_zero, lambda: onto, lambda: nom / denom * onto)


def _ortho(u: Core, ortho_base: Sequence[Core] | Core, factors: Factors, unroll: bool | int = 1):
    ortho_base = jnp.asarray(ortho_base)
    def _ortho_inner(i, u):
        ui = ortho_base[i]
        u = u - _proj(u, ui, factors)
        return u
    
    return jax.lax.fori_loop(0, ortho_base.shape[0], _ortho_inner, u, unroll=unroll)


def _normalize(v, factors):
    norm = jnp.sqrt(_inner_prod(v, v, factors))
    return v / norm


def _factors(elm: TPELM, tg: TensorGrid):
    weights = tuple(jnp.sqrt(w) for w in tg.weights) if tg.weights is not None else None
    tg = TensorGrid(*tg.tensor_grid, weights=weights)
    return elm.factors(tg, mul_weights=True)


def eigenmodes(
    solver: optax.GradientTransformationExtraArgs,
    operator: Callable[[FunctionalTucker], FunctionalTucker],
    v0: FunctionalTucker,
    tg: TensorGrid,
    n: int,
    tol: float = 1e-7,
    maxiter: int | None = None,
    negate_op: bool = True,
    unroll: bool | int = 1,
    holomorphic: bool = False,
) -> tuple[jax.Array, Sequence[FunctionalTucker]]:
    mode = lambda c: FunctionalTucker(c, v0.elm)
    eigenvalues = jnp.zeros((n,), dtype=v0.core.dtype)
    mode_cores = jnp.zeros_like(jnp.asarray([v0.core for _ in range(n)]), dtype=v0.core.dtype)
    
    def body(i, state):
        v0, eigenvalues, mode_cores = state
        lam, mode = minimize_mode(
            solver, operator, v0, tg, mode_cores, 
            tol=tol, maxiter=maxiter, negate_op=negate_op, unroll=unroll,
            holomorphic=holomorphic,
        )
        eigenvalues = eigenvalues.at[i].set(lam)
        mode_cores = mode_cores.at[i].set(mode.core)
        return ortho(v0, mode_cores, tg, unroll=False), eigenvalues, mode_cores
    
    v0, eigenvalues, mode_cores = jax.lax.fori_loop(0, n, body, (v0, eigenvalues, mode_cores), unroll=False)
    return eigenvalues, [mode(c) for c in mode_cores]

    
def minimize_mode(
    solver: optax.GradientTransformationExtraArgs,
    operator: Callable[[FunctionalTucker], FunctionalTucker],
    v0: FunctionalTucker,
    tg: TensorGrid,
    modes: Sequence[Core] | Core,
    tol: float = 1e-4,
    maxiter: int | None = None,
    negate_op: bool = True,
    unroll: bool | int = 1,
    holomorphic: bool = False,
) -> tuple[Scalar, FunctionalTucker]:
    modes = jnp.asarray(modes, dtype=v0.core.dtype)
    elm = v0.elm
    mode = lambda c: FunctionalTucker(c, elm)
    
    @jax.jit
    def loss(mode_core):
        def _loss(c):
            return reyleigh_quotient(operator, mode(c), tg)
        
        l =_loss(mode_core)
        l = -l if negate_op else l
        return l    

    @jax.jit
    def update(params, opt_state, value, grad):
        
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=lambda p: loss(p)
        )
        params = optax.apply_updates(params, updates) 
        params = ortho(mode(params), modes, tg, unroll=unroll).core
        value, grad = jax.value_and_grad(loss, holomorphic=holomorphic)(params)
            
        return params, opt_state, value, grad
    
    def step(state):
        i, params, opt_state, value, grad = state
        params, opt_state, value, grad = update(params, opt_state, value, grad)
        return i + 1, params, opt_state, value, grad
    
    def cond(state):
        i, _, _, _, grad = state
        grad_norm = jnp.linalg.norm(grad)
        stop = grad_norm < tol
        if maxiter is not None:
            stop = stop | (i >= maxiter)
        
        return ~stop
    
    params = v0.core
    opt_state = solver.init(params)
    value, grad = jax.value_and_grad(loss, holomorphic=holomorphic)(params)
    iternum, params, opt_state, value, grad = jax.lax.while_loop(cond, step, (0, params, opt_state, value, grad))
    grad_norm = jnp.linalg.norm(grad)
    jax.debug.print("Found mode with lambda {value} after {iternum} iterations with tol {tol}", value=value, iternum=iternum, tol=grad_norm)
    return value, mode(params)
