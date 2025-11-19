from . import *


def flower_state(x):
    a, b, c = 1, 2, 1
    mx = 1 / a * x[..., 0] * x[..., 2]
    my = 1 / c * x[..., 1] * x[..., 2] + 1 / b**3 * x[..., 1]**3 * x[..., 2]**3
    mz = jnp.ones_like(mx)
    m = jnp.stack([mx, my, mz], axis=-1)
    m = m / jnp.linalg.norm(m, axis=-1, keepdims=True)
    return m


def vortex_state(x):
    rc = 0.14
    r = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    k = r**2 / rc**2
    c = jnp.sqrt(1 - jnp.exp(-4 * k))
    mx = -jnp.asarray(jnp.where(jnp.abs(r) < 1e-9, 0.0, x[..., 1] / r) * c)
    my = jnp.asarray(jnp.where(jnp.abs(r) < 1e-9, 0.0, x[..., 0] / r) * c)
    mz = jnp.exp(-2 * k)

    mag = jnp.stack([mx, my, mz], axis=-1)
    return mag  # no normalization


def m_uniform(x):
    m = jnp.zeros_like(x)
    return m.at[..., -1].set(1.0)
