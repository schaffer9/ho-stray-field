
import numpy as np
from scipy.integrate import nquad
import multiprocessing


def _flower_state_np(x):
    a, b, c = 1, 2, 1
    mx = 1 / a * x[..., 0] * x[..., 2]
    my = 1 / c * x[..., 1] * x[..., 2] + 1 / b**3 * x[..., 1]**3 * x[..., 2]**3
    mz = np.ones_like(mx)
    m = np.stack([mx, my, mz], axis=-1)
    m = m / np.linalg.norm(m, axis=-1, keepdims=True)
    return m


def direct_int(x):
    opts = [{"epsrel": 1e-16, "epsabs":1e-13, "points": [xi]}
            for xi in x]
    def integrand(y1, y2, y3, x, i):
        y = np.stack([y1, y2, y3])
        return (np.asarray(np.linalg.norm(x - y) * _flower_state_np(y)))[i]
    res0 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 0,), opts=opts, full_output=True)
    res1 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 1,), opts=opts, full_output=True)
    res2 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 2,), opts=opts, full_output=True)
    res = np.array([res0[0], res1[0], res2[0]])
    err = np.max(np.array([res0[1], res1[1], res2[1]]))
    return res, err


def sp_for_flower_with_np(targets):
    _targets = np.asarray(targets.reshape(-1, 3))
    with multiprocessing.Pool() as pool:
        results = list(pool.map(direct_int, _targets))
    sp = np.asarray([res[0] for res in results])
    err = np.asarray([res[1] for res in results])
    sp = sp.reshape(*targets.shape)
    max_err = np.max(err)
    return np.asarray(sp), np.asarray(max_err)


t = np.linspace(-0.5, 0.5, 5)
targets = np.stack(np.meshgrid(t, t, t, indexing="ij"), axis=-1)
sp, error = sp_for_flower_with_np(targets)

print("quadrature error:", error)
with open('sp_flower.npy', 'wb') as f:
    np.save(f, sp)
