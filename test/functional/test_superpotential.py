import timeit
from functools import partial
from pathlib import Path


import pytest
import numpy as np

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.base import fit
from tpelm.tucker_tensor import TuckerTensor, tucker_dot
from tpelm.gs import superpotential, fit_superpotential, superpotential_factors, merge_quad_info, GS

from .. import *
from ..sources import flower_state, vortex_state, m_uniform
from .utils import write_csv_row


# def _flower_state_np(x):
#     a, b, c = 1, 2, 1
#     mx = 1 / a * x[..., 0] * x[..., 2]
#     my = 1 / c * x[..., 1] * x[..., 2] + 1 / b**3 * x[..., 1]**3 * x[..., 2]**3
#     mz = np.ones_like(mx)
#     m = np.stack([mx, my, mz], axis=-1)
#     m = m / np.linalg.norm(m, axis=-1, keepdims=True)
#     return m


# def direct_int(x):
#     opts = [{"epsrel": 1e-16, "epsabs":1e-12, "points": [xi]}
#             for xi in x]
#     def integrand(y1, y2, y3, x, i):
#         y = np.stack([y1, y2, y3])
#         return (np.asarray(np.linalg.norm(x - y) * _flower_state_np(y)))[i]
#     res0 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 0,), opts=opts, full_output=True)
#     res1 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 1,), opts=opts, full_output=True)
#     res2 = nquad(integrand, [(-0.5, 0.5)] * 3, (x, 2,), opts=opts, full_output=True)
#     res = np.array([res0[0], res1[0], res2[0]])
#     err = np.max(np.array([res0[1], res1[1], res2[1]]))
#     return 1 / (8 * np.pi) * res, err


# def sp_for_flower_with_np(targets):
#     pool = multiprocessing.Pool()
#     _targets = np.asarray(targets.reshape(-1, 3))
#     results = list(pool.map(direct_int, _targets))
#     sp = np.asarray([res[0] for res in results])
#     err = np.asarray([res[1] for res in results])
#     sp = sp.reshape(*targets.shape)
#     max_err = np.max(err)
#     return np.asarray(sp), np.asarray(max_err)


# t = np.linspace(-0.5, 0.5, 2)
# targets = np.stack(np.meshgrid(t, t, t), axis=-1)
# sp, error = sp_for_flower_with_np(targets)


@pytest.fixture(scope="module")
def flower_superpotential():
    file_path = Path(__file__).parent / "sp_flower.npy"
    with open(file_path, "rb") as f:
        sp_flower = np.load(f)
    
    return sp_flower
    #return sp_for_flower_with_np(targets)



@pytest.mark.functional
class TestSuperpotential:
    
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
    def test_superpotential_flower_state(self, k, artifact_dir, flower_superpotential):
        csv_file = artifact_dir / "flower_state_sp_approx.csv"
        #result, max_quad_error = flower_superpotential
        result = flower_superpotential
        degree = k - 1
        n = 40

        # setup mag elm
        tg_m = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
        elm_m = BSpline(tg_m, degree=degree)
        tg_quad = tg_m.to_gauss(degree)
        F = jnp.apply_along_axis(flower_state, -1, tg_quad.grid)
        inv_factors = elm_m.pinv(tg_quad)
        core_m = fit(inv_factors, F)

        # compute superpotential
        gs = GS.from_sinc_1_over_sqrtx(46, 1.9)
        #t = jnp.linspace(-0.5, 0.5, result.shape[0])
        _t = (jnp.linspace(-0.5, 0.5, result.shape[i]) for i in range(3))
        _targets = TensorGrid(*_t)
        sp_factors, info = superpotential_factors(elm_m, _targets, tg_m, gs,
                                                  epsabs=1e-14, epsrel=0.0, order=31, max_ninter=200)

        gs_result = superpotential(core_m, sp_factors, gs)
        
        err = jnp.max(jnp.abs(gs_result - result))
        data = {
            "k": k,
            "r": n,
            "error_max": err,
            "int_error": info.err,
            "int_status": info.status,
            #"direct_int_err": max_quad_error,
        }
        write_csv_row(csv_file, data)
