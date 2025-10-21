import timeit
from functools import partial

import pytest
from quadax import quadgk

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.functional_tucker import fit, fit_divergence, fit_laplace, fit_grad
from tpelm.tucker_tensor import TuckerTensor, tucker_dot
from tpelm.gs import superpotential, fit_superpotential, superpotential_factors, merge_quad_info, GS

from .. import *
from ..sources import flower_state, vortex_state, m_uniform
from .utils import write_csv_row


targets = TensorGrid(*([jnp.linspace(-0.5, 0.5, 10)] * 3))

@pytest.fixture(scope="module")
def flower_superpotential():
    print("computing")
    quad = partial(quadgk, epsabs=1e-10, epsrel=0.0, max_ninter=30)
    
    @jax.jit
    def direct_int(x):
        def i1(y1):
            def i2(y2, y1):
                def i3(y3, y2, y1):
                    y = jnp.stack([y1, y2, y3])
                    return jnp.linalg.norm(x - y) * flower_state(y)
                
                return quad(i3, jnp.array([-0.5, x[2], 0.5]), (y2, y1))[0]
            
            return quad(i2, jnp.array([-0.5, x[1], 0.5]), (y1,))[0]
        
        I, info = quad(i1, jnp.array([-0.5, x[0], 0.5]))
        return 1 / (8 * jnp.pi) * I, info

    _targets = targets.grid.reshape(-1, 3)
    result, info = jax.lax.map(direct_int, _targets, batch_size=20)
    result = result.reshape(*targets.grid.shape)
    return result, info


@pytest.mark.functional
class TestSuperpotential:
    
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    # @pytest.mark.parametrize("n", [10, 20, 40, 80])
    # @pytest.mark.parametrize("device", ["cpu", "gpu"])
    # def test_fit_flower_state(self, n, k, device, artifact_dir):  
     
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
    def test_superpotential_flower_state(self, k, artifact_dir, flower_superpotential):
        csv_file = artifact_dir / "flower_state_sp_approx.csv"
        result, info_direct = flower_superpotential
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
        sp_factors, info = superpotential_factors(elm_m, targets, tg_m, gs,
                                                  epsabs=1e-14, epsrel=0.0, order=31, max_ninter=200)

        gs_result = superpotential(core_m, sp_factors, gs)

        err = jnp.max(jnp.abs(gs_result - result))
        data = {
            "k": k,
            "r": n,
            "error_max": err,
            "int_error": info.err,
            "int_status": info.status,
            "direct_int_err": merge_quad_info(info_direct).err,
            "direct_int_status": merge_quad_info(info_direct).status
        }
        write_csv_row(csv_file, data)
