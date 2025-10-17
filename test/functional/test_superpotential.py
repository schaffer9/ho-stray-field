import timeit
from functools import partial

import pytest

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.functional_tucker import fit, fit_divergence, fit_laplace, fit_grad
from tpelm.tucker_tensor import TuckerTensor, tucker_dot
from tpelm.gs import fit_superpotential, superpotential_factors, GS

from .. import *
from ..sources import flower_state, vortex_state, m_uniform
from .utils import write_csv_row


@pytest.mark.functional
class TestSuperpotential:
    quadrature_points = 140
    
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    # @pytest.mark.parametrize("n", [10, 20, 40, 80])
    # @pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8, 9])
    # @pytest.mark.parametrize("device", ["cpu", "gpu"])
    # def test_fit_flower_state(self, n, k, device, artifact_dir):        
    def test_energy_uniform_cube(self, artifact_dir):
        
        csv_file = artifact_dir / "sp_uniform.csv"
        
        @partial(jax.jit, device=jax.devices("cpu")[0])
        def _energy():
            degree = 6
            n = 10
            tg_m = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3))
            elm_m = BSpline(tg_m, degree=0)
            tg_quad = tg_m.to_gauss(3)
            inv_factors = elm_m.pinv(tg_quad)
            F = jnp.apply_along_axis(m_uniform, -1, tg_quad.grid)
            core_m = fit(inv_factors, F)
            gs = GS.from_sinc_1_over_sqrtx(46, 1.9)
            
            tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            elm_sp = BSpline(tg, degree=degree)
            tg_quad = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
            #tg_quad = tg.to_gauss(13)
            #inv_factors = elm_sp.pinv(tg_quad, alpha=1e-12)
            inv_factors = elm_sp.pinv(tg_quad)
            _, partials = elm_sp.factors_and_partials(tg_quad)
            sp_factors, info = superpotential_factors(
                elm_m, tg_quad, tg_m, gs,
                epsabs=1e-14, epsrel=0.0, order=41, max_ninter=20, full_output=True)
            core_sp = fit_superpotential(inv_factors, sp_factors, core_m)
            _core = fit_divergence(inv_factors, partials, core_sp)
            _core = fit_laplace(inv_factors, partials, _core)
            core_h = fit_grad(inv_factors, partials, _core)
            
            factors_h = elm_sp.factors(tg_quad, mul_weights=True)
            factors_m = elm_m.factors(tg_quad)
            e = -1 / 2 * tucker_dot(
                TuckerTensor(core_h, factors_h),
                TuckerTensor(core_m, factors_m),
            )
            return e, info
        
        e, info = _energy()
        
        print(info)
        print(e)
        
        assert False
        
        
        
        
        # def fit_flower(inv_factors, F):
        #     core = fit(inv_factors, F)
        #     return core
        
        # core = fit_flower(inv_factors, F).block_until_ready()
        # runs = 10
        # fit_time = timeit.timeit(lambda: fit_flower(inv_factors, F).block_until_ready(), number=runs) / runs
        
        # tg_val = TensorGrid(*([jnp.linspace(-0.5, 0.5, 200)] * 3))
        # factors = elm_m.factors(tg_val)
        
        # mag_true = jnp.apply_along_axis(flower_state, -1, tg_val.grid)
        # mag_pred = TuckerTensor(core, factors).to_tensor()
        # err = jnp.max(jnp.abs(mag_true - mag_pred))

        # data = {
        #     "k": k,
        #     "knots": n,
        #     "error_max": err,
        #     "fit_time": fit_time
        # }
        # write_csv_row(csv_file, data)
