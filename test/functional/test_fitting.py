import timeit
from functools import partial

import pytest

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.functional_tucker import fit
from tpelm.tucker_tensor import TuckerTensor

from .. import *
from ..sources import flower_state, vortex_state
from .utils import write_csv_row


@pytest.mark.functional
class TestFitting:
    quadrature_points = 140
    
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_fit_flower_state(self, n, k, device, artifact_dir):        
        if device == "gpu" and not jax.devices("gpu"):
            pytest.skip("GPU not available")
            
        _device = jax.devices(device)[0]
        
        csv_file = artifact_dir / f"fit_flower_{device}.csv"
        degree = k - 1
        tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
        elm = BSpline(tg, degree=degree)
        tg_quad = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
        
        inv_factors = elm.pinv(tg_quad)
        F = jnp.apply_along_axis(flower_state, -1, tg_quad.grid)
        
        @partial(jax.jit, device=_device)
        def fit_flower(inv_factors, F):
            core = fit(inv_factors, F)
            return core
        
        core = fit_flower(inv_factors, F).block_until_ready()
        runs = 10
        fit_time = timeit.timeit(lambda: fit_flower(inv_factors, F).block_until_ready(), number=runs) / runs
        
        tg_val = TensorGrid(*([jnp.linspace(-0.5, 0.5, 200)] * 3))
        factors = elm.factors(tg_val)
        
        mag_true = jnp.apply_along_axis(flower_state, -1, tg_val.grid)
        mag_pred = TuckerTensor(core, factors).to_tensor()
        err = jnp.max(jnp.abs(mag_true - mag_pred))

        data = {
            "k": k,
            "knots": n,
            "error_max": err,
            "fit_time": fit_time
        }
        write_csv_row(csv_file, data)
        
    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_fit_vortex_state(self, n, k, device, artifact_dir):
        if device == "gpu" and not jax.devices("gpu"):
            pytest.skip("GPU not available")
            
        _device = jax.devices(device)[0]
        
        csv_file = artifact_dir / f"fit_vortex_{device}.csv"
        degree = k - 1
        tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
        elm = BSpline(tg, degree=degree)
        tg_quad = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
        
        inv_factors = elm.pinv(tg_quad)
        F = jnp.apply_along_axis(vortex_state, -1, tg_quad.grid)
        
        @partial(jax.jit, device=_device)
        def fit_flower(inv_factors, F):
            core = fit(inv_factors, F)
            return core
        
        core = fit_flower(inv_factors, F).block_until_ready()
        runs = 10
        fit_time = timeit.timeit(lambda: fit_flower(inv_factors, F).block_until_ready(), number=runs) / runs
        
        tg_val = TensorGrid(*([jnp.linspace(-0.5, 0.5, 200)] * 3))
        factors = elm.factors(tg_val)
        
        mag_true = jnp.apply_along_axis(vortex_state, -1, tg_val.grid)
        mag_pred = TuckerTensor(core, factors).to_tensor()
        err = jnp.max(jnp.abs(mag_true - mag_pred))

        data = {
            "k": k,
            "knots": n,
            "error_max": err,
            "fit_time": fit_time
        }
        write_csv_row(csv_file, data)
