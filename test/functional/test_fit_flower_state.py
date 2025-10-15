import timeit

import pytest

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.functional_tucker import fit
from tpelm.tucker_tensor import TuckerTensor

from .. import *
from ..sources import flower_state
from .utils import write_csv_row


@pytest.mark.functional
class TestFitFlower:
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8])
    def test_fit_flower_state(self, n, k, artifact_dir):
        csv_file = artifact_dir / "fit_flower.csv"
        degree = k - 1
        tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
        elm = BSpline(tg, degree=degree)
        tg_quad = tg.to_gauss(3)
        
        inv_factors = elm.pinv(tg_quad)

        @jax.jit
        def fit_flower(inv_factors, tg_quad):
            core = fit(inv_factors, flower_state, tg_quad)
            return core
        
        core = fit_flower(inv_factors, tg_quad).block_until_ready()
        fit_time = timeit.timeit(lambda: fit_flower(inv_factors, tg_quad).block_until_ready(), number=5)
        
        # key = random.key(0)
        # t1, t2, t3 = random.uniform(key, (3, 10), minval=-0.5, maxval=0.5)
        # tg_val = TensorGrid(t1, t2, t3)
        tg_val = TensorGrid(*([jnp.linspace(-0.5, 0.5, 150)] * 3))
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
    
