from pathlib import Path

import pytest
import numpy as np

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.base import fit
from tpelm.gs import superpotential, superpotential_factors, GS

from .. import *
from ..sources import flower_state
from .utils import write_csv_row


@pytest.fixture(scope="module")
def flower_superpotential():
    file_path = Path(__file__).parent / "sp_flower.npy"
    with open(file_path, "rb") as f:
        sp_flower = np.load(f)
    
    return sp_flower


@pytest.mark.functional
class TestSuperpotential:
    
    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
    def test_superpotential_flower_state(self, k, artifact_dir, flower_superpotential):
        csv_file = artifact_dir / "flower_state_sp_approx.csv"
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
        }
        write_csv_row(csv_file, data)
