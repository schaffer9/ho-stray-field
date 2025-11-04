
from tpelm.base import fit, regularized_pinv, fit_divergence, fit_laplace, fit_grad
from tpelm.bspline import basis, eval_spline, BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.tucker_tensor import TuckerTensor
from tpelm.integrate import gauss

from . import *


class TestFunctionalTucker(JaxTestCase):
    def test_000_call(self):
        t1 = jnp.linspace(0, 1, 5)
        t2 = jnp.linspace(-1, 0, 6)
        tg_basis = TensorGrid(t1, t2)
        ft = BSpline(tg_basis, degree=2)

        core = jnp.ones((6, 7))
        x = jnp.array([1, -1])
        y = ft(x, core)
        self.assertIsclose(y, jnp.array(1.0))

        x = jnp.array([[1, -1], [1, -1]])
        y = ft(x, core)
        self.assertIsclose(y, jnp.array([1.0, 1.0]))
