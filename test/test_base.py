
from ho_stray_field.base import fit, regularized_pinv, fit_divergence, fit_laplace, fit_grad
from ho_stray_field.bspline import basis, eval_spline, BSpline
from ho_stray_field.tensor_grid import TensorGrid
from ho_stray_field.tucker_tensor import TuckerTensor
from ho_stray_field.integrate import gauss

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
