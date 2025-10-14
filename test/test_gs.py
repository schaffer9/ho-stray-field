from tpelm.gs import GS, superpotential_factors, integrate_gs_term
from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid

from . import *


class TestIntegrateGSTerm(JaxTestCase):
    def test_000_integrate_basis(self):
        t = jnp.linspace(-0.5, 0.5, 5)
        tg = TensorGrid(t, t)
        bspline = BSpline(tg, degree=3)
        def b1(y):
            y = TensorGrid(jnp.asarray([y]))
            b = bspline.factors(y)
            print(tree.map(lambda t: t.shape, b))
            return b[0][0]
        
        y = b1(jnp.array(0.0))
        I, info = integrate_gs_term(b1, 0.1, -0.5, 0.5, 1000)
        print(tree.map(lambda t: t.shape, y))
        print("I", I.shape)
        print(info)
        assert False


class TestSuperpotential(JaxTestCase):
    def test_000_superpotential(self):
        t1 = jnp.linspace(-0.5, 0.5, 4)
        t2 = jnp.linspace(-0.5, 0.5, 5)
        t3 = jnp.linspace(-0.5, 0.5, 6)
        tg_spline = TensorGrid(t1, t2, t3)
        bspline = BSpline(tg_spline, degree=3)
        gs = GS(jnp.array([1.0, 1.0, 1.0, 1.0]), jnp.array([10, 100, 1000, 10000]))
        
        t1 = jnp.linspace(-0.5, 0.5, 10)
        t2 = jnp.linspace(-0.5, 0.5, 12)
        t3 = jnp.linspace(-0.5, 0.5, 14)

        tg_targets = TensorGrid(t1, t2, t3)

        factors, info = superpotential_factors(bspline, tg_targets, gs)
        print("factors", tree.map(lambda t: t.shape, factors))
    
        assert False