from tpelm.gs import GS, superpotential_factors, integrate_gs_term, integrate_r2_gs_term, fit_superpotential
from tpelm.base import fit, factors_pinv
from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.integrate import gauss, sinc_quad_1_over_sqrtx

from . import *
from .sources import flower_state


def test_gs_from_sinc():
    gs = GS.from_sinc_1_over_sqrtx(100, 2.0)
    assert gs.omega.shape == (100,)
    assert gs.alpha.shape == (100,)


class TestIntegrateGS(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_000_integrate_gs_basis(self):
        t = jnp.linspace(-0.5, 0.5, 5)
        tg = TensorGrid(t, t)
        bspline = BSpline(tg, degree=3)
        def b1(y):
            b = bspline.basis(y, mode=0)
            return b
        
        I, info = integrate_gs_term(b1, 0.1, t, 100000, epsabs=1e-13, epsrel=0.0, order=31, max_ninter=50)
        self.assertEqual(I.shape, (7,))
        self.assertTrue(info.err < 1e-13)

    def test_001_integrate_r2_gs_basis(self):
        t = jnp.linspace(-0.5, 0.5, 5)
        tg = TensorGrid(t, t)
        bspline = BSpline(tg, degree=3)
        def b1(y):
            b = bspline.basis(y, mode=0)
            return b
        
        I, info = integrate_r2_gs_term(b1, 0.1, t, 1000000, epsabs=1e-13, epsrel=0.0, order=31, max_ninter=50)
        self.assertEqual(I.shape, (7,))
        self.assertTrue(info.err < 1e-13)


class TestSuperpotential(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_000_superpotential(self):
        t1 = jnp.linspace(-0.5, 0.5, 4)
        t2 = jnp.linspace(-0.5, 0.5, 5)
        t3 = jnp.linspace(-0.5, 0.5, 6)
        tg_spline = TensorGrid(t1, t2, t3)
        bspline = BSpline(tg_spline, degree=3)
        gs = GS(jnp.array([1.0, 1.0, 1.0, 1.0]), jnp.array([10, 1000, 10000, 1000000]))
        
        t1 = jnp.linspace(-0.5, 0.5, 10)
        t2 = jnp.linspace(-0.5, 0.5, 12)
        t3 = jnp.linspace(-0.5, 0.5, 14)

        tg_targets = TensorGrid(t1, t2, t3)

        factors, info = superpotential_factors(bspline, tg_targets, tg_spline, gs, epsabs=1e-13, epsrel=0.0, order=31)
        factor_shapes = (((4, 10, 6), (4, 12, 7), (4, 14, 8)), ((4, 10, 6), (4, 12, 7), (4, 14, 8)), ((4, 10, 6), (4, 12, 7), (4, 14, 8)))
        self.assertEqual(tree.map(lambda t: t.shape, factors), factor_shapes)
        self.assertTrue(info.err < 1e-13)

    def test_001_fit_superpotential(self):
        t1 = jnp.linspace(-0.5, 0.5, 10)
        t2 = jnp.linspace(-0.5, 0.5, 11)
        t3 = jnp.linspace(-0.5, 0.5, 12)
        tg_source = TensorGrid(t1, t2, t3)
        elm_source = BSpline(tg_source, degree=3)
        gs = GS(*sinc_quad_1_over_sqrtx(10))

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2), gauss(3)(t3))
        tg_quad = TensorGrid(*nodes, weights=weights)
        F = jnp.apply_along_axis(flower_state, -1, tg_quad.grid)
        
        core_source = fit(factors_pinv(elm_source.factors(tg_quad), tg_quad.weights), F)

        t1 = jnp.linspace(-0.5, 0.5, 7)
        t2 = jnp.linspace(-0.5, 0.5, 8)
        t3 = jnp.linspace(-0.5, 0.5, 9)
        elm_sp = BSpline(TensorGrid(t1, t2, t3), degree=3)
        
        inv_factors = factors_pinv(elm_sp.factors(tg_quad), tg_quad.weights)
        factors_superpot, info = superpotential_factors(elm_source, tg_quad, tg_source, gs, epsabs=1e-13, epsrel=0.0, order=31)
        self.assertTrue(info.err < 1e-13)

        core = fit_superpotential(inv_factors, factors_superpot, core_source, gs)
        self.assertEqual(core.shape, (9, 10, 11, 3))