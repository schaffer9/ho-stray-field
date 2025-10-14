
from tpelm.bspline import basis, eval_spline, BSpline, fit, regularized_pinv
from tpelm.tensor_grid import TensorGrid
from tpelm.tucker_tensor import TuckerTensor
from tpelm.integrate import gauss

from . import *


class TestSplineBasis(JaxTestCase):
    def test_000_basis0(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.1, -1.0, -0.5, -0.1, 0.25, 1.0, 1.1])
        b = basis(x, t, 0, open_spline=True)
        self.assertIsclose(
            b,
            jnp.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ),
        )

    def test_001_basis1(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = -1
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, jnp.array([0, 0, 0]))
        x = -2
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, jnp.array([0, 0, 0]))
        x = 0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, jnp.array([0, 1, 0]))
        x = 1.0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, jnp.array([0, 0, 0]))
        x = 2.0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, jnp.array([0, 0, 0]))

    def test_002_basis2(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.0, -0.25, 0.25, 1.0])
        b = basis(x, t, 2, open_spline=True)
        self.assertIsclose(
            b,
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )

    def test_003_vectorized(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        t = jnp.asarray([t, t])
        x = jnp.array([-1.0, -0.25, 0.25, 1.0])
        x = jnp.asarray([x, x]).T
        b = basis(x, t, 2, open_spline=True)
        self.assertIsclose(
            b[:, 0],
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )
        self.assertIsclose(
            b[:, 1],
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )

    def test_004_closed_spline(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.1, -1.0, -0.25, 0.25, 1.0, 1.1])
        b = basis(x, t, 2, open_spline=False)
        result = jnp.array(
            [
                [1.44, -0.46000013, 0.02000001, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.125, 0.75, 0.125, 0.0, 0.0],
                [0.0, 0.0, 0.125, 0.75, 0.125, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.02000001, -0.46000013, 1.44],
            ]
        )
        self.assertIsclose(b, result)

    def test_005_eval_spline(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        params = jnp.array([1, 1, 1, 1, 1, 1, 1])
        x = jnp.linspace(-1, 1, 100)
        result = jnp.ones((100,))
        y = eval_spline(x, t, params, degree=3, open_spline=False)
        self.assertIsclose(y, result)

    def test_006_eval_spline_closed(self):
        t = jnp.array([-1, 0.0, 1.0])
        params = jnp.array([1, ])
        x = jnp.linspace(-1, 1, 100)
        result = jnp.where(x < 0, 1 + x, 1 - x)
        y = eval_spline(x, t, params, degree=1, open_spline=True)
        self.assertIsclose(y, result)


class TestBSpline(JaxTestCase):
    def test_000_tensor_product_elm(self):
        t = jnp.linspace(-0.5, 0.5, 5)
        tg = TensorGrid(t, t)
        bspline = BSpline(tg, degree=3)
        b = bspline.factors(tg)
        
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0].shape, (len(t), len(t) + 2))
        self.assertEqual(b[1].shape, (len(t), len(t) + 2))

    def test_001_elementwise_derivative(self):
        t1_spline = jnp.linspace(-0.5, 0.5, 3)
        t2_spline = jnp.linspace(-0.5, 0.5, 4)
        tg = TensorGrid(t1_spline, t2_spline)
        bspline = BSpline(tg, degree=3)
        t1 = jnp.linspace(-0.5, 0.5, 10)
        t2 = jnp.linspace(-0.5, 0.5, 15)
        tg2 = TensorGrid(t1, t2)
        factors_true = bspline.factors(tg2)
        factors, d_factors = bspline.factors_derivative(tg2)
        self.assertEqual(d_factors[0].shape, (10, 5))
        self.assertEqual(d_factors[1].shape, (15, 6))

        def basis1(x):
            return basis(x, t1_spline, degree=3)
        
        def basis2(x):
            return basis(x, t2_spline, degree=3)

        b1 = jax.vmap(jax.jacfwd(basis1))(t1)
        b2 = jax.vmap(jax.jacfwd(basis2))(t2)
        self.assertPytreeEqual(d_factors, (b1, b2))
        self.assertPytreeEqual(factors, factors_true)


class TestTPELMFit(JaxTestCase):
    def test_001_fit_poly(self):
        f = lambda x: jnp.prod(x ** 3)
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        factors_pinv = bspline.factors_pinv(tg_quad)
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        core = fit(factors_pinv, F)

        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        factors = bspline.factors(tg_val)
        f_approx = jnp.einsum("ab,ia,jb->ij", core, *factors)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertIsclose(f_approx, f_true, atol=1e-5)

    def test_002_fit_sin(self):
        f = lambda x: jnp.prod(x ** 3)
        t1 = jnp.linspace(-jnp.pi, jnp.pi, 10)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 11)
        t3 = jnp.linspace(-jnp.pi, jnp.pi, 12)
        tg_basis = TensorGrid(t1, t2, t3)
        bspline = BSpline(tg_basis, degree=3)

        t1 = jnp.linspace(-jnp.pi, jnp.pi, 70)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 71)
        t3 = jnp.linspace(-jnp.pi, jnp.pi, 72)
        tg_quad = TensorGrid(t1, t2, t3)
        factors = bspline.factors(tg_quad)
        factors_pinv = tuple(regularized_pinv(X) for X in factors)
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        core = fit(factors_pinv, F)

        t1 = jnp.linspace(-jnp.pi, jnp.pi, 20)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 20)
        t3 = jnp.linspace(-jnp.pi, jnp.pi, 20)
        tg_val = TensorGrid(t1, t2, t3)
        factors = bspline.factors(tg_val)
        f_approx = jnp.einsum("abc,ia,jb,kc->ijk", core, *factors)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)
        self.assertIsclose(f_approx, f_true, atol=1e-2)

    def test_003_fit_to_tucker(self):
        f = lambda x: jnp.prod(x ** 3)  # define the function
        t1 = jnp.linspace(-1, 1, 5)
        t2 = jnp.linspace(-1, 1, 6)
        tg_basis = TensorGrid(t1, t2)
        bspline_result = BSpline(tg_basis, degree=3)  # then fit a B-spline to the function 

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        factors_pinv = bspline_result.factors_pinv(tg_quad)
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        f_core = fit(factors_pinv, F)  # and compute the tucker tensor format of the fit

        t1 = jnp.linspace(-1, 1, 7)
        t2 = jnp.linspace(-1, 1, 8)
        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)  # Then fit a new B-spline to the tucker format
        factors_pinv = bspline.factors_pinv(tg_quad)
        f_factors = bspline_result.factors(tg_quad)
        F = TuckerTensor(f_core, f_factors)
        
        core = jax.jit(fit)(factors_pinv, F)  # by using fit_to_tucker
        
        self.assertEqual(core.shape, (9, 10))  # and check the result
        
        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        f_approx = bspline(tg_val, core)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertIsclose(f_approx, f_true, atol=1e-5)

    def test_004_not_implemented(self):
        t1 = jnp.linspace(-1, 1, 10)
        tg_basis = TensorGrid(t1)
        bspline = BSpline(tg_basis, degree=3)

        factors_pinv = bspline.factors_pinv(tg_basis)
        F = 42
        with self.assertRaises(NotImplementedError):
            fit(factors_pinv, F)

    def test_005_fit_multivariate_function(self):
        f = lambda x: jnp.sin(x)
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        factors_pinv = bspline.factors_pinv(tg_quad)
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        core = fit(factors_pinv, F)

        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        factors = bspline.factors(tg_val)
        f_approx = jnp.einsum("abd,ia,jb->ijd", core, *factors)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertEqual(f_approx.shape, (20, 20, 2))
        self.assertEqual(f_true.shape, (20, 20, 2))
        self.assertIsclose(f_approx, f_true, atol=1e-3)

    def test_006_fit_multivariate_function_from_tucker(self):
        f = lambda x: jnp.sin(x)  # define the function
        t1 = jnp.linspace(-1, 1, 5)
        t2 = jnp.linspace(-1, 1, 6)
        tg_basis = TensorGrid(t1, t2)
        bspline_result = BSpline(tg_basis, degree=3)  # then fit a B-spline to the function 

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        factors_pinv = bspline_result.factors_pinv(tg_quad)
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        f_core = fit(factors_pinv, F)  # and compute the tucker tensor format of the fit

        t1 = jnp.linspace(-1, 1, 7)
        t2 = jnp.linspace(-1, 1, 8)
        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)  # Then fit a new B-spline to the tucker format
        factors_pinv = bspline.factors_pinv(tg_quad)
        f_factors = bspline_result.factors(tg_quad)
        F = TuckerTensor(f_core, f_factors)
        
        core = jax.jit(fit)(factors_pinv, F)  # by using fit_to_tucker
        
        self.assertEqual(core.shape, (9, 10, 2))  # and check the result
        
        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        f_approx = bspline(tg_val, core)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertEqual(f_approx.shape, (20, 20, 2))
        self.assertEqual(f_true.shape, (20, 20, 2))
        self.assertIsclose(f_approx, f_true, atol=1e-3)
    
