
from ho_stray_field.base import fit, regularized_pinv, fit_divergence, fit_laplace, fit_grad, factors_pinv
from ho_stray_field.bspline import basis, eval_spline, BSpline
from ho_stray_field.tensor_grid import TensorGrid
from ho_stray_field.tucker_tensor import TuckerTensor
from ho_stray_field.integrate import gauss

from . import *


class TestSplineBasis(JaxTestCase):
    def test_000_basis0(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.1, -1.0, -0.5, -0.1, 0.25, 1.0, 1.1])
        b = basis(x, t, 0)
        self.assertIsclose(
            b,
            jnp.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ]
            ),
        )

    def test_001_basis1(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = -1
        b = basis(x, t, 1)
        print(b)
        self.assertIsclose(b, jnp.array([1, 0, 0, 0, 0]))
        x = -2
        b = basis(x, t, 1)
        self.assertIsclose(b, jnp.array([0, 0, 0, 0, 0]))
        x = 0
        b = basis(x, t, 1)
        self.assertIsclose(b, jnp.array([0, 0, 1, 0, 0]))
        x = 1.0
        b = basis(x, t, 1)
        self.assertIsclose(b, jnp.array([0, 0, 0, 0, 1]))
        x = 2.0
        b = basis(x, t, 1)
        self.assertIsclose(b, jnp.array([0, 0, 0, 0, 0]))

    def test_002_basis2(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.0, -0.25, 0.25, 1.0])
        b = basis(x, t, 2)
        self.assertIsclose(
            b,
            jnp.array(
                [[1.,    0.,  0.,  0.,  0.,  0.],
                 [0.,   0.125, 0.75, 0.125, 0., 0.],
                 [0.,   0., 0.125, 0.75, 0.125, 0.],
                 [0.,   0., 0., 0., 0., 1.]]
            ),
        )

    def test_004_extrapolate(self):
        t = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = jnp.array([-1.1, -1.0, -0.25, 0.25, 1.0, 1.1])
        b = basis(x, t, 2, extrapolate=True)
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
        y = eval_spline(x, t, params, degree=3)
        self.assertIsclose(y, result)

    def test_006_eval_spline_closed(self):
        t = jnp.array([-1, 0.0, 1.0])
        params = jnp.array([0, 1, 0])
        x = jnp.linspace(-1, 1, 100)
        result = jnp.where(x < 0, 1 + x, 1 - x)
        y = eval_spline(x, t, params, degree=1)
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
        factors, d_factors = bspline.factors_and_partials(tg2)
        self.assertEqual(tree.map(lambda t: t.shape, d_factors), 
                         (((10, 5), (15, 6)), ((10, 5), (15, 6))))

        def basis1(x):
            return basis(x, t1_spline, degree=3)
        
        def basis2(x):
            return basis(x, t2_spline, degree=3)

        f1, f2 = factors
        b1 = jax.vmap(jax.jacfwd(basis1))(t1)
        b2 = jax.vmap(jax.jacfwd(basis2))(t2)
        self.assertPytreeEqual(d_factors, ((b1, f2), (f1, b2)))
        self.assertIsclose(factors, factors_true)


class TestTPELMFit(JaxTestCase):
    def test_001_fit_poly(self):
        f = lambda x: jnp.prod(x ** 3)
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        _factors_pinv = factors_pinv(bspline.factors(tg_quad))
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        core = fit(_factors_pinv, F)

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

        t1 = jnp.linspace(-jnp.pi, jnp.pi, 100)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 101)
        t3 = jnp.linspace(-jnp.pi, jnp.pi, 102)
        tg_quad = TensorGrid(t1, t2, t3)
        factors = bspline.factors(tg_quad)
        factors_pinv = tuple(regularized_pinv(X, tol=0.0) for X in factors)
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
        _factors_pinv = factors_pinv(bspline_result.factors(tg_quad))
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        f_core = fit(_factors_pinv, F)  # and compute the tucker tensor format of the fit

        t1 = jnp.linspace(-1, 1, 7)
        t2 = jnp.linspace(-1, 1, 8)
        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)  # Then fit a new B-spline to the tucker format
        _factors_pinv = factors_pinv(bspline.factors(tg_quad))
        f_factors = bspline_result.factors(tg_quad)
        F = TuckerTensor(f_core, f_factors)
        
        core = jax.jit(fit)(_factors_pinv, F)  # by using fit_to_tucker
        
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

        _factors_pinv = factors_pinv(bspline.factors(tg_basis))
        F = 42
        with self.assertRaises(NotImplementedError):
            fit(_factors_pinv, F)

    def test_005_fit_multivariate_function(self):
        f = lambda x: jnp.sin(x)
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        _factors_pinv = factors_pinv(bspline.factors(tg_quad))
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        core = fit(_factors_pinv, F)

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
        _inv_factors = factors_pinv(bspline_result.factors(tg_quad))
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        f_core = fit(_inv_factors, F)  # and compute the tucker tensor format of the fit

        t1 = jnp.linspace(-1, 1, 7)
        t2 = jnp.linspace(-1, 1, 8)
        weights, nodes = zip(gauss(3)(t1), gauss(3)(t2))
        tg_quad = TensorGrid(*nodes, weights=weights)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)  # Then fit a new B-spline to the tucker format
        _factors_pinv = factors_pinv(bspline.factors(tg_quad))
        f_factors = bspline_result.factors(tg_quad)
        F = TuckerTensor(f_core, f_factors)
        
        core = jax.jit(fit)(_factors_pinv, F)  # by using fit_to_tucker
        
        self.assertEqual(core.shape, (9, 10, 2))  # and check the result
        
        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        f_approx = bspline(tg_val, core)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertEqual(f_approx.shape, (20, 20, 2))
        self.assertEqual(f_true.shape, (20, 20, 2))
        self.assertIsclose(f_approx, f_true, atol=1e-3)
    
    def test_007_fit_poly_function(self):
        f = lambda x: jnp.prod(x ** 3) + x[0] * x[1] - x[0] ** 2 - 1
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        tg_quad = tg_basis.to_gauss(3)
        _inv_factors = factors_pinv(bspline.factors(tg_quad))
        core = fit(_inv_factors, f, tg_quad)

        t1 = jnp.linspace(-1, 1, 20)
        t2 = jnp.linspace(-1, 1, 20)
        tg_val = TensorGrid(t1, t2)
        f_approx = bspline(tg_val, core)
        f_true = jnp.apply_along_axis(f, -1, tg_val.grid)

        self.assertIsclose(f_approx, f_true, atol=1e-5)
    
    def test_008_fit_fails_with_type_error(self):
        f = lambda x: jnp.prod(x ** 3) + x[0] * x[1] - x[0] ** 2 - 1
        t1 = jnp.linspace(-1, 1, 10)
        t2 = jnp.linspace(-1, 1, 11)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=3)

        tg_quad = tg_basis.to_gauss(3)
        _inv_factors = factors_pinv(bspline.factors(tg_quad))
        F = jnp.apply_along_axis(f, -1, tg_quad.grid)
        with self.assertRaises(TypeError):
            fit(_inv_factors, F, tg_quad)

    def test_009_fit_func_tucker(self):
        pass


def _divergence(f):
    def div_f(*args, **kwargs):
        Jf = jax.jacfwd(f)(*args, **kwargs)

        def _compute_div(Jf):
            return jnp.sum(jnp.diag(Jf))

        return tree.map(_compute_div, Jf)

    return div_f


def hvp(f, primals, tangents):
    return jax.jvp(jax.jacfwd(f), primals, tangents)[1]


def hessian_diag(f, primals):
    primals = jnp.asarray(primals)
    vs = jnp.eye(primals.shape[0])

    def comp(v):
        return tree.map(lambda a: a @ v, hvp(f, [primals], [v]))

    diag_entries = jax.vmap(comp)(vs)
    return diag_entries


def laplace(f):
    def lap(x, *args, **kwargs):
        H_diag = hessian_diag(lambda x: f(x, *args, **kwargs), x)
        return tree.map(lambda d: jnp.sum(d, axis=0), H_diag)

    return lap


class TestDivergence(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_000_divergence(self):
        f = lambda x: jnp.array([jnp.prod(jnp.sin(x)), jnp.prod(jnp.cos(x))])
        t1 = jnp.linspace(-jnp.pi, jnp.pi, 60)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 60)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=5)

        tg_quad = tg_basis.to_gauss(4)
        inv_factors = factors_pinv(bspline.factors(tg_quad))
        core = fit(inv_factors, f, tg_quad)

        div_f = _divergence(f)
        div_F_true = jnp.apply_along_axis(div_f, -1, tg_quad.grid)

        factors, partials = bspline.factors_and_partials(tg_quad)
        core_div = fit_divergence(inv_factors, partials, core)
        div_F_pred = TuckerTensor(core_div, factors).to_tensor()
        self.assertIsclose(div_F_pred, div_F_true, atol=1e-7, rtol=0.0)

class TestLaplace(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_001_laplace(self):
        f = lambda x: jnp.prod(jnp.sin(x))
        t1 = jnp.linspace(-jnp.pi, jnp.pi, 70)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 70)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=6)

        tg_quad = tg_basis.to_gauss(4)
        inv_factors = factors_pinv(bspline.factors(tg_quad))
        core = fit(inv_factors, f, tg_quad)

        lap_f = laplace(f)
        lap_F_true = jnp.apply_along_axis(lap_f, -1, tg_quad.grid)

        factors, partials = bspline.factors_and_partials(tg_quad)
        core_div = fit_laplace(inv_factors, partials, core)
        lap_F_pred = TuckerTensor(core_div, factors).to_tensor()
        self.assertIsclose(lap_F_pred, lap_F_true, atol=1e-7, rtol=0.0)


class TestGrad(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_001_laplace(self):
        f = lambda x: jnp.prod(jnp.sin(x))
        t1 = jnp.linspace(-jnp.pi, jnp.pi, 70)
        t2 = jnp.linspace(-jnp.pi, jnp.pi, 70)
        tg_basis = TensorGrid(t1, t2)
        bspline = BSpline(tg_basis, degree=6)

        tg_quad = tg_basis.to_gauss(4)
        inv_factors = factors_pinv(bspline.factors(tg_quad))
        core = fit(inv_factors, f, tg_quad)

        grad_f = jax.jacfwd(f)
        grad_F_true = jnp.apply_along_axis(grad_f, -1, tg_quad.grid)

        factors, partials = bspline.factors_and_partials(tg_quad)
        core_div = fit_grad(inv_factors, partials, core)
        grad_F_pred = TuckerTensor(core_div, factors).to_tensor()
        self.assertIsclose(grad_F_pred, grad_F_true, atol=1e-7, rtol=0.0)
