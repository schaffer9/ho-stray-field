from ho_stray_field.integrate import gauss, sinc_quad, sinc_quad_1_over_sqrtx

from . import *



class TestGauss(JaxTestCase):
    def test_000_gauss_3(self):
        weights, nodes = gauss(3)(jnp.array([-1, 0, 1]))
        self.assertEqual(weights.shape, (6,))
        self.assertEqual(nodes.shape, (6,))
        self.assertPytreeEqual(jnp.sum(weights), 2)


class TestSincQuad(JaxTestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", False)

    def test_000_shape(self):
        n, c0 = 10, 1.0
        w, x = sinc_quad(n, c0)
        w, x = jnp.array(w, dtype=jnp.float64), jnp.array(x, dtype=jnp.float64)
        assert w.shape == (n,)
        assert x.shape == (n,)
        assert jnp.all(w > 0), "All weights should be positive"
        assert jnp.all(jnp.diff(x) > 0), "Nodes should be monotonically increasing"

    def test_001_sinc_quad_integral_accuracy(self):
        f = jax.jit(lambda x: jnp.exp(-x**2))
        n, c0 = 100, 1.0
        w, x = sinc_quad(n, c0)
        w, x = jnp.array(w, dtype=jnp.float64), jnp.array(x, dtype=jnp.float64)
        I = jnp.sum(w * f(x))
        ref = jnp.sqrt(jnp.pi)
        err = jnp.abs(I - ref)
        assert err < 1e-3, f"Integral too inaccurate: got {I}, expected {ref}"

    def test_002_sinc_quad_1_over_r(self):
        omega, alpha = sinc_quad_1_over_sqrtx(200, 1.9)
        omega, alpha = jnp.array(omega, dtype=jnp.float64), jnp.array(alpha, dtype=jnp.float64)
        r = jnp.logspace(1e-3, 1e0, 1000)

        def exp_sum(r, omega, alpha):
            return omega @ jnp.exp(-r ** 2 * alpha)
        
        err = jnp.abs(1 / r - jax.vmap(exp_sum, (0, None, None))(r, omega, alpha))
        self.assertTrue(jnp.all(err < 1e-10))

    def test_003_sinc_quad_r(self):
        omega, alpha = sinc_quad_1_over_sqrtx(200, 1.9)
        omega, alpha = jnp.array(omega, dtype=jnp.float64), jnp.array(alpha, dtype=jnp.float64)
        r = jnp.logspace(1e-3, 1e0, 1000)

        def exp_sum(r, omega, alpha):
            return r**2 * omega @ jnp.exp(-r ** 2 * alpha)
        
        err = jnp.abs(r - jax.vmap(exp_sum, (0, None, None))(r, omega, alpha))
        self.assertTrue(jnp.all(err < 1e-14))