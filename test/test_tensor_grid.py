from . import *

from tpelm.tensor_grid import TensorGrid


class TestTensorGrid(JaxTestCase):
    def test_000_init(self):
        k = random.key(0)
        t = random.uniform(k, (2, 100), minval=-1.0, maxval=1.0)
        tg = TensorGrid(*t)

        self.assertEqual(tg.dim, 2)
        self.assertPytreeEqual(tg[0], t[0])
        self.assertPytreeEqual(tg[1], t[1])
        self.assertPytreeEqual(tg.bounds[0], tuple(jnp.min(t, axis=1)))
        self.assertPytreeEqual(tg.bounds[1], tuple(jnp.max(t, axis=1)))
    
    def test_001_grid(self):
        t = jnp.linspace(0, 1, 50)
        tg = TensorGrid(t, t)

        self.assertPytreeEqual(tg.bounds[0], (0, 0))
        self.assertPytreeEqual(tg.bounds[1], (1, 1))

        grid = jnp.stack(jnp.meshgrid(t, t, indexing="ij"), axis=-1)
        self.assertPytreeEqual(tg.grid, grid)

    def test_002_jit(self):
        @jax.jit
        def foo(tg: TensorGrid):
            return jnp.prod(tg.grid, axis=-1)
        
        t = jnp.linspace(0, 1, 50)
        result = jnp.prod(jnp.stack(jnp.meshgrid(t, t, indexing="ij"), axis=-1), axis=-1)
        tg = TensorGrid(t, t)
        self.assertPytreeEqual(foo(tg), result)

    def test_003_weights(self):
        k = random.key(0)
        t = random.uniform(k, (2, 100), minval=-1.0, maxval=1.0)
        weights = jnp.ones_like(t) / 2
        tg = TensorGrid(*t, weights=weights)

        self.assertPytreeEqual(tg.weights[0], jnp.ones_like(t[0]) / 2)
        self.assertPytreeEqual(tg.weights[1], jnp.ones_like(t[1]) / 2)

    def test_004_weights(self):
        t = (
            jnp.linspace(0, 1, 100),
            jnp.linspace(0, 2, 50),
        )
        tg = TensorGrid(*t)

        self.assertPytreeEqual(tg.weights[0], jnp.ones((100,)))
        self.assertPytreeEqual(tg.weights[1], jnp.ones((50,)))

    def test_005_from_gauss(self):
        t = (
            jnp.linspace(0, 1, 101),
            jnp.linspace(0, 2, 51),
        )
        tg = TensorGrid.from_gauss(*t, degree=3)
        self.assertIsclose(jnp.sum(tg.weights[0]), 1.0)
        self.assertIsclose(jnp.sum(tg.weights[1]), 2.0)
        self.assertEqual(tg[0].shape, (100 * 3,))
        self.assertEqual(tg[1].shape, (50 * 3,))

