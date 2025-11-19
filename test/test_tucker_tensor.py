from . import *

from ho_stray_field.tucker_tensor import TuckerTensor


class TestTuckerTensor(JaxTestCase):
    def test_000_dot(self):
        key = random.key(0)
        key, k1, k2, k3, k4 = random.split(key, 5)
        A = TuckerTensor(
            random.uniform(k1, (10, 11, 12, 2)),
            (
                random.uniform(k2, (20, 10)),
                random.uniform(k3, (20, 11)),
                random.uniform(k4, (20, 12))
            )
        )
        key, k1, k2, k3, k4 = random.split(key, 5)
        B = TuckerTensor(
            random.uniform(k1, (5, 6, 7, 2)),
            (
                random.uniform(k2, (20, 5)),
                random.uniform(k3, (20, 6)),
                random.uniform(k4, (20, 7))
            )
        )
        
        result = jnp.tensordot(A.to_tensor(), B.to_tensor(), ((0,1,2,3), (0,1,2,3)))
        dot_result = A.dot(B)
        self.assertIsclose(result, dot_result)
        
    